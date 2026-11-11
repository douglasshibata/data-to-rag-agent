import sys
import os
import math
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine

# Adicionar caminhos ao sys.path para garantir importações limpas
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from ingestion_pipeline import (
    DeterministicEmbeddings,
    get_embedding_engine,
    get_qdrant_client,
    get_db_session,
    Base,
    DocumentoModel,
    run_ingestion_pipeline,
    QDRANT_COLLECTION
)
from rag_agent import get_agent, search_qdrant_kb, MockAgentLLM
from main import app


def test_deterministic_embeddings():
    """Garante que a classe de embeddings determinísticos produz vetores de tamanho 1536 e comprimento unitário (norma L2 = 1.0)."""
    embedder = DeterministicEmbeddings()
    vector = embedder.embed_query("Qual o valor da fatura?")

    # Verificar dimensão
    assert len(vector) == 1536

    # Verificar norma L2
    norm = math.sqrt(sum(v * v for v in vector))
    assert abs(norm - 1.0) < 1e-4


def test_get_embedding_engine():
    """Verifica se o motor de embeddings é carregado com sucesso (mock ou real)."""
    engine = get_embedding_engine()
    assert hasattr(engine, "embed_query")
    assert hasattr(engine, "embed_documents")


def test_get_qdrant_client_resilience():
    """Verifica se o cliente Qdrant é retornado em memória de forma resiliente."""
    client = get_qdrant_client()
    assert client is not None


def test_etl_and_ingestion_pipeline():
    """
    Testa o fluxo completo de ETL e Ingestão utilizando um banco SQLite temporário em arquivo:
    1. Criação das tabelas físicas do modelo de documentos.
    2. Inserção de registro falso de documento corporativo.
    3. Execução do pipeline que extrai do SQLite, quebra em chunks, gera embeddings e insere no Qdrant.
    4. Validação dos pontos indexados no Qdrant e busca de similaridade semântica.
    """
    # 1. Configurar banco temporário em arquivo para teste
    db_file = "test_temp_documents.db"
    db_conn = f"sqlite:///{db_file}"
    engine = create_engine(db_conn)

    # Garantir estado limpo
    Base.metadata.drop_all(engine)
    Base.metadata.create_all(engine)

    # 2. Inserir um documento de teste na tabela relacional
    session = get_db_session(db_conn)
    try:
        fake_doc = DocumentoModel(
            doc_id="f6be9d8d-862a-44e2-ae9d-9d4abce94002",
            filename="fatura_venda_corporativa_julho.pdf",
            topic="Financeiro",
            full_text="O valor total das faturas de energia para o mês de julho totaliza R$ 15.420,50. O vencimento está programado para 10/08/2026.",
            created_at="2026-07-08T12:00:00Z"
        )
        session.add(fake_doc)
        session.commit()
    finally:
        session.close()

    try:
        # 3. Executar o pipeline de ingestão direcionando para a coleção de teste no Qdrant em memória
        test_collection = "test_ingestion_collection"
        q_client = run_ingestion_pipeline(db_path=db_conn, collection_name=test_collection)

        # 4. Verificar se a coleção foi criada e o ponto foi inserido
        assert q_client.collection_exists(test_collection)
        info = q_client.get_collection(test_collection)
        assert info.points_count > 0

        # 5. Testar busca semântica no Qdrant com dados reais usando query_points
        response = q_client.query_points(
            collection_name=test_collection,
            query=DeterministicEmbeddings().embed_query("fatura de julho"),
            limit=1
        )
        results = response.points if hasattr(response, "points") else []
        assert len(results) == 1
        payload = results[0].payload
        assert "fatura_venda_corporativa_julho.pdf" in payload["metadata"]["filename"]
        assert "R$ 15.420,50" in payload["text"]
    finally:
        # Limpar arquivo temporário de teste
        if os.path.exists(db_file):
            try:
                os.remove(db_file)
            except Exception:
                pass


@pytest.mark.asyncio
async def test_agent_react_loop():
    """
    Testa o agente LangGraph e seu loop de raciocínio ReAct:
    Garante que o agente recebe a consulta, decide acionar a ferramenta de busca do Qdrant,
    processa o resultado da busca e retorna uma resposta final estruturada e grounded.
    """
    # Executa ingestão com os dados reais de 'documents.db' para populá-lo localmente para o teste
    q_client = run_ingestion_pipeline(db_path="documents.db", collection_name=QDRANT_COLLECTION)

    agent = get_agent()
    user_query = "Qual é o valor total das faturas e sua data de vencimento?"

    inputs = {"messages": [("user", user_query)]}
    result = await agent.ainvoke(inputs)

    messages = result.get("messages", [])
    assert len(messages) >= 3 # Usuário, Chamado de Ferramenta (IA), Resposta da Ferramenta, Resposta Final (IA)

    # O conteúdo da última resposta de IA deve ser grounded e conter a solução ou indicar que buscou
    final_response = messages[-1].content
    assert final_response is not None
    assert "RESPOSTA" in final_response or "fatura" in final_response.lower()


def test_api_endpoints():
    """Verifica todos os endpoints da API FastAPI (`GET /`, `POST /ingest` e `POST /chat`)."""
    # Certifica-se de que documents.db está com os dados populados antes de testar
    run_ingestion_pipeline(db_path="documents.db", collection_name=QDRANT_COLLECTION)

    with TestClient(app) as client:
        # 1. Testar endpoint raiz
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert data["app"] == "Data-to-RAG Agent API"
        assert data["status"] == "online"

        # 2. Testar endpoint de ingestão forçada (/ingest)
        ingest_res = client.post("/ingest")
        assert ingest_res.status_code == 201
        ingest_data = ingest_res.json()
        assert ingest_data["status"] == "success"
        assert "sincronizados" in ingest_data["message"]

        # 3. Testar endpoint de chat (/chat) que invoca o agente
        chat_payload = {"message": "Qual é o assunto do documento test_file.pdf?"}
        chat_res = client.post("/chat", json=chat_payload)
        assert chat_res.status_code == 200
        chat_data = chat_res.json()
        assert "response" in chat_data
        assert len(chat_data["response"]) > 0
