import os
import logging
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field
from dotenv import load_dotenv

from ingestion_pipeline import run_ingestion_pipeline, get_qdrant_client, QDRANT_COLLECTION
from rag_agent import get_agent

# Configuração de Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("DataToRAGMain")

load_dotenv()

app = FastAPI(
    title="Data-to-RAG Agent API",
    description="API empresarial para Ingestão e Atendimento Agentivo utilizando SQLAlchemy, Qdrant e LangGraph.",
    version="1.0.0"
)


# ---------------------------------------------------------------------------
# MODELOS DE DADOS (Pydantic)
# ---------------------------------------------------------------------------
class ChatRequest(BaseModel):
    message: str = Field(..., description="A pergunta ou mensagem enviada pelo usuário para o agente.")


class ChatResponse(BaseModel):
    response: str = Field(..., description="A resposta detalhada, estruturada e fundamentada fornecida pelo agente.")


class IngestResponse(BaseModel):
    status: str = Field(..., description="Status do processo de ingestão.")
    message: str = Field(..., description="Mensagem informativa detalhando o resultado da operação.")


# ---------------------------------------------------------------------------
# EVENTOS DE INICIALIZAÇÃO
# ---------------------------------------------------------------------------
@app.on_event("startup")
def on_startup():
    """Verifica e executa automaticamente a ingestão de dados se a coleção do Qdrant estiver vazia."""
    logger.info("[Startup] Verificando existência de dados na coleção do Qdrant...")
    try:
        client = get_qdrant_client()
        if not client.collection_exists(QDRANT_COLLECTION):
            logger.info(f"[Startup] Coleção '{QDRANT_COLLECTION}' não encontrada. Executando ingestão inicial do banco relacional...")
            run_ingestion_pipeline()
        else:
            # Coleção já existe, vamos verificar se possui pontos
            info = client.get_collection(QDRANT_COLLECTION)
            if info.points_count == 0:
                logger.info(f"[Startup] Coleção '{QDRANT_COLLECTION}' existe mas está vazia. Executando ingestão...")
                run_ingestion_pipeline()
            else:
                logger.info(f"[Startup] Coleção '{QDRANT_COLLECTION}' carregada e pronta com {info.points_count} documentos indexados!")
    except Exception as e:
        logger.error(f"[Startup] Falha ao inicializar banco de dados vetorial ou executar ingestão no startup: {e}")


# ---------------------------------------------------------------------------
# ENDPOINTS REST API
# ---------------------------------------------------------------------------
@app.get("/")
def read_root():
    """Retorna informações gerais sobre o estado da API."""
    return {
        "app": "Data-to-RAG Agent API",
        "status": "online",
        "collection_monitored": QDRANT_COLLECTION,
        "openai_configured": bool(os.getenv("OPENAI_API_KEY"))
    }


@app.post("/chat", response_model=ChatResponse, status_code=status.HTTP_200_OK)
async def chat_endpoint(request: ChatRequest):
    """
    Endpoint para interagir com o agente de IA conversacional.
    O agente interpreta a pergunta, executa busca vetorial semântica no Qdrant,
    e retorna uma resposta rica e fundamentada com citações das fontes.
    """
    logger.info(f"[API] Recebida solicitação de chat: '{request.message}'")
    try:
        # Recuperar o executor do agente
        agent_executor = get_agent()

        # Invocação assíncrona do agente através do grafo do LangGraph
        # Passa a entrada do usuário como mensagem inicial
        inputs = {"messages": [("user", request.message)]}

        # Executar o fluxo
        result = await agent_executor.ainvoke(inputs)

        # Recuperar a última mensagem de IA que contém a resposta final sintetizada
        messages = result.get("messages", [])
        if not messages:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="O agente de IA não retornou mensagens de resposta."
            )

        final_message = messages[-1]
        final_text = final_message.content if hasattr(final_message, "content") else str(final_message)

        return ChatResponse(response=final_text)

    except Exception as e:
        logger.error(f"[API] Erro ao processar mensagem no chat: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Erro interno ao invocar o agente conversacional: {str(e)}"
        )


@app.post("/ingest", response_model=IngestResponse, status_code=status.HTTP_201_CREATED)
def force_ingest():
    """
    Força a execução manual do pipeline de ETL / ingestão.
    Lê os dados da tabela relacional original e reconstrói o índice vetorial no Qdrant.
    """
    logger.info("[API] Executando pipeline de ingestão sob demanda...")
    try:
        run_ingestion_pipeline()
        return IngestResponse(
            status="success",
            message=f"Pipeline de ingestão executado com sucesso! Dados sincronizados com a coleção '{QDRANT_COLLECTION}'."
        )
    except Exception as e:
        logger.error(f"[API] Falha ao forçar ingestão manual: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Falha ao executar pipeline de ingestão: {str(e)}"
        )
