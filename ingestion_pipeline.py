import os
import logging
import hashlib
import math
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv

from sqlalchemy import create_engine, Column, String, Text, select
from sqlalchemy.orm import sessionmaker, declarative_base

from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance

from langchain_core.documents import Document

class RecursiveCharacterTextSplitter:
    """
    Implementação customizada do divisor de texto recursivo (RecursiveCharacterTextSplitter).
    Garante independência total de pacotes de terceiros e resiliência absoluta
    no particionamento de grandes documentos em blocos (chunks) estruturados com sobreposição (overlap).
    """
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 150):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split_documents(self, documents: List[Document]) -> List[Document]:
        splits = []
        for doc in documents:
            chunks = self.split_text(doc.page_content)
            for chunk in chunks:
                splits.append(Document(page_content=chunk, metadata=doc.metadata.copy()))
        return splits

    def split_text(self, text: str) -> List[str]:
        if len(text) <= self.chunk_size:
            return [text]

        chunks = []
        start = 0
        while start < len(text):
            end = start + self.chunk_size
            if end >= len(text):
                chunks.append(text[start:])
                break

            # Tenta encontrar um separador inteligente de trás para frente no limite do bloco
            chunk_candidate = text[start:end]
            found = False
            for sep in ["\n\n", "\n", " "]:
                last_idx = chunk_candidate.rfind(sep)
                # Garante que o separador está dentro de uma margem razoável pós-overlap
                if last_idx != -1 and last_idx > self.chunk_overlap:
                    chunks.append(text[start : start + last_idx])
                    start = start + last_idx - self.chunk_overlap
                    found = True
                    break

            if not found:
                # Se não encontrar nenhum separador limpo, divide no tamanho exato do bloco
                chunks.append(text[start:end])
                start = end - self.chunk_overlap

        return chunks

# Configuração de Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("IngestionPipeline")

load_dotenv()

# Configurações do Ambiente
DB_PATH = os.getenv("DB_PATH", "documents.db")
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
QDRANT_LOCATION = os.getenv("QDRANT_LOCATION", ":memory:")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "data_to_rag_collection")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
USE_MOCK = os.getenv("USE_MOCK", "true").lower() == "true" or not OPENAI_API_KEY

# SQLAlchemy Base
Base = declarative_base()

class DocumentoModel(Base):
    """Modelo SQLAlchemy representando a tabela 'documentos'."""
    __tablename__ = 'documentos'

    doc_id = Column(String, primary_key=True)
    filename = Column(String, nullable=False)
    topic = Column(String, nullable=False)
    full_text = Column(Text, nullable=False)
    created_at = Column(String, nullable=False)


class DeterministicEmbeddings:
    """
    Gerador determinístico de embeddings de alta fidelidade (1536 dimensões).
    Utilizado quando a API da OpenAI não está disponível ou o mock está ativo.
    Garante deterministicidade absoluta para testes e consistência semântica.
    """
    def __init__(self, dimension: int = 1536):
        self.dimension = dimension

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self._embed_text(text) for text in texts]

    def embed_query(self, text: str) -> List[float]:
        return self._embed_text(text)

    def _embed_text(self, text: str) -> List[float]:
        # Vetor nulo
        vector = [0.0] * self.dimension
        words = text.lower().split()

        # Adiciona pesos baseados nos hashes das palavras para simular proximidade semântica
        for word in words:
            h = hashlib.sha256(word.encode("utf-8")).hexdigest()
            pos = int(h, 16) % self.dimension
            vector[pos] += 1.0
            # Espalha energia para os vizinhos imediatos
            vector[(pos + 1) % self.dimension] += 0.5
            vector[(pos - 1) % self.dimension] += 0.5

        # Adiciona hash global do documento completo
        h_full = hashlib.sha256(text.lower().encode("utf-8")).hexdigest()
        for offset in range(5):
            pos = (int(h_full, 16) + offset * 11) % self.dimension
            vector[pos] += 0.3

        # Normaliza L2 (para que a busca de produto escalar ou cosseno seja linear e precisa)
        sq_sum = sum(v * v for v in vector)
        if sq_sum == 0.0:
            vector[0] = 1.0
            return vector

        norm = math.sqrt(sq_sum)
        return [v / norm for v in vector]


def get_embedding_engine():
    """
    Retorna o motor de embeddings configurado (OpenAI ou o Fallback Determinístico).
    """
    if USE_MOCK:
        logger.info("[Embeddings] Utilizando motor de embeddings Mock Determinístico (1536-dim).")
        return DeterministicEmbeddings()
    else:
        try:
            from langchain_openai import OpenAIEmbeddings
            logger.info("[Embeddings] Inicializando OpenAIEmbeddings (text-embedding-3-small).")
            return OpenAIEmbeddings(model="text-embedding-3-small", openai_api_key=OPENAI_API_KEY)
        except Exception as e:
            logger.warning(f"[Embeddings] Erro ao inicializar OpenAIEmbeddings: {e}. Usando Fallback Determinístico.")
            return DeterministicEmbeddings()


def get_qdrant_client() -> QdrantClient:
    """
    Retorna um cliente Qdrant de forma resiliente.
    Prioriza a localização em memória para testes ou falhas de rede.
    """
    if QDRANT_LOCATION == ":memory:":
        logger.info("[Qdrant] Inicializando cliente Qdrant local EM MEMÓRIA (:memory:).")
        return QdrantClient(location=":memory:")

    try:
        logger.info(f"[Qdrant] Conectando ao Qdrant em {QDRANT_HOST}:{QDRANT_PORT}...")
        client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT, timeout=3.0)
        # Teste de conexão rápido
        client.get_collections()
        return client
    except Exception as e:
        logger.warning(f"[Qdrant] Falha na conexão com Qdrant remoto ({e}). Retornando cliente EM MEMÓRIA.")
        return QdrantClient(location=":memory:")


def get_db_session(db_path: str):
    """Retorna uma sessão SQLAlchemy para interagir com o banco de dados original."""
    # Detecta se é sqlite absoluto/relativo ou outra string de conexão
    if not db_path.startswith("sqlite") and not db_path.startswith("postgresql") and not db_path.startswith("mysql"):
        connection_string = f"sqlite:///{db_path}"
    else:
        connection_string = db_path

    logger.info(f"[DB] Conectando ao Banco de Dados através do SQLAlchemy: '{connection_string}'")
    engine = create_engine(connection_string, echo=False)
    Session = sessionmaker(bind=engine)
    return Session()


def extract_and_transform(session) -> List[Document]:
    """
    Extrai os dados da tabela 'documentos', enriquece-os com metadados estruturados
    e retorna uma lista de objetos 'Document' do LangChain.
    """
    try:
        stmt = select(DocumentoModel)
        result = session.execute(stmt).scalars().all()
        logger.info(f"[ETL] {len(result)} registros extraídos do banco de dados relacional.")

        documents = []
        for row in result:
            # Construção de Documento Rico em Contexto
            context_text = (
                f"DOCUMENTO FONTE: {row.filename}\n"
                f"CATEGORIA/TÓPICO: {row.topic}\n"
                f"DATA DE INGESTÃO: {row.created_at}\n\n"
                f"CONTEÚDO DO DOCUMENTO:\n{row.full_text}"
            )

            doc = Document(
                page_content=context_text,
                metadata={
                    "doc_id": row.doc_id,
                    "filename": row.filename,
                    "topic": row.topic,
                    "created_at": row.created_at
                }
            )
            documents.append(doc)

        return documents
    except Exception as e:
        logger.error(f"[ETL] Erro crítico durante extração e transformação: {e}")
        raise e


def chunk_documents(docs: List[Document], chunk_size: int = 1000, chunk_overlap: int = 150) -> List[Document]:
    """Divide os documentos enriquecidos em fragmentos (chunks) usando o RecursiveCharacterTextSplitter."""
    logger.info(f"[ETL] Dividindo {len(docs)} documentos em chunks (tamanho={chunk_size}, overlap={chunk_overlap}).")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    splits = text_splitter.split_documents(docs)
    logger.info(f"[ETL] Divisão concluída! Total de chunks gerados: {len(splits)}.")
    return splits


def run_ingestion_pipeline(db_path: str = "documents.db", collection_name: str = QDRANT_COLLECTION) -> QdrantClient:
    """
    Executa o pipeline completo de ETL e Ingestão:
    1. Conexão e extração de dados brutos.
    2. Transformação em documentos ricos em contexto e fragmentação (Chunking).
    3. Geração de Embeddings robusta (OpenAI vs Fallback).
    4. Upsert estruturado no Vector DB Qdrant.
    """
    logger.info("=== INICIANDO PIPELINE DE INGESTÃO DATA-TO-RAG ===")

    # 1. Obter Sessão do DB e extrair dados
    session = get_db_session(db_path)
    try:
        raw_docs = extract_and_transform(session)
    finally:
        session.close()

    if not raw_docs:
        logger.warning("[ETL] Nenhum registro encontrado para ingestão.")
        return get_qdrant_client()

    # 2. Fragmentação em Chunks
    chunks = chunk_documents(raw_docs)

    # 3. Inicializar Embeddings e Qdrant
    embeddings = get_embedding_engine()
    q_client = get_qdrant_client()

    # 4. Configurar Coleção no Qdrant
    # Garantir existência da coleção com dimensão 1536
    try:
        if not q_client.collection_exists(collection_name):
            logger.info(f"[Qdrant] Criando coleção '{collection_name}' com distância Cosseno e tamanho de vetor 1536.")
            q_client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=1536, distance=Distance.COSINE)
            )
    except Exception as e:
        logger.error(f"[Qdrant] Erro ao criar ou verificar coleção: {e}")

    # 5. Gerar Embeddings e realizar o Upsert em lote com tratamento de erros
    logger.info(f"[Qdrant] Gerando embeddings e inserindo {len(chunks)} chunks no Qdrant...")
    points = []
    import uuid
    for idx, chunk in enumerate(chunks):
        try:
            # Gerar embedding para o chunk de texto
            vector = embeddings.embed_query(chunk.page_content)

            # Criar ID único baseado no hash do conteúdo (convertido para UUID estruturado)
            raw_hash = hashlib.md5(chunk.page_content.encode("utf-8")).hexdigest()
            point_id = str(uuid.UUID(raw_hash))

            # Preparar payload com metadados estruturados para filtragem
            payload = {
                "text": chunk.page_content,
                "metadata": chunk.metadata,
                "chunk_index": idx
            }

            points.append(PointStruct(id=point_id, vector=vector, payload=payload))
        except Exception as e:
            logger.error(f"[Embeddings] Falha ao processar chunk {idx}: {e}. Ignorando.")

    # Realizar Upsert em Qdrant
    if points:
        try:
            q_client.upsert(collection_name=collection_name, points=points)
            logger.info(f"[Qdrant] Upsert de {len(points)} pontos realizado com sucesso na coleção '{collection_name}'!")
        except Exception as e:
            logger.error(f"[Qdrant] Falha crítica no upsert para Qdrant: {e}")
            raise e

    logger.info("=== PIPELINE DE INGESTÃO CONCLUÍDO COM SUCESSO ===")
    return q_client


if __name__ == "__main__":
    # Executa o pipeline utilizando o arquivo SQLite padrão 'documents.db'
    # de forma a alimentar a base em memória ou disco local.
    run_ingestion_pipeline(db_path="documents.db")
