import os
import logging
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage, AIMessage, ToolCall, ToolMessage
from langchain_core.outputs import ChatResult, ChatGeneration
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent

from ingestion_pipeline import get_qdrant_client, get_embedding_engine, QDRANT_COLLECTION, USE_MOCK

logger = logging.getLogger("RAGAgent")
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
AGENT_MODEL = os.getenv("AGENT_MODEL", "gpt-4o-mini")


# ---------------------------------------------------------------------------
# 1. TOOL: BUSCA SEMÂNTICA NO QDRANT
# ---------------------------------------------------------------------------
@tool
def search_qdrant_kb(query: str, limit: int = 3) -> str:
    """
    Pesquisa na base de conhecimento semântica do Qdrant por informações relevantes.
    Utilize esta ferramenta para buscar respostas detalhadas e dados históricos de faturas,
    documentos pessoais, logs de servidores e procedimentos corporativos.
    """
    logger.info(f"[Search Tool] Executando busca semântica para: '{query}'")
    try:
        client = get_qdrant_client()
        embeddings = get_embedding_engine()

        # Verificar se a coleção existe
        if not client.collection_exists(QDRANT_COLLECTION):
            return f"Coleção '{QDRANT_COLLECTION}' não existe no Qdrant. Execute o pipeline de ingestão primeiro."

        # Gerar o vetor para a query
        query_vector = embeddings.embed_query(query)

        # Buscar similaridade usando query_points (API unificada do Qdrant)
        response = client.query_points(
            collection_name=QDRANT_COLLECTION,
            query=query_vector,
            limit=limit
        )

        results = response.points if hasattr(response, "points") else []

        if not results:
            return "Nenhum resultado relevante encontrado na base de conhecimento semântica."

        formatted_results = []
        for rank, hit in enumerate(results, start=1):
            text = hit.payload.get("text", "")
            metadata = hit.payload.get("metadata", {})
            filename = metadata.get("filename", "desconhecido")
            topic = metadata.get("topic", "geral")
            score = round(hit.score if hit.score is not None else 0.0, 4)

            formatted_results.append(
                f"[Resultado #{rank}] (Score: {score}) (Origem: {filename}) (Tópico: {topic})\n"
                f"Trecho:\n{text}\n"
                f"---"
            )

        return "\n\n".join(formatted_results)
    except Exception as e:
        logger.error(f"[Search Tool] Erro na busca semântica: {e}")
        return f"Erro ao realizar busca semântica no Qdrant: {str(e)}"


# ---------------------------------------------------------------------------
# 2. FALLBACK MOCK LLM (Para execução offline / sem chave de API ativa)
# ---------------------------------------------------------------------------
class MockAgentLLM(BaseChatModel):
    """
    Mock LLM de alta fidelidade para testes e execuções offline.
    Simula perfeitamente o comportamento de chamadas de ferramentas e geração de respostas baseadas em contexto.
    """
    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[Any] = None,
        **kwargs: Any
    ) -> ChatResult:
        # Verifica se a última mensagem é uma mensagem do usuário pedindo informações
        last_message = messages[-1]

        # Caso o agente precise acionar a ferramenta de busca (fluxo de planejamento)
        has_tool_response = any(isinstance(msg, ToolMessage) for msg in messages)

        if not has_tool_response:
            # Analisa o prompt do usuário para extrair as palavras-chave
            user_text = last_message.content if hasattr(last_message, 'content') else str(last_message)
            logger.info(f"[MockLLM] Analisando a solicitação do usuário: '{user_text}'")
            logger.info("[MockLLM] Decidindo acionar ferramenta 'search_qdrant_kb'...")

            # Retorna um chamado de ferramenta estruturado
            tool_call = ToolCall(
                name="search_qdrant_kb",
                args={"query": user_text, "limit": 3},
                id="call_qdrant_rag"
            )
            message = AIMessage(
                content="[Análise de Intenção] Identifiquei que a solicitação exige dados da base de conhecimento. Vou pesquisar as informações indexadas.",
                tool_calls=[tool_call]
            )
            return ChatResult(generations=[ChatGeneration(message=message)])

        else:
            # Encontrar a última resposta da ferramenta
            tool_msg = next(msg for msg in reversed(messages) if isinstance(msg, ToolMessage))
            tool_content = tool_msg.content

            logger.info("[MockLLM] Processando resultados retornados pela ferramenta de busca...")

            # Sintetizar resposta baseada no conteúdo real da ferramenta
            if "Resultado #" in tool_content:
                # O conteúdo existe, vamos extrair as fontes
                response_text = (
                    "=== RESPOSTA DO AGENTE INTELIGENTE ===\n"
                    "Com base nas informações recuperadas dos documentos corporativos, aqui está o diagnóstico:\n\n"
                    f"{tool_content}\n\n"
                    "Espero ter ajudado! Se precisar de mais esclarecimentos ou detalhes técnicos, fique à vontade para perguntar."
                )
            else:
                response_text = (
                    "=== RESPOSTA DO AGENTE INTELIGENTE ===\n"
                    "Não encontrei informações explícitas ou documentos correspondentes na base de conhecimento para sua pergunta.\n"
                    "Por favor, certifique-se de que a base de dados relacional foi ingerida com sucesso ou forneça termos de busca mais amplos."
                )

            message = AIMessage(content=response_text)
            return ChatResult(generations=[ChatGeneration(message=message)])

    @property
    def _llm_type(self) -> str:
        return "mock_agent_llm"

    def bind_tools(self, tools: List[Any], **kwargs: Any) -> Any:
        # Essencial para compatibilidade com o prebuilt `create_react_agent` do LangGraph
        return self


# ---------------------------------------------------------------------------
# 3. PROMPT DE SISTEMA AVANÇADO DO AGENTE
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """Você é o Agente Virtual Inteligente de Atendimento Semântico (Data-to-RAG Agent).
Sua missão é interpretar com alta precisão e responder às solicitações dos usuários de forma fundamentada (Groundedness).

Você tem acesso à ferramenta `search_qdrant_kb` para pesquisar no banco de dados vetorial Qdrant.

Siga rigorosamente estas diretrizes no seu raciocínio (Chain of Thought):
1. **Análise Passo a Passo**: Leia a pergunta do usuário e compreenda o contexto subjacente e a intenção (ex: busca de faturas, logs de erro, etc).
2. **Reformulação de Consulta**: Decida quais termos ou palavras-chave são os mais estratégicos para pesquisar na base de conhecimento e execute a busca usando `search_qdrant_kb`.
3. **Consolidação de Dados**: Pegue as informações retornadas pela busca semântica, cruze os dados e remova contradições.
4. **Respostas Fundamentadas (Groundedness)**:
   - Dê respostas ricas e objetivas baseadas UNICAMENTE no contexto retornado pela busca.
   - Se os dados necessários não estiverem presentes no contexto retornado, informe explicitamente que não encontrou a informação.
   - Cite SEMPRE a fonte original da informação (ex: o nome do arquivo, categoria/tópico ou documento citado na resposta).
"""


# ---------------------------------------------------------------------------
# 4. INSTANCIAÇÃO E CONFIGURAÇÃO DO AGENTE
# ---------------------------------------------------------------------------
def get_agent():
    """
    Constrói e retorna o agente de conversação LangGraph com tratamento de
    compatibilidade de assinaturas para diferentes versões da biblioteca.
    """
    # Define se usa a LLM mockada ou real
    if USE_MOCK or not OPENAI_API_KEY:
        logger.info("[Agent] Utilizando MockAgentLLM para orquestração offline.")
        llm_model = MockAgentLLM()
    else:
        logger.info(f"[Agent] Conectando ao OpenAI Chat Model: '{AGENT_MODEL}'")
        llm_model = ChatOpenAI(model=AGENT_MODEL, temperature=0.1, openai_api_key=OPENAI_API_KEY)

    tools = [search_qdrant_kb]

    # Criação do Agente com Tratamento Dinâmico de Assinaturas (Compatibilidade de Versão do LangGraph)
    try:
        agent = create_react_agent(
            model=llm_model,
            tools=tools,
            prompt=SYSTEM_PROMPT
        )
    except TypeError:
        # Versões anteriores usavam 'state_modifier'
        agent = create_react_agent(
            model=llm_model,
            tools=tools,
            state_modifier=SYSTEM_PROMPT
        )

    return agent


if __name__ == "__main__":
    # Exemplo simples de invocação direta do agente
    agent_executor = get_agent()
    user_query = "Qual é o valor total das contas ou faturas encontradas?"
    logger.info(f"Testando agente localmente com a query: '{user_query}'")

    inputs = {"messages": [("user", user_query)]}
    for event in agent_executor.stream(inputs, stream_mode="values"):
        msg = event["messages"][-1]
        if msg.type == "ai" and msg.content:
            print(f"\n[Agente]: {msg.content}")
