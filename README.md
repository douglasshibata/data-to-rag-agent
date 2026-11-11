# 🤖 Data-to-RAG Agent (Sistema Inteligente de Ingestão e Atendimento Agentivo)

Este projeto implementa um pipeline completo de ponta a ponta (**Data-to-RAG**) projetado para extrair dados estruturados de um banco de dados relacional clássico, enriquecê-los em documentos ricos em contexto, gerar embeddings semânticos de alta fidelidade e indexá-los em um banco de dados vetorial (**Qdrant**). Além disso, disponibiliza um agente de conversação inteligente (**LangGraph ReAct Agent**) que interpreta solicitações dos usuários, reformula consultas dinamicamente e responde de forma fundamentada (*Groundedness*) citando as fontes originais dos documentos.

---

## 🏗️ 1. Arquitetura do Sistema

A arquitetura do sistema divide-se em um fluxo assíncrono de carga (ETL/Ingestão) e um fluxo síncrono de consulta/resposta orientado a agentes (Chat).

```text
==========================================================================================
                     DIAGRAMA DE ARQUITETURA DO PIPELINE DATA-TO-RAG
==========================================================================================

1. FLUXO ASSÍNCRONO DE INGESTÃO (ETL)
-------------------------------------
+-------------------------+      +-----------------------+      +------------------------+
| Banco de Dados Original | ---> |   Extrator / Chunker  | ---> | Gerador de Embeddings  |
|  (SQLAlchemy SQLite/PG) |      | (Enrichment/Splitter) |      | (OpenAI / Deterministic)
+-------------------------+      +-----------------------+      +------------------------+
                                                                             |
                                                                             v
                                                                +------------------------+
                                                                |   Vector DB (Qdrant)   |
                                                                | (Coleção de Conhecimento)
                                                                +------------------------+

2. FLUXO SÍNCRONO DO AGENTE (CHAT)
----------------------------------
+---------+      +-------------------------------+      +-------------------------+
| Usuário | ---> |  Agente Analisador de Intenção | ---> |    Query Reformulator   |
+---------+      |     (LangGraph ReAct Loop)    |      | (Otimização de Termos)  |
   ^             +-------------------------------+      +-------------------------+
   |                                                                 |
   |                                                                 v
+------------------------+      +------------------------------------+
|  Resposta Estruturada  | <--- | Busca Vetorial Semântica (Qdrant)  |
| (Groundedness/Citação) |      |         (search_qdrant_kb)         |
+------------------------+      +------------------------------------+
```

### Componentes Principais:
1. **Conector Relacional (SQLAlchemy)**: Extrai registros brutos de forma segura e flexível do banco relacional original (por padrão, o SQLite `documents.db`).
2. **Chunker & Enricher Customizado**: Consolida os metadados de cada registro (nome do arquivo, tópico, data de criação) no próprio texto para garantir retenção de contexto durante a fragmentação em blocos.
3. **Dual Embedding Engine**: Se a chave de API da OpenAI estiver ausente ou `USE_MOCK` for ativado, o sistema utiliza um algoritmo matemático/hash determinístico resiliente que simula embeddings de 1536 dimensões perfeitamente normalizadas L2. Se ativado, usa o modelo real `text-embedding-3-small`.
4. **Armazenamento Vetorial (Qdrant)**: Realiza upserts em lote de pontos estruturados contendo payloads ricos e IDs mapeados para formato UUID.
5. **Orquestrador de Agente (LangGraph)**: Utiliza `create_react_agent` para gerenciar as tomadas de decisões e execução em malha de loops até a consolidação da resposta final.

---

## 🛠️ 2. Instalação

### Pré-requisitos:
- **Python 3.10 ou superior** instalado na máquina.
- **pip** (instalador de pacotes do Python).

### Instalação das Dependências:
Navegue para o diretório raiz do projeto e instale as dependências listadas no `requirements.txt`:
```bash
cd data-to-rag-agent
pip install -r requirements.txt
```

---

## ⚙️ 3. Configuração

Toda a configuração do sistema é gerenciada via variáveis de ambiente. Crie um arquivo chamado `.env` no diretório `data-to-rag-agent/` com as seguintes definições estruturadas:

```env
# Banco de Dados de Origem (SQLAlchemy)
DB_PATH=documents.db

# Configurações do Banco Vetorial Qdrant
QDRANT_HOST=localhost
QDRANT_PORT=6333
# Use :memory: para carregar o Qdrant localmente em memória (perfeito para testes e execução offline)
QDRANT_LOCATION=:memory:
QDRANT_COLLECTION=data_to_rag_collection

# Chaves de API e Inteligência Artificial
OPENAI_API_KEY=sua-chave-openai-real-se-houver
AGENT_MODEL=gpt-4o-mini

# Defina como "true" para rodar localmente sem consumir custos de tokens (usando o mock determinístico)
USE_MOCK=true
```

---

## 🚀 4. Execução & Uso

O sistema oferece três formas de interação de ponta a ponta:

### 4.1 Iniciar o Servidor API FastAPI
Para subir o servidor de produção local e expor os endpoints REST:
```bash
# A partir do diretório do projeto
uvicorn main:app --reload --port 8000
```
A API estará acessível em `http://localhost:8000`.

### 4.2 Endpoint: Ingestão Manual (`POST /ingest`)
Para forçar manualmente a sincronização de dados do banco de dados relacional para o Qdrant:
```bash
curl -X POST http://localhost:8000/ingest
```

### 4.3 Endpoint: Atendimento Conversacional (`POST /chat`)
Para conversar com o agente e obter respostas inteligentes baseadas em seus documentos:
```bash
curl -X POST http://localhost:8000/chat \
     -H "Content-Type: application/json" \
     -d '{"message": "Qual é o assunto do documento test_file.pdf?"}'
```

---

## 🧪 5. Testes

O projeto conta com uma suíte de testes automatizados unitários e de integração de altíssima cobertura, validando o processador de embeddings, o pipeline ETL de ponta a ponta, o agente conversacional ReAct e os endpoints da FastAPI.

Para executar os testes:
```bash
# No diretório raiz do monorepo ou no diretório do projeto
PYTHONPATH=data-to-rag-agent pytest data-to-rag-agent/tests/ -v
```

---

## 🛡️ 6. Segurança

1. **Prevenção de SQL Injection**: Toda a extração de dados é realizada utilizando o ORM do SQLAlchemy com seletores de tipos tipados ou consultas parametrizadas.
2. **Armazenamento Seguro de Credenciais**: Chaves de API e dados confidenciais nunca são persistidos diretamente no código, sendo obrigatoriamente lidos do arquivo `.env` via biblioteca `python-dotenv`.
3. **Groundedness Regras Rígidas**: O System Prompt força o agente a responder exclusivamente com base no contexto retornado pela ferramenta de busca vetorial, prevenindo alucinações de dados pessoais ou confidenciais de outros escopos.

---

## 🔍 7. Troubleshooting (Solução de Problemas)

| Sintoma | Causa Comum | Solução Recomendada |
|---------|-------------|---------------------|
| `sqlite3.OperationalError: no such table: documentos` | O arquivo `documents.db` não foi encontrado ou está vazio. | Certifique-se de que o arquivo `documents.db` está localizado na pasta raiz e contém a tabela `documentos`. Você pode rodar o pipeline para testar passando um banco de testes. |
| `AttributeError: 'QdrantClient' object has no attribute 'search'` | Versão atual do Qdrant Client utiliza a interface moderna `query_points`. | O código já está padronizado para usar a API unificada `query_points` que substitui o `.search` legado nas versões recentes. |
| Erros de limite de taxa ou rede da OpenAI | Rate limiting na geração de embeddings. | Ative `USE_MOCK=true` no arquivo `.env` para rodar os testes e a API de forma offline com nosso motor de simulação inteligente de alto desempenho. |
