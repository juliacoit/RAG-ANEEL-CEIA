# RAG ANEEL ⚡

Sistema de IA para busca e resposta especializada sobre a legislação do setor elétrico brasileiro.

## 🚀 Objetivo
Este projeto utiliza a arquitetura **RAG (Retrieval-Augmented Generation)** para permitir que usuários façam perguntas em linguagem natural sobre atos normativos da ANEEL, obtendo respostas precisas, sem alucinações e com citação direta da fonte.

## 🛠️ Tecnologias Utilizadas
- **Linguagem:** Python 3.10+
- **Orquestração:** LangChain
- **Base de Dados Vetorial:** Qdrant (Busca Híbrida: Vetorial + BM25)
- **Embeddings:** OpenAI `text-embedding-3-small`
- **LLMs:** Claude 3.5 Sonnet / GPT-4o
- **API:** FastAPI

## 📋 Divisão da Equipe (Sprint 12 Dias)
- **Pessoa 1 (Data Engineer):** Ingestão de dados, Parsing de PDFs e estratégia de Chunking.
- **Pessoa 2 (Search Engineer):** Indexação vetorial, gestão da Vector Store e Busca Híbrida.
- **Pessoa 3 (AI Architect):** Desenvolvimento da API, Prompt Engineering e Avaliação (Benchmark).

## 🔧 Como Executar
1. Clone o repositório: `git clone ...`
2. Instale as dependências: `pip install -r requirements.txt`
3. Configure o arquivo `.env` com suas chaves de API.
4. Execute a ingestão: `python src/ingestion/parser.py`
5. Inicie a API: `uvicorn src.api.main:app --reload`

## 📈 Benchmark
O sistema é avaliado com base em:
- **Faithfulness:** A resposta é fiel aos documentos?
- **Answer Relevance:** A resposta resolve a dúvida do especialista?
- **Citation Accuracy:** As resoluções e datas citadas estão corretas?
