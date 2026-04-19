# RAG ANEEL

Sistema de perguntas e respostas sobre a legislação do setor elétrico brasileiro, baseado na arquitetura RAG (Retrieval-Augmented Generation). Permite consultas em linguagem natural sobre atos normativos da ANEEL, com respostas citando a fonte exata e sem alucinações.

---

## Stack

| Camada | Tecnologia |
|---|---|
| Orquestração | LangChain |
| Vector Store | Qdrant (busca híbrida: vetorial + BM25) |
| Embeddings | OpenAI `text-embedding-3-small` |
| LLMs | Claude 3.5 Sonnet / GPT-4o |
| API | FastAPI |
| Extração PDF | PyMuPDF |
| Serialização | Parquet |

---

## Arquitetura

```
JSON (metadados ANEEL)
        │
        ▼
downloader.py  ──►  data/raw/*.pdf
        │
        ▼
parser.py      ──►  texto limpo
        │
        ▼
chunker.py     ──►  data/processed/chunks.parquet
        │
        ▼
vector_db.py   ──►  Qdrant (vetores + BM25 indexados)
        │
        ▼  (em produção)
hybrid_search.py  ◄──  pergunta do usuário
        │
        ▼
llm_chain.py   ──►  resposta com citação
        │
        ▼
main.py (FastAPI)  ──►  POST /query → JSON
```

---

## Estrutura

```
RAG-ANEEL-CEIA/
├── data/
│   ├── raw/            # JSON de metadados + PDFs baixados (PDFs não versionados)
│   ├── processed/      # Chunks em .parquet (não versionados)
│   └── samples/        # Amostra para testes rápidos (50–200 chunks)
├── src/
│   ├── ingestion/
│   │   ├── downloader.py    # Download assíncrono dos PDFs
│   │   ├── parser.py        # Extração e limpeza de texto
│   │   └── chunker.py       # Fatiamento em chunks com overlap
│   ├── retrieval/
│   │   ├── vector_db.py     # Indexação no Qdrant
│   │   └── hybrid_search.py # Busca vetorial + BM25 com fusão RRF
│   ├── api/
│   │   ├── main.py          # Endpoints FastAPI
│   │   └── llm_chain.py     # Prompt engineering e chamada ao LLM
│   └── utils/               # Logger, helpers, constantes
├── tests/
├── docs/                    # ADRs, benchmarks, análises de custo
├── requirements.txt
├── .env.example
└── .gitignore
```

---

## Setup

**Pré-requisitos:** Python 3.10+, Docker, chaves de API OpenAI e Anthropic.

### 1. Clonar e instalar

```bash
git clone <url-do-repo>
cd RAG-ANEEL-CEIA

python -m venv venv
source venv/bin/activate        # Linux/Mac
venv\Scripts\activate           # Windows

pip install -r requirements.txt
```

### 2. Configurar variáveis de ambiente

```bash
cp .env.example .env
# Preencher .env com as chaves de API
```

```env
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
QDRANT_URL=http://localhost:6333
QDRANT_API_KEY=
CHUNK_SIZE=600
CHUNK_OVERLAP=90
TOP_K_RETRIEVAL=5
LLM_MODEL=claude-3-5-sonnet-20241022
```

### 3. Processar metadados

```bash
python src/limpar_json_aneel.py \
  --input data/raw/biblioteca_aneel_gov_br_legislacao_2016_metadados.json \
  --output data/raw/
# Gera: data/raw/aneel_limpo.json e data/raw/aneel_vigentes.json
```

### 4. Baixar os PDFs

```bash
python src/ingestion/downloader.py --input data/raw/aneel_limpo.json
# Destino: data/raw/*.pdf  (não versionados no Git)
```

### 5. Extrair texto e gerar chunks

```bash
python src/ingestion/parser.py
python src/ingestion/chunker.py
# Gera: data/processed/chunks.parquet
```

### 6. Indexar no Qdrant

```bash
docker run -p 6333:6333 qdrant/qdrant

python src/retrieval/vector_db.py
```

### 7. Iniciar a API

```bash
uvicorn src.api.main:app --reload
# Documentação: http://localhost:8000/docs
```

---

## Avaliação

O sistema é avaliado por três métricas (resultados em `docs/`):

- **Faithfulness** — a resposta é fiel aos documentos recuperados?
- **Answer Relevance** — a resposta resolve a dúvida do usuário?
- **Citation Accuracy** — as resoluções e datas citadas estão corretas?

---

## Divisão da Equipe

| Papel | Responsabilidade |
|---|---|
| Data Engineer | Ingestão, parsing de PDFs, chunking |
| Search Engineer | Indexação vetorial, busca híbrida |
| AI Architect | API, prompt engineering, benchmark |
