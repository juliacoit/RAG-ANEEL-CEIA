# RAG ANEEL

Sistema de perguntas e respostas sobre a legislação do setor elétrico brasileiro, baseado na arquitetura RAG (Retrieval-Augmented Generation). Permite consultas em linguagem natural sobre atos normativos da ANEEL, com respostas citando a fonte exata e sem alucinações.

> **Fonte dos dados (Fase 1):** O plano original era extrair texto dos PDFs completos, mas o site da ANEEL bloqueia scripts automatizados via Cloudflare (apenas 29 de ~17 mil PDFs foram baixados). A solução adotada foi usar as **ementas** do JSON de metadados — resumos oficiais de cada ato normativo. O sistema cobre 15.528 atos (anos 2016, 2021 e 2022) e responde bem a perguntas do tipo "do que trata tal ato" ou "quais atos falam sobre PCH". Perguntas sobre artigos específicos ficam fora do escopo até que os PDFs completos sejam obtidos por outro meio.

---

## Stack

| Camada | Tecnologia |
|---|---|
| Orquestração | LangChain |
| Vector Store | Qdrant (busca híbrida: vetorial + BM25) |
| Embeddings | OpenAI `text-embedding-3-small` |
| LLMs | Claude 3.5 Sonnet / GPT-4o |
| API | FastAPI |
| Extração PDF | PyMuPDF (PDFs) / ementas do JSON (atual) |
| Serialização | Parquet |

---

## Arquitetura

```
JSON (metadados ANEEL: ementas, datas, tipos)
        │
        ▼
limpar_json_aneel.py  ──►  data/aneel_limpo_completo.json
        │
        ▼
chunker_json.py       ──►  data/processed/chunks_json_todos.parquet
                            (16.167 chunks · 15.528 atos · 2016/2021/2022)
        │
        ▼
vector_db.py          ──►  Qdrant (vetores + BM25 indexados)
        │
        ▼  (em produção)
hybrid_search.py  ◄──  pergunta do usuário
        │
        ▼
llm_chain.py      ──►  resposta com citação
        │
        ▼
main.py (FastAPI)  ──►  POST /query → JSON
```

---

## Estrutura

```
RAG-ANEEL-CEIA/
├── data/
│   ├── raw/                        # JSON de metadados original
│   ├── processed/                  # Chunks em .parquet (não versionados)
│   │   └── chunks_json_todos.parquet  # 16.167 chunks prontos para indexação
│   ├── aneel_limpo_completo.json   # Metadados limpos (todos os atos)
│   ├── aneel_vigentes_completo.json
│   └── samples/                    # Amostra para testes rápidos
├── src/
│   ├── p1_ingestion/
│   │   ├── limpar_json_aneel.py    # Limpeza e filtragem do JSON de metadados
│   │   ├── chunker_json.py         # Geração de chunks a partir das ementas (principal)
│   │   ├── baixar_pdfs_aneel.py    # Tentativa de download dos PDFs (bloqueado por Cloudflare)
│   │   └── parser.py               # Extração de texto de PDFs (para uso futuro)
│   ├── retrieval/
│   │   ├── vector_db.py            # Indexação no Qdrant
│   │   └── hybrid_search.py        # Busca vetorial + BM25 com fusão RRF
│   ├── api/
│   │   ├── main.py                 # Endpoints FastAPI
│   │   └── llm_chain.py            # Prompt engineering e chamada ao LLM
│   └── utils/                      # Logger, helpers, constantes
├── tests/
├── docs/                           # ADRs, benchmarks, RELATORIO_FASE_1.md
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

### 3. Limpar e processar o JSON de metadados

```bash
python src/p1_ingestion/limpar_json_aneel.py
# Gera: data/aneel_limpo_completo.json e data/aneel_vigentes_completo.json
```

### 4. Gerar chunks a partir das ementas

```bash
python src/p1_ingestion/chunker_json.py
# Gera: data/processed/chunks_json_todos.parquet
# 16.167 chunks · 15.528 atos normativos (2016, 2021, 2022)
```

> **Nota:** O download dos PDFs completos (`baixar_pdfs_aneel.py`) foi bloqueado pelo Cloudflare do site da ANEEL. Os chunks atuais são baseados nas ementas (resumos oficiais) do JSON de metadados. Veja `docs/RELATORIO_FASE_1.md` para detalhes.

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
| Data Engineer | Ingestão, chunking por ementas ✅ (Fase 1 concluída) |
| Search Engineer | Indexação vetorial, busca híbrida |
| AI Architect | API, prompt engineering, benchmark |
