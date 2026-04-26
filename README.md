# RAG ANEEL

Sistema de IA para busca e resposta especializada sobre legislação do setor elétrico brasileiro, construído sobre arquitetura **RAG (Retrieval-Augmented Generation)**.

## Objetivo

Permite que usuários façam perguntas em linguagem natural sobre atos normativos da ANEEL (resoluções, despachos, portarias) e recebam respostas fundamentadas com citação direta da fonte — sem alucinações.

## Stack Tecnológica

| Componente | Tecnologia |
|---|---|
| Linguagem | Python 3.11 / 3.12 |
| Vector Store | Qdrant (Docker local, porta 6333) |
| Embeddings | `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` (384 dims, local) |
| Busca esparsa | BM25Okapi (`rank-bm25`) |
| LLM principal | Groq — `llama-3.3-70b-versatile` (gratuito, 100k tokens/dia) |
| LLM fallback | DeepSeek R1 — `deepseek-reasoner` (fallback automático no rate limit) |
| API | FastAPI + Uvicorn |
| Interface | Streamlit |
| Dados | Parquet (`pyarrow`) |

## Escala dos Dados

- **562.152 chunks** indexados (vetorial + BM25)
- Fontes: PDFs, XLSXs e HTMLs de atos normativos da ANEEL — anos 2016, 2021 e 2022
- Tipos de ato: DSP, REN, REH, REA, PRT, entre outros
- ~25.202 arquivos baixados (~27k PDFs + HTMLs + XLSXs + RARs)

## Arquitetura do Pipeline

```
JSONs brutos ANEEL
       ↓ limpar_json_aneel.py
aneel_vigentes_completo.json
       ↓ baixar_pdfs_aneel.py
pdfs/  (PDFs, HTMLs, XLSXs, RARs)
       ↓ parser.py + chunker_json.py
chunks_*.parquet
       ↓ unir_parquets.py
chunks_completo_unificado.parquet  (562k chunks)
       ↓ p2_indexar.py
Qdrant (vetorial) + BM25 (esparso)
       ↓
API FastAPI (main.py)
  ├── query_optimizer.py  → rewriting, HyDE, decomposição, filtros
  ├── buscar()            → busca híbrida (0.6 semântico + 0.4 BM25)
  └── llm_chain.py        → geração com Groq/Llama 3.3 70B
       ↓
Streamlit (app.py)
```

## Fases do Projeto

### P1 — Engenharia de Dados (`src/p1_ingestion/`)

| Script | Responsabilidade |
|---|---|
| `limpar_json_aneel.py` | Limpeza e filtragem dos JSONs brutos da ANEEL |
| `baixar_pdfs_aneel.py` | Download de PDFs, HTMLs, XLSXs e RARs via `curl_cffi` (contorna Cloudflare) |
| `chunker_json.py` | Chunking das ementas via LangChain `RecursiveCharacterTextSplitter` |
| `parser.py` | Extração de texto de PDFs (PyMuPDF + pdfplumber), HTMLs (BeautifulSoup) e XLSXs (openpyxl) |
| `unir_parquets.py` | União dos parquets de ementas e PDFs |

### P2 — Busca Híbrida (`src/p2_search/`)

| Recurso | Detalhe |
|---|---|
| Vetorial | Qdrant, distância cosseno, embeddings locais MiniLM-L12 |
| Esparso | BM25Okapi sobre corpus tokenizado em PT-BR |
| Combinação | Weighted hybrid: 0.6 semântico + 0.4 BM25 |
| Expansão | Chunks vizinhos (até 8 base + 2 vizinhos por lado) |
| Filtros | Número de ato e ano extraídos automaticamente da query via regex |
| Fallback | Busca sem filtro quando score < 0.5 (retorna "menções") |
| HyDE | Trecho hipotético gerado pelo LLM para enriquecer o embedding da query |

### P3 — API, LLM e Interface (`src/api/`, `app.py`)

| Módulo | Responsabilidade |
|---|---|
| `main.py` | FastAPI com lifespan, cache LRU (100 entradas, TTL 5min), ThreadPoolExecutor |
| `query_optimizer.py` | Rewriting, HyDE, decomposição, memória de conversa, detecção de ambiguidade |
| `llm_chain.py` | 6 system prompts adaptativos (BUSCA, RESUMO, COMPARACAO, TABELA, ESPECIFICA, MENCAO) |
| `analytics.py` | Perguntas analíticas sobre metadados via pandas — sem chamar o LLM |
| `app.py` | Interface Streamlit com histórico de conversa (10 turnos) e métricas |

## Funcionalidades de Query Optimization

- **Rewriting** — expande vocabulário técnico ANEEL
- **HyDE** — gera trecho hipotético para melhorar o embedding semântico
- **Decomposição** — divide perguntas complexas em sub-queries
- **Filtros automáticos** — extrai número e ano do ato da query por regex
- **Memória** — resolve referências ("esse despacho", "ela", etc.) nos últimos 6 turnos
- **Detecção de mudança de contexto** — descarta histórico irrelevante automaticamente
- **Esclarecimento progressivo** — até 3 pedidos de esclarecimento por contexto
- **Roteamento analítico** — perguntas de contagem/ranking vão direto para pandas, sem LLM

## Endpoints da API

| Método | Rota | Descrição |
|---|---|---|
| `POST` | `/query` | Recebe pergunta + histórico, retorna resposta com citações |
| `GET` | `/health` | Status do sistema (Qdrant, BM25, modelo) |
| `GET` | `/docs` | Swagger automático (FastAPI) |

## Como Executar

### Pré-requisitos

Ver [SETUP.md](SETUP.md) para instalação de dependências externas (Tesseract, Poppler, 7-Zip, Docker).

```bash
# 1. Instalar dependências Python
pip install -r requirements.txt

# 2. Criar .env (ver env.example)
cp env.example .env
# editar .env e adicionar GROQ_API_KEY e ANTHROPIC_API_KEY

# 3. Subir Qdrant
docker compose up -d

# 4. Pipeline completo (primeira vez — ~30 min)
python scripts/setup_pipeline.py

# 5. Subir API
python -m uvicorn src.api.main:app --reload --port 8000

# 6. Interface Streamlit
python -m streamlit run app.py
```

### Testar retrieval sem gastar tokens

```python
from src.p2_search.p2_indexar import conectar_qdrant, carregar_indices, buscar

qclient = conectar_qdrant()
bm25, bm25_ids = carregar_indices(qclient)
resultados = buscar("valor TUST ciclo 2021-2022", qclient, bm25, bm25_ids, n_resultados=5)
for r in resultados:
    print(r["tipo_nome"], r["numero"], r["ano"], "| score:", r["score_final"])
    print(" ", r["texto"][:200])
```

## Estrutura de Pastas

```
RAG-ANEEL/
├── data/
│   ├── raw/                              # JSONs brutos (versionar)
│   ├── aneel_vigentes_completo.json      # gerado (não versionar)
│   └── processed/
│       ├── chunks_json_todos.parquet     # ementas (2.4MB — pode versionar)
│       ├── chunks_pdf_completo.parquet   # PDFs (grande — não versionar)
│       └── chunks_completo_unificado.parquet  # 562k chunks (111MB — não versionar)
├── db/bm25/                             # índice BM25 (gerado localmente)
├── pdfs/                                # ~27k arquivos baixados (não versionar)
├── scripts/
│   └── setup_pipeline.py               # orquestrador completo do pipeline
├── src/
│   ├── p1_ingestion/                   # download, parse, chunk
│   ├── p2_search/                      # indexação e busca híbrida
│   └── api/                            # FastAPI, LLM, analytics
├── app.py                              # interface Streamlit
├── docker-compose.yml
├── requirements.txt
└── env.example
```

## Variáveis de Ambiente

```env
GROQ_API_KEY=gsk_...                   # console.groq.com — gratuito
DEEPSEEK_API_KEY=sk-...                # platform.deepseek.com (fallback automático)
LLM_PROVIDER=groq                      # groq | deepseek
LLM_MODEL=llama-3.3-70b-versatile
DEEPSEEK_MODEL=deepseek-reasoner
QDRANT_URL=http://localhost:6333
TOP_K_RETRIEVAL=10
OPTIMIZER_TIMEOUT=12
```

## Limitações Conhecidas

1. **Anos disponíveis:** apenas 2016, 2021 e 2022. Documentos de outros anos não estão na base.
2. **Cloudflare:** ~25.202 de ~27k PDFs foram baixados com sucesso. Documentos ausentes são reportados como "menção".
3. **XLSXs sem cabeçalho por chunk:** cálculos e somatórios sobre valores financeiros não funcionam via RAG puro — requer módulo analítico dedicado (planejado).
4. **Rate limit Groq:** 100k tokens/dia no plano gratuito (~60–70 perguntas complexas/dia).
5. **Embeddings:** indexação base usa MiniLM-L12 (384 dims, local). Uma nova indexação com `BAAI/bge-m3` (1024 dims) foi gerada no Google Colab (GPU L4) e pode ser carregada automaticamente se o arquivo de índice estiver na raiz do projeto — caso contrário o sistema usa o MiniLM local.

## Benchmark de Avaliação

O sistema é avaliado com 45 perguntas em 10 categorias temáticas usando métricas RAGAS:

- **Faithfulness** — a resposta é fiel aos documentos recuperados?
- **Response Relevancy** — a resposta resolve a dúvida do usuário?
- **Context Precision** — os chunks recuperados são relevantes para a pergunta?

Avaliador: Groq (mesmo provider — sem custo extra de OpenAI).
