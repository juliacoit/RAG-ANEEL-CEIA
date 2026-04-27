# CLAUDE.md — RAG ANEEL ⚡

Guia de contexto para o agente Claude Code trabalhar no projeto RAG ANEEL: sistema de busca e resposta sobre legislação do setor elétrico brasileiro usando arquitetura RAG.

---

## Visão Geral do Projeto

Sistema RAG (Retrieval-Augmented Generation) que permite consultas em linguagem natural sobre atos normativos da ANEEL (Resoluções, Portarias, Despachos), retornando respostas com citação direta da fonte e zero alucinações.

### Status das Fases

- **Fase 1 (Ingestão) — concluída**: 562.152 chunks em `data/processed/chunks_completo_unificado.parquet`. Fontes: PDFs, HTMLs e XLSXs de 25.202 arquivos baixados (anos 2016, 2021, 2022) via `curl_cffi` (contorna Cloudflare). Ementas em `chunks_json_todos.parquet` (16.167 chunks).
- **Fase 2 (Indexação/Retrieval) — concluída**: 562.152 vetores indexados no Qdrant com BGE-M3 (1024 dims) + índice BM25 em `db/bm25/`. Busca híbrida com pesos dinâmicos por tipo de query.
- **Fase 3 (API/LLM) — concluída**: API FastAPI rodando, LLM principal Qwen3-30B-A3B com cadeia de fallback automático (Groq → Gemini). Interface Streamlit, avaliação RAGAS com 45 perguntas.

---

## Stack Tecnológica

| Camada | Tecnologia |
|---|---|
| Linguagem | Python 3.11 / 3.12 |
| Vector Store | Qdrant (Docker local, porta 6333) |
| Embeddings | `BAAI/bge-m3` (1024 dims, padrão) / `MiniLM-L12-v2` (384 dims, fallback local) |
| Busca esparsa | BM25Okapi (`rank-bm25`) |
| LLM principal | Qwen3-30B-A3B via Dashscope (API OpenAI-compatível) |
| LLM fallback 1 | Groq — `llama-3.3-70b-versatile` (gratuito, 100k tokens/dia) |
| LLM fallback 2 | Gemini Flash — `gemini-2.0-flash` (opcional) |
| API | FastAPI + Uvicorn |
| Interface | Streamlit |
| Serialização | Parquet (via pandas + pyarrow) |
| Download | `curl_cffi` (impersona TLS Chrome 124 — contorna Cloudflare) |
| Extração PDF | PyMuPDF (fitz) + pdfplumber + Tesseract OCR |
| Extração HTML | BeautifulSoup4 |
| Extração XLSX | openpyxl |

---

## Estrutura de Pastas

```
RAG-ANEEL-CEIA/
├── data/
│   ├── raw/                              # JSONs brutos da ANEEL (versionar)
│   ├── aneel_limpo_completo.json         # todos os atos limpos
│   ├── aneel_vigentes_completo.json      # apenas atos vigentes
│   └── processed/                        # Parquets (não versionados)
│       ├── chunks_json_todos.parquet        # 16.167 chunks de ementas
│       ├── chunks_pdf_completo.parquet      # chunks de PDFs/HTML/XLSX
│       └── chunks_completo_unificado.parquet  # 562k chunks unificados (111MB)
├── db/
│   └── bm25/
│       ├── bm25_index.pkl               # índice BM25 gerado pelo p2_indexar
│       └── bm25_ids.pkl
├── pdfs/                                # ~25.202 arquivos baixados (não versionar)
├── 7-Zip/                               # binários 7-Zip para extração de RAR
├── poppler/Library/bin/                 # Poppler local para OCR (não versionar)
├── index_manual/                        # embeddings BGE-M3 pré-calculados (Colab)
│   └── embeddings_bge_m3.npy            # se existir, p2_indexar usa BGE-M3
├── scripts/
│   └── setup_pipeline.py               # orquestrador do pipeline completo
├── src/
│   ├── p1_ingestion/
│   │   ├── limpar_json_aneel.py         # limpeza dos JSONs brutos
│   │   ├── baixar_pdfs_aneel.py         # download via curl_cffi
│   │   ├── chunker_json.py              # chunking das ementas
│   │   ├── parser.py                    # extração multi-formato (PDF/HTML/XLSX)
│   │   └── unir_parquets.py             # une ementas + PDFs num parquet unificado
│   ├── p2_search/
│   │   └── p2_indexar.py               # indexação Qdrant + BM25 + busca híbrida
│   ├── api/
│   │   ├── main.py                      # FastAPI: endpoints, cache LRU, paralelismo
│   │   ├── query_optimizer.py           # rewriting, HyDE, decomposição, memória
│   │   ├── llm_chain.py                 # geração com cadeia Qwen→Groq→Gemini
│   │   └── analytics.py                 # perguntas analíticas via pandas (sem LLM)
│   └── utils/
│       └── logger_metrics.py            # logger + salvar_log para JSONL
├── tests/
│   └── banco_perguntas.py              # 45 perguntas categorizadas (importável)
├── banco_perguntas.py                  # cópia na raiz (uso direto)
├── eval_ragas.py                       # CLI de avaliação RAGAS
├── app.py                              # interface Streamlit
├── docker-compose.yml
├── requirements.txt
├── env.example
└── README.md
```

---

## Pipeline de Execução

```
JSONs brutos (data/raw/)
    │
    ▼
limpar_json_aneel.py  →  data/aneel_vigentes_completo.json
    │
    ▼
baixar_pdfs_aneel.py  →  pdfs/  (25.202 arquivos · curl_cffi · contorna Cloudflare)
    │
    ▼
chunker_json.py  →  chunks_json_todos.parquet   (ementas · 16k chunks)
parser.py        →  chunks_pdf_completo.parquet  (PDFs/HTML/XLSX · 562k chunks)
    │
    ▼
unir_parquets.py  →  chunks_completo_unificado.parquet  (111 MB)
    │
    ▼
p2_indexar.py  →  Qdrant (BGE-M3 1024 dims) + BM25 (db/bm25/)
    │
    ▼ (produção)
main.py (FastAPI)  ←  POST /query
    ├── query_optimizer.py  →  rewriting · HyDE · filtros · memória
    ├── busca híbrida       →  BM25 (0.70) + semântico (0.30), pesos por tipo
    └── llm_chain.py        →  Qwen3-30B → Groq → Gemini
    │
    ▼
app.py (Streamlit)  →  interface com histórico e métricas
```

---

## Convenções de Código

### Geral
- Python 3.11/3.12; usar type hints em todas as funções públicas
- Docstrings em português no estilo Google Docstrings
- Variáveis e funções em `snake_case`; classes em `PascalCase`
- Nunca hardcodar chaves de API, URLs ou parâmetros — usar sempre `.env` via `python-dotenv`
- `load_dotenv()` deve ser chamado **antes** de qualquer `os.getenv()` no topo do módulo

### Logs
- Usar o logger configurado em `src/utils/logger_metrics.py` — nunca `print()` em produção
- Logar sempre: início/fim de operações longas, erros com traceback, métricas de custo (tokens, latência, modelo usado)
- Cada chamada ao LLM é salva via `salvar_log()` em `data/logs/logs.jsonl`

### Tratamento de Erros
- Tratar explicitamente: erros de rede (timeout, 429), PDFs corrompidos/escaneados, falhas de API
- Fallback em cascata no LLM: Qwen → Groq → Gemini → mensagem de erro clara, sem crash

### Dados e Arquivos
- PDFs nunca entram no Git (`pdfs/` está no `.gitignore`)
- Parquets processados não são versionados (`data/processed/`)
- JSONs de metadados limpos (`data/aneel_*.json`) estão versionados
- Embeddings BGE-M3 pré-calculados (`index_manual/*.npy`) não são versionados

---

## Módulos e Responsabilidades

### `src/p1_ingestion/limpar_json_aneel.py`
- Lê os JSONs brutos da ANEEL (3 arquivos: 2016, 2021, 2022)
- Filtra e normaliza campos (datas, tipos de ato, URLs)
- Gera `data/aneel_limpo_completo.json` e `data/aneel_vigentes_completo.json`

### `src/p1_ingestion/baixar_pdfs_aneel.py`
- Download de PDFs, HTMLs, XLSXs e RARs via `curl_cffi` (impersona TLS Chrome 124)
- Contorna proteção Cloudflare do site da ANEEL
- Download paralelo com workers configuráveis
- Falhas registradas em log para reprocessamento

### `src/p1_ingestion/chunker_json.py`
- Lê as ementas de `aneel_vigentes_completo.json`
- Gera chunks com `RecursiveCharacterTextSplitter` (600 tokens / 90 overlap)
- Metadados por chunk: número do ato, tipo, data, URL
- Output: `data/processed/chunks_json_todos.parquet`

### `src/p1_ingestion/parser.py`
- Extração de texto multi-formato: PDF (PyMuPDF + pdfplumber + OCR), HTML (BS4), XLSX (openpyxl)
- Chunk size: 800 chars / 120 overlap para PDFs
- Cabeçalho de coluna injetado em chunks de XLSX (crítico para LLM entender tabelas)
- Processamento paralelo com `ProcessPoolExecutor`

### `src/p1_ingestion/unir_parquets.py`
- Alinha colunas e concatena `chunks_json_todos.parquet` + `chunks_pdf_completo.parquet`
- Output: `chunks_completo_unificado.parquet` (562k chunks · 111 MB)

### `src/p2_search/p2_indexar.py` — núcleo da Fase 2
- Detecta automaticamente modelo de embedding: BGE-M3 (se `index_manual/embeddings_bge_m3.npy` existir) ou MiniLM-L12 (fallback local)
- Indexa no Qdrant com distância cosseno
- Cria e persiste índice BM25 em `db/bm25/`
- **Pesos híbridos dinâmicos por tipo de query**: BM25 dominante (0.70) para buscas gerais; mais equilibrado (0.60/0.40) para procedimentos e queries híbridas complexas
- Expansão de contexto: chunks vizinhos (±2)
- Filtros automáticos: número e ano extraídos da query via regex
- Fallback "modo menção": quando score < 0.5, busca sem filtros e marca `busca_fallback=True`

### `src/api/main.py`
- Endpoint `POST /query`: recebe pergunta + filtros, orquestra optimizer → busca → LLM
- Cache LRU (100 entradas, TTL 5 min) para evitar chamadas repetidas
- `ThreadPoolExecutor` para paralelismo (optimizer + embedding simultâneos)
- Timeout de 12s no optimizer — fallback para query original se demorar
- `GET /health` e `GET /docs` (Swagger automático)

### `src/api/query_optimizer.py`
- Chama o Groq para reescrever a query com vocabulário técnico ANEEL
- Gera HyDE (trecho hipotético que simula um ato normativo real)
- Decompõe perguntas compostas em sub-queries (max 3)
- Resolve referências pronominais via histórico dos últimos 6 turnos
- Detecta mudança de contexto e descarta histórico quando necessário
- Esclarecimento progressivo: até 3 pedidos antes de buscar com o que tem
- Roteamento analítico: queries de contagem/ranking vão para `analytics.py`, sem LLM

### `src/api/llm_chain.py`
- Cadeia de fallback: Qwen3-30B-A3B → Groq llama-3.3-70b → Gemini Flash
- 6 system prompts adaptativos: BUSCA · RESUMO · COMPARACAO · TABELA · ESPECIFICA · MENCAO
- Seleção automática por palavras-chave na pergunta; MENCAO ativado por `busca_fallback=True`
- Temperatura 0.1 (quasi-determinístico)
- Log completo a cada chamada via `salvar_log()`: tokens, latência, modelo, faithfulness

### `src/api/analytics.py`
- Responde perguntas analíticas (contagem, ranking, agregação) diretamente via pandas
- Não chama LLM — lê o parquet de metadados
- Detectado automaticamente pelo `query_optimizer` antes de acionar a busca vetorial

### `eval_ragas.py` e `banco_perguntas.py`
- 45 perguntas em 10 categorias temáticas (microgeração, tarifas, PCH, concessões, etc.)
- CLI: `python eval_ragas.py --limite N --categoria X --tipo Y --saida arquivo.csv`
- Métricas RAGAS: Faithfulness, Response Relevancy, Context Precision
- Avaliador: Groq (sem custo extra)

---

## Variáveis de Ambiente

Sempre ler do `.env` via `python-dotenv`. Nunca commitar o `.env` real.

```env
# LLM principal
LLM_PROVIDER=qwen
LLM_MODEL=qwen3-30b-a3b
QWEN_API_KEY=sk-...
QWEN_BASE_URL=https://dashscope-intl.aliyuncs.com/compatible-mode/v1

# Fallback 1 (gratuito)
GROQ_API_KEY=gsk_...
GROQ_MODEL=llama-3.3-70b-versatile

# Fallback 2 (opcional)
GEMINI_API_KEY=
GEMINI_MODEL=gemini-2.0-flash

# Qdrant
QDRANT_URL=http://localhost:6333
QDRANT_COLLECTION=aneel_chunks

# Busca
TOP_K_RETRIEVAL=10
OPTIMIZER_TIMEOUT=12
```

---

## Decisões de Arquitetura (ADRs resumidos)

| Decisão | Escolha | Motivo |
|---|---|---|
| Vector Store | Qdrant | Suporte a busca vetorial + BM25, Docker local, sem custo |
| Embeddings | BGE-M3 (1024 dims) | Melhor qualidade semântica para PT-BR; gerado no Colab L4 (~25 min) |
| Embedding fallback | MiniLM-L12 (384 dims) | Roda local sem GPU; ativado se `embeddings_bge_m3.npy` não existir |
| Busca esparsa | BM25Okapi | Captura termos técnicos exatos (siglas, números de resolução) |
| Fusão híbrida | Pesos dinâmicos por tipo | BM25 dominante (0.70) para buscas gerais; equilibrado para queries complexas |
| LLM principal | Qwen3-30B-A3B | Boa qualidade, API OpenAI-compatível, custo baixo |
| LLM fallback | Groq → Gemini | Gratuitos; ativados automaticamente em cascata |
| Download de PDFs | `curl_cffi` | Impersona TLS Chrome 124 — contorna proteção Cloudflare da ANEEL |
| Serialização | Parquet | Leitura colunar rápida, compressão eficiente |
| Anti-alucinação | System prompt + temperatura 0.1 | Responde apenas com base nos chunks; recusa obrigatória quando ausente |

---

## O Que Nunca Fazer

- ❌ Hardcodar chaves de API ou URLs no código
- ❌ Chamar `os.getenv()` antes de `load_dotenv()` — variáveis serão `None`
- ❌ Usar `print()` em vez do logger configurado
- ❌ Commitar `.env`, PDFs, Parquets processados ou embeddings `.npy`
- ❌ Substituir pesos híbridos dinâmicos por valor fixo sem revisar os tipos de query
- ❌ Deixar o LLM responder além do contexto recuperado (zero inferência externa)
- ❌ Alterar o system prompt base sem revisão da equipe
- ❌ Processar a base completa em testes — usar `--limite N` no setup_pipeline

---

## Métricas de Avaliação (Benchmark)

O sistema é avaliado com 45 perguntas via `eval_ragas.py`:

- **Faithfulness**: a resposta é fiel aos documentos recuperados?
- **Response Relevancy**: a resposta resolve a dúvida?
- **Context Precision**: os chunks recuperados são relevantes para a pergunta?

Resultados salvos em `docs/ragas_resultado_<timestamp>.csv`.
