# CLAUDE.md — RAG ANEEL ⚡

Guia de contexto para o agente Claude Code trabalhar no projeto RAG ANEEL: sistema de busca e resposta sobre legislação do setor elétrico brasileiro usando arquitetura RAG.

---

## Visão Geral do Projeto

Sistema RAG (Retrieval-Augmented Generation) que permite consultas em linguagem natural sobre atos normativos da ANEEL (Resoluções, Portarias, Despachos), retornando respostas com citação direta da fonte e zero alucinações.

### Status das Fases

- **Fase 1 (Ingestão) — :** Os chunks estão prontos em `data/processed/chunks_json_todos.parquet` (16.167 chunks, 15.528 atos, anos 2016/2021/2022). A fonte dos chunks são as **ementas** do JSON de metadados — Download dos PDFS em andamento. Ver `docs/RELATORIO_FASE_1.md`.
- **Fase 2 (Indexação/Retrieval) — em andamento**: Indexação no Qdrant concluído, porém apenas com as ementas, os pdfs ainda nao foram implementados.
- **Fase 3 (API/LLM) — em andamento**: API rodando e modelo respondendo, porém não tem contexto suficiente devido a falta dos PDFs baixados.

### Limitação conhecida
O sistema responde bem a perguntas sobre o tema e escopo dos atos ("do que trata tal resolução", "quais atos falam sobre PCH"). Perguntas sobre artigos ou incisos específicos não são respondíveis com a base atual — requerem os PDFs completos, que precisam ser obtidos por outro meio (ex: acesso direto, scraping manual, fonte alternativa).

---

## Stack Tecnológica

| Camada | Tecnologia |
|---|---|
| Linguagem | Python 3.10+ |
| Orquestração | LangChain |
| Vector Store | Qdrant (busca híbrida: vetorial + BM25) |
| Embeddings | OpenAI `text-embedding-3-small` (1536 dims) |
| LLMs | Llama 3.3 70b versatile (`llama-3.3-70b-versatile`) / GPT-4o |
| API | FastAPI + Uvicorn |
| Serialização | Parquet (via pandas + pyarrow) |
| Download | aiohttp + asyncio |
| Extração PDF | PyMuPDF (fitz) |

---

## Estrutura de Pastas

```
RAG-ANEEL-CEIA/
├── data/
│   ├── raw/                         # JSON de metadados original
│   ├── processed/                   # Chunks em .parquet (não versionados)
│   │   └── chunks_json_todos.parquet   # 16.167 chunks prontos para indexação
│   ├── aneel_limpo_completo.json    # Metadados limpos (todos os atos)
│   ├── aneel_vigentes_completo.json
│   └── samples/                     # Amostra de 50–200 chunks para testes rápidos
├── src/
│   ├── p1_ingestion/
│   │   ├── limpar_json_aneel.py    # Limpeza e filtragem do JSON de metadados
│   │   ├── chunker_json.py         # Geração de chunks a partir das ementas (principal)
│   │   ├── baixar_pdfs_aneel.py    # Download de PDFs (bloqueado por Cloudflare — não usar)
│   │   └── parser.py               # Extração de texto de PDFs (para uso futuro)
│   ├── retrieval/
│   │   ├── vector_db.py    # Conexão Qdrant, geração de embeddings, upsert
│   │   └── hybrid_search.py # Busca híbrida vetorial + BM25 com RRF
│   ├── api/
│   │   ├── main.py         # FastAPI: endpoints, orquestração, rate limiting
│   │   └── llm_chain.py    # Prompt engineering, chamada ao LLM, logging
│   └── utils/              # Logger, helpers de Parquet, constantes, config .env
├── tests/
│   ├── test_parser.py
│   ├── test_chunker.py
│   ├── test_hybrid_search.py
│   ├── test_llm_chain.py
│   └── test_api.py
├── docs/                   # ADRs, benchmarks, análises de custo
├── requirements.txt        # Dependências fixadas com ==
├── .env.example
├── .gitignore
└── README.md
```

---

## Pipeline de Execução

```
JSON metadados (ementas, datas, tipos)
    │
    ▼
limpar_json_aneel.py  →  data/aneel_limpo_completo.json
    │
    ▼
chunker_json.py  →  data/processed/chunks_json_todos.parquet
                     (16.167 chunks · 15.528 atos · 2016/2021/2022)
    │
    ▼
vector_db.py   →  Qdrant (vetores + BM25 indexados)
    │
    ▼ (produção)
hybrid_search.py  ←  pergunta do usuário
    │
    ▼
llm_chain.py   →  resposta com citação
    │
    ▼
main.py (FastAPI)  →  POST /query → JSON
```

---

## Convenções de Código

### Geral
- Python 3.10+; usar type hints em todas as funções públicas
- Docstrings em português no estilo Google Docstrings
- Variáveis e funções em `snake_case`; classes em `PascalCase`
- Nunca hardcodar chaves de API, URLs ou parâmetros de configuração — usar sempre `.env` via `python-dotenv`
- Constantes do projeto (chunk size, top-k, nome da collection) ficam em `src/utils/`

### Logs
- Usar o logger configurado em `src/utils/` — nunca `print()` em código de produção
- Logar sempre: início/fim de operações longas, erros com traceback, métricas de custo (tokens, latência)

### Tratamento de Erros
- Tratar explicitamente: erros de rede (timeout, 404), PDFs corrompidos/escaneados, falhas na API OpenAI/Anthropic
- Usar retentativas com backoff exponencial em chamadas de API externas

### Dados e Arquivos
- PDFs nunca entram no Git (`data/raw/*.pdf` está no `.gitignore`)
- Parquets processados também não são versionados (`data/processed/`)
- Somente o JSON de metadados original vai para o repositório

---

## Módulos e Responsabilidades

### `src/p1_ingestion/limpar_json_aneel.py`
- Lê o JSON de metadados bruto da ANEEL
- Filtra e normaliza campos (datas, tipos de ato, URLs)
- Gera `data/aneel_limpo_completo.json` e `data/aneel_vigentes_completo.json`

### `src/p1_ingestion/chunker_json.py` — principal da Fase 1
- Lê as ementas de `aneel_limpo_completo.json`
- Gera chunks usando `RecursiveCharacterTextSplitter` do LangChain
- Chunk size: `CHUNK_SIZE` (600 tokens), overlap: `CHUNK_OVERLAP` (90 tokens, ~15%)
- Cada chunk carrega metadados: número do ato, tipo, data, URL
- Output: `data/processed/chunks_json_todos.parquet` (16.167 chunks · 2,4 MB)

### `src/p1_ingestion/baixar_pdfs_aneel.py` — NÃO USAR
- Script de download dos PDFs completos
- Bloqueado pelo Cloudflare do site da ANEEL (apenas 29 de ~17 mil baixaram)
- Mantido no repo para referência/tentativas futuras

### `src/p1_ingestion/parser.py` — para uso futuro
- Extração de texto de PDFs via PyMuPDF
- Não utilizado na pipeline atual (depende dos PDFs completos)

### `src/retrieval/vector_db.py`
- Collection Qdrant: 1536 dimensões, distância cosseno, índice BM25
- Geração de embeddings em batches (otimizar custo OpenAI)
- Funções: verificar collection, contar pontos, apagar e reindexar

### `src/retrieval/hybrid_search.py`
- Busca vetorial (semântica) + BM25 (lexical) em paralelo
- Fusão com **Reciprocal Rank Fusion (RRF)** — não normalizar scores manualmente
- Retornar Top-K chunks com scores e metadados completos
- RRF é obrigatório: não substituir por média de scores

### `src/api/main.py`
- Endpoint principal: `POST /query` (pergunta → resposta com citações)
- Validação de entrada com **Pydantic**
- Rate limiting para proteger custos de API
- Health check: `GET /health`
- Swagger automático em `/docs`

### `src/api/llm_chain.py`
- Montar prompt: system prompt anti-alucinação + contexto (chunks) + pergunta
- **Toda resposta deve citar**: número da Resolução/Portaria, artigo, inciso
- Instrução de recusa obrigatória: se o contexto não contiver a resposta, retornar `"Não encontrado nos atos normativos consultados."` — nunca inferir
- Logar cada chamada: pergunta, tokens, latência, modelo

#### System Prompt Base (não alterar sem revisão da equipe)
```
Você é um especialista em regulação do setor elétrico brasileiro.
Responda EXCLUSIVAMENTE com base nos trechos fornecidos abaixo.
Toda afirmação deve citar o ato normativo de origem (ex: "conforme o Art. 3º
da Resolução Normativa ANEEL nº 687/2016").
Se a informação não constar nos documentos, responda:
"Não encontrado nos atos normativos consultados."
Nunca infira, suponha ou complete com conhecimento externo.
```

---

## Variáveis de Ambiente

Sempre ler do `.env` via `python-dotenv`. Nunca commitar o `.env` real.

```env
OPENAI_API_KEY=sk-...
GROQ_API_KEY=sk-ant-...
QDRANT_URL=http://localhost:6333
QDRANT_API_KEY=
CHUNK_SIZE=600
CHUNK_OVERLAP=90
TOP_K_RETRIEVAL=5
LLM_MODEL=llama-3.3-70b-versatile
```

---

## Testes

- Rodar testes antes de qualquer PR: `pytest tests/`
- Usar `data/samples/` para testes de retrieval — nunca a base completa
- `test_api.py` usa `TestClient` do FastAPI (sem servidor real)
- Manter cobertura mínima nos módulos críticos: `parser.py`, `chunker.py`, `hybrid_search.py`, `llm_chain.py`

---

## Decisões de Arquitetura (ADRs resumidos)

| Decisão | Escolha | Motivo |
|---|---|---|
| Vector Store | Qdrant | Suporte nativo a busca híbrida vetorial + BM25 |
| Embeddings | `text-embedding-3-small` | Custo-benefício: qualidade suficiente a menor custo que `large` |
| Fusão de rankings | RRF | Elimina necessidade de normalizar scores de origens diferentes |
| Serialização | Parquet | Leitura colunar rápida, tipos nativos, compressão eficiente vs. CSV |
| Download de PDFs | Bloqueado (Cloudflare) | Site da ANEEL bloqueia automação — plano B: ementas do JSON |
| Fonte dos chunks | Ementas do JSON | PDFs inacessíveis via automação; ementas cobrem tema/escopo dos atos |
| Anti-alucinação | Instrução de recusa explícita no prompt | Garante que o modelo não infere além do contexto |

---

## O Que Nunca Fazer

- ❌ Hardcodar chaves de API ou URLs no código
- ❌ Usar `print()` em vez do logger configurado
- ❌ Commitar `.env`, PDFs ou Parquets processados
- ❌ Substituir RRF por média simples de scores
- ❌ Deixar o LLM responder além do contexto recuperado (zero inferência externa)
- ❌ Alterar o system prompt sem revisão da equipe
- ❌ Processar a base completa em testes — usar sempre `data/samples/`

---

## Métricas de Avaliação (Benchmark)

O sistema é avaliado por três métricas em `docs/`:

- **Faithfulness**: a resposta é fiel aos documentos recuperados?
- **Answer Relevance**: a resposta resolve a dúvida do especialista?
- **Citation Accuracy**: resoluções e datas citadas estão corretas?