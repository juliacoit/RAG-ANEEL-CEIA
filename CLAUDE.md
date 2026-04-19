# CLAUDE.md — RAG ANEEL ⚡

Guia de contexto para o agente Claude Code trabalhar no projeto RAG ANEEL: sistema de busca e resposta sobre legislação do setor elétrico brasileiro usando arquitetura RAG.

---

## Visão Geral do Projeto

Sistema RAG (Retrieval-Augmented Generation) que permite consultas em linguagem natural sobre atos normativos da ANEEL (Resoluções, Portarias, Despachos), retornando respostas com citação direta da fonte e zero alucinações.

---

## Stack Tecnológica

| Camada | Tecnologia |
|---|---|
| Linguagem | Python 3.10+ |
| Orquestração | LangChain |
| Vector Store | Qdrant (busca híbrida: vetorial + BM25) |
| Embeddings | OpenAI `text-embedding-3-small` (1536 dims) |
| LLMs | Claude 3.5 Sonnet (`claude-3-5-sonnet-20241022`) / GPT-4o |
| API | FastAPI + Uvicorn |
| Serialização | Parquet (via pandas + pyarrow) |
| Download | aiohttp + asyncio |
| Extração PDF | PyMuPDF (fitz) |

---

## Estrutura de Pastas

```
rag-aneel-2016/
├── data/
│   ├── raw/          # PDFs originais + JSON de metadados (PDFs não versionados)
│   ├── processed/    # Chunks em .parquet (não versionados)
│   └── samples/      # Amostra de 50–200 chunks para testes rápidos
├── src/
│   ├── ingestion/
│   │   ├── downloader.py   # Download assíncrono dos PDFs
│   │   ├── parser.py       # Extração e limpeza de texto via PyMuPDF
│   │   └── chunker.py      # Fatiamento com RecursiveCharacterTextSplitter
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
JSON metadados
    │
    ▼
downloader.py  →  data/raw/*.pdf
    │
    ▼
parser.py      →  texto limpo (memória)
    │
    ▼
chunker.py     →  data/processed/chunks.parquet
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

### `src/ingestion/downloader.py`
- Download assíncrono com `asyncio` + `aiohttp`
- Controle de concorrência via semáforo (não sobrecarregar o servidor ANEEL)
- Nomes de arquivo padronizados: `resolucao_normativa_687_2016.pdf`
- Retentativas em erros de rede

### `src/ingestion/parser.py`
- Extração com PyMuPDF preservando ordem de leitura (atenção a PDFs com duas colunas)
- Limpar: cabeçalhos/rodapés repetidos, hifenizações, espaços duplos, linhas em branco
- Sinalizar páginas com baixa qualidade (PDFs escaneados sem OCR)

### `src/ingestion/chunker.py`
- `RecursiveCharacterTextSplitter` do LangChain
- Chunk size: `CHUNK_SIZE` (padrão 600 tokens), overlap: `CHUNK_OVERLAP` (padrão 90 tokens, ~15%)
- Cada chunk carrega metadados: número do ato, tipo, data, URL, página
- Output: `.parquet` em `data/processed/`

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
ANTHROPIC_API_KEY=sk-ant-...
QDRANT_URL=http://localhost:6333
QDRANT_API_KEY=
CHUNK_SIZE=600
CHUNK_OVERLAP=90
TOP_K_RETRIEVAL=5
LLM_MODEL=claude-3-5-sonnet-20241022
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
| Download | aiohttp assíncrono | Dezenas de PDFs em paralelo sem bloquear o processo |
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