# RAG ANEEL — Sistema de IA para Legislação do Setor Elétrico

Sistema RAG (Retrieval-Augmented Generation) que permite consultas em linguagem natural sobre atos normativos da ANEEL (Resoluções, Despachos, Portarias), retornando respostas com **citação direta da fonte e zero alucinação**.

> **Escala:** 562.152 chunks · 25.202 arquivos · 3 anos de legislação (2016, 2021, 2022) · Custo operacional: $0

---

## Sumário

1. [Pré-requisitos](#1-pré-requisitos)
2. [Instalação](#2-instalação)
3. [Configuração](#3-configuração)
4. [Dados de entrada](#4-dados-de-entrada)
5. [Executar o pipeline](#5-executar-o-pipeline)
6. [Subir a API](#6-subir-a-api)
7. [Subir a interface Streamlit](#7-subir-a-interface-streamlit)
8. [Testar sem gastar tokens](#8-testar-sem-gastar-tokens)
9. [Avaliação RAGAS](#9-avaliação-ragas)
10. [Estrutura de pastas](#10-estrutura-de-pastas)
11. [Arquitetura](#11-arquitetura)
12. [Limitações conhecidas](#12-limitações-conhecidas)

---

## 1. Pré-requisitos

### 1.1 Python

Versão recomendada: **Python 3.11 ou 3.12** (3.13+ pode ter incompatibilidades).

```bash
python --version   # deve mostrar 3.11.x ou 3.12.x
```

### 1.2 Docker Desktop

Necessário para rodar o Qdrant (banco vetorial).

Instalar em: https://www.docker.com/products/docker-desktop/

Após instalar, abra o Docker Desktop e mantenha-o rodando.

### 1.3 Tesseract OCR

Necessário para extrair texto de PDFs escaneados.

Download (Windows 64-bit): https://github.com/UB-Mannheim/tesseract/wiki

Durante a instalação, marcar **"Additional language data → Portuguese"**.

Instalar no caminho padrão: `C:\Program Files\Tesseract-OCR\`

Verificar:
```bash
tesseract --version   # deve mostrar: tesseract 5.x.x
```

Se o comando não for reconhecido, adicionar ao PATH do sistema: `C:\Program Files\Tesseract-OCR\`

### 1.4 Poppler

Necessário para converter páginas PDF em imagens (pre-processamento de OCR).

**Instalar dentro da pasta do projeto** (não no sistema):

1. Baixar o release mais recente em: https://github.com/oschwartz10612/poppler-windows/releases
2. Extrair com a estrutura abaixo:

```
RAG-ANEEL-CEIA/
└── poppler/
    └── Library/
        └── bin/
            ├── pdftoppm.exe
            ├── pdfinfo.exe
            └── ...
```

Verificar:
```bash
python -c "from pathlib import Path; p = Path('poppler/Library/bin'); print('OK' if p.exists() else 'NÃO encontrado')"
```

### 1.5 7-Zip (somente para arquivos RAR)

Necessário para extrair atos normativos distribuídos em `.rar`.

Download: https://www.7-zip.org/

Instalar no caminho padrão: `C:\Program Files\7-Zip\`

### 1.6 Chaves de API

| Chave | Onde obter | Custo |
|---|---|---|
| `GROQ_API_KEY` | https://console.groq.com | Gratuito (sem cartão) |
| `DEEPSEEK_API_KEY` | https://platform.deepseek.com | Pago (~$0.05/100 perguntas) — **opcional**, usado só como fallback |

---

## 2. Instalação

```bash
# Clonar o repositório
git clone <url-do-repo>
cd RAG-ANEEL-CEIA

# Instalar dependências Python
pip install -r requirements.txt
```

> Na primeira execução do pipeline, o modelo de embeddings `paraphrase-multilingual-MiniLM-L12-v2` (~120 MB) será baixado automaticamente pelo `sentence-transformers` e ficará em cache.

---

## 3. Configuração

```bash
# Windows
copy env.example .env

# Linux / macOS
cp env.example .env
```

Editar o `.env` e preencher as chaves:

```env
GROQ_API_KEY=gsk_...              # obrigatório
DEEPSEEK_API_KEY=sk-...           # opcional (fallback automático)
LLM_PROVIDER=groq
LLM_MODEL=llama-3.3-70b-versatile
DEEPSEEK_MODEL=deepseek-reasoner
QDRANT_URL=http://localhost:6333
QDRANT_COLLECTION=aneel_chunks
TOP_K_RETRIEVAL=10
OPTIMIZER_TIMEOUT=12
```

Subir o Qdrant:

```bash
docker compose up -d

# Verificar se está rodando
docker ps   # deve mostrar o container "qdrant_aneel" com status "Up"
```

Dashboard opcional: http://localhost:6333/dashboard

---

## 4. Dados de entrada

O pipeline parte de três arquivos JSON de metadados da ANEEL. Eles devem estar em `data/raw/` com os seguintes nomes exatos:

```
data/raw/
├── biblioteca_aneel_gov_br_legislacao_2016_metadados.json
├── biblioteca_aneel_gov_br_legislacao_2021_metadados.json
└── biblioteca_aneel_gov_br_legislacao_2022_metadados.json
```

> Esses arquivos **não estão versionados** no repositório por causa do tamanho. Solicitá-los à equipe responsável pela P1 ou ao orientador do projeto.

### Modo alternativo (sem os JSONs brutos)

Se os arquivos Parquet já foram gerados e compartilhados (`chunks_completo_unificado.parquet`, 111 MB), colocá-los em `data/processed/` e pular diretamente para o [Passo 5.3 — Indexação](#53-só-a-indexação-parquets-já-prontos).

---

## 5. Executar o pipeline

O script `scripts/setup_pipeline.py` orquestra todas as fases. O pipeline completo faz:

1. **P1a** — Limpeza dos JSONs brutos
2. **P1b** — Download dos PDFs, HTMLs, XLSXs e RARs (25.202 arquivos)
3. **P1c** — Chunking das ementas
4. **P1d** — Extração de texto dos PDFs/HTMLs/XLSXs
5. **P1e** — União dos parquets (ementas + PDFs → 562k chunks)
6. **P2** — Indexação no Qdrant + criação do índice BM25

### 5.1 Pipeline completo (primeira vez — ~2–4 horas)

```bash
python scripts/setup_pipeline.py
```

### 5.2 Modo teste rápido (50 documentos — ~5 min)

Ideal para verificar se o ambiente está configurado corretamente:

```bash
python scripts/setup_pipeline.py --limite 50 --testar
```

### 5.3 Só a indexação (parquets já prontos)

```bash
python scripts/setup_pipeline.py --apenas-p2
```

Para reindexar do zero (apaga a coleção existente):

```bash
python scripts/setup_pipeline.py --apenas-p2 --resetar
```

### 5.4 Outras opções úteis

```bash
# Só download de PDFs (P1b)
python scripts/setup_pipeline.py --apenas-download

# Só parser (PDFs já baixados, P1d)
python scripts/setup_pipeline.py --apenas-parser

# Sem OCR (mais rápido, mas perde PDFs escaneados)
python scripts/setup_pipeline.py --sem-ocr

# Controlar workers do parser (padrão: 16)
python scripts/setup_pipeline.py --apenas-parser --workers 8
```

---

## 6. Subir a API

Antes de subir a API, confirmar que o Qdrant está rodando (`docker compose up -d`) e que a indexação foi concluída.

```bash
# Desenvolvimento (com reload automático)
python -m uvicorn src.api.main:app --reload --port 8000

# Produção
python -m uvicorn src.api.main:app --host 0.0.0.0 --port 8000
```

Swagger UI disponível em: http://localhost:8000/docs

### Testar via curl

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"pergunta": "O que é microgeração distribuída?", "historico": []}'
```

### Verificar saúde do sistema

```bash
curl http://localhost:8000/health
```

---

## 7. Subir a interface Streamlit

Com a API rodando na porta 8000:

```bash
python -m streamlit run app.py
```

Acesse em: http://localhost:8501

A interface oferece:
- Campo de pergunta em linguagem natural
- Filtros por tipo de ato, ano, número e palavra-chave
- Resposta com fontes citadas e scores semântico + BM25
- Histórico dos últimos 10 turnos
- Indicador de cache hit e tipo de resposta detectado
- Dashboard de métricas das consultas

---

## 8. Testar sem gastar tokens

Para verificar o retrieval de forma isolada (sem chamar nenhum LLM):

```python
from src.p2_search.p2_indexar import conectar_qdrant, carregar_indices, buscar

qclient = conectar_qdrant()
bm25, bm25_ids = carregar_indices(qclient)

resultados = buscar(
    "valor TUST ciclo 2021-2022",
    qclient, bm25, bm25_ids,
    n_resultados=5
)
for r in resultados:
    print(r["tipo_nome"], r["numero"], r["ano"], "| score:", r["score_final"])
    print(" ", r["texto"][:200])
    print()
```

### Checklist de verificação do ambiente

```bash
# 1. Python OK?
python --version

# 2. Dependências OK?
python -c "import fitz, pdfplumber, qdrant_client, sentence_transformers; print('OK')"

# 3. Qdrant OK?
python -c "from qdrant_client import QdrantClient; QdrantClient(url='http://localhost:6333').get_collections(); print('Qdrant OK')"

# 4. Índice BM25 OK?
python -c "
import pickle
from pathlib import Path
bm25 = pickle.load(open('db/bm25/bm25_index.pkl','rb'))
ids  = pickle.load(open('db/bm25/bm25_ids.pkl','rb'))
print(f'BM25 OK — {len(ids):,} documentos')
"

# 5. Busca híbrida OK?
python -c "
from src.p2_search.p2_indexar import conectar_qdrant, carregar_indices, buscar
qclient = conectar_qdrant()
bm25, ids = carregar_indices(qclient)
r = buscar('tarifa TUSD distribuição', qclient, bm25, ids, n_resultados=1)
print('Busca OK:', r[0]['titulo'] if r else 'sem resultado')
"
```

---

## 9. Avaliação RAGAS

O sistema inclui um banco de 45 perguntas em 10 categorias temáticas para avaliação com métricas RAGAS.

```bash
# Rodar avaliação completa
python tests/banco_perguntas.py

# Opções
python tests/banco_perguntas.py --limite 10
python tests/banco_perguntas.py --categoria microgeracao
python tests/banco_perguntas.py --tipo semantica
python tests/banco_perguntas.py --saida resultados.json
```

Métricas avaliadas:
- **Faithfulness** — a resposta é fiel aos chunks recuperados?
- **Response Relevancy** — a resposta resolve a dúvida?
- **Context Precision** — os chunks recuperados são relevantes?

---

## 10. Estrutura de pastas

```
RAG-ANEEL-CEIA/
├── data/
│   ├── raw/                              # JSONs brutos da ANEEL (obrigatório para P1)
│   │   ├── biblioteca_aneel_..._2016_metadados.json
│   │   ├── biblioteca_aneel_..._2021_metadados.json
│   │   └── biblioteca_aneel_..._2022_metadados.json
│   ├── aneel_vigentes_completo.json      # gerado pelo pipeline
│   └── processed/
│       ├── chunks_json_todos.parquet          # ementas (~2.4 MB)
│       ├── chunks_pdf_completo.parquet        # PDFs (~grande)
│       └── chunks_completo_unificado.parquet  # 562k chunks unificados (111 MB)
├── db/
│   └── bm25/
│       ├── bm25_index.pkl   # gerado pelo p2_indexar.py
│       └── bm25_ids.pkl
├── pdfs/                    # ~27k arquivos baixados
├── poppler/                 # Poppler local (instalar manualmente)
│   └── Library/bin/
├── scripts/
│   └── setup_pipeline.py   # orquestrador do pipeline completo
├── src/
│   ├── p1_ingestion/
│   │   ├── limpar_json_aneel.py   # limpeza dos JSONs brutos
│   │   ├── baixar_pdfs_aneel.py   # download de PDFs/HTMLs/XLSXs/RARs
│   │   ├── chunker_json.py        # chunking das ementas
│   │   ├── parser.py              # extração de texto multi-formato
│   │   └── unir_parquets.py       # união dos parquets
│   ├── p2_search/
│   │   └── p2_indexar.py          # indexação Qdrant + BM25 + busca híbrida
│   ├── api/
│   │   ├── main.py                # FastAPI: endpoints, cache LRU
│   │   ├── query_optimizer.py     # rewriting, HyDE, decomposição, memória
│   │   ├── llm_chain.py           # geração com Groq/DeepSeek, 6 prompts adaptativos
│   │   └── analytics.py           # perguntas analíticas via pandas (sem LLM)
│   └── utils/
│       └── logger_metrics.py      # logger configurado
├── tests/
│   └── banco_perguntas.py         # 45 perguntas + avaliação RAGAS
├── app.py                          # interface Streamlit
├── docker-compose.yml
├── requirements.txt
├── env.example
└── SETUP.md                        # detalhes das dependências externas
```

---

## 11. Arquitetura

```
JSONs brutos ANEEL (data/raw/)
       ↓  limpar_json_aneel.py
aneel_vigentes_completo.json
       ↓  baixar_pdfs_aneel.py  (curl_cffi — contorna Cloudflare)
pdfs/  (PDFs · HTMLs · XLSXs · RARs — 25.202 arquivos)
       ↓  parser.py + chunker_json.py
chunks_*.parquet
       ↓  unir_parquets.py
chunks_completo_unificado.parquet  (562.152 chunks · 111 MB)
       ↓  p2_indexar.py
Qdrant  (vetorial: MiniLM-L12 384 dims ou BGE-M3 1024 dims)
BM25    (índice esparso: rank-bm25 sobre 562k docs)
       ↓
API FastAPI  (src/api/main.py — porta 8000)
  ├── query_optimizer.py  →  rewriting · HyDE · decomposição · memória (6 turnos)
  ├── busca híbrida        →  0.6 × semântico + 0.4 × BM25
  └── llm_chain.py        →  Groq llama-3.3-70b (fallback: DeepSeek R1)
       ↓
Streamlit  (app.py — porta 8501)
```

### Busca híbrida

| Canal | Tecnologia | Peso |
|---|---|---|
| Semântico | Qdrant + MiniLM-L12-v2 (384 dims) | 0.6 |
| Lexical | BM25Okapi sobre corpus PT-BR | 0.4 |
| Expansão | Chunks vizinhos (±2 ao chunk base) | — |
| Filtros | Número e ano do ato extraídos via regex | — |
| Fallback | Busca sem filtro quando score < 0.5 | — |

### Upgrade de embedding (opcional — Google Colab)

O sistema detecta automaticamente se o índice BGE-M3 (`bge_index.bin`) existe na raiz do projeto. Se existir, usa BGE-M3 (1024 dims, ~+20% qualidade semântica). Caso contrário, usa MiniLM local.

Para gerar o índice BGE-M3, rodar `p2_indexar.py` em uma máquina com GPU (ex: Google Colab com T4/L4 — gratuito). A indexação de 562k chunks leva ~25 min na GPU L4.

### Stack completa

| Componente | Tecnologia |
|---|---|
| Linguagem | Python 3.11 / 3.12 |
| Vector Store | Qdrant (Docker local, porta 6333) |
| Embeddings | MiniLM-L12-v2 384 dims (local) / BGE-M3 1024 dims (opcional) |
| Busca esparsa | BM25Okapi (`rank-bm25`) |
| LLM principal | Groq — `llama-3.3-70b-versatile` (gratuito, 100k tokens/dia) |
| LLM fallback | DeepSeek R1 — `deepseek-reasoner` (ativado em rate limit) |
| Download | `curl_cffi` (impersona TLS Chrome 124 — contorna Cloudflare) |
| Parser PDF | PyMuPDF + pdfplumber + Tesseract OCR |
| Parser HTML | BeautifulSoup4 |
| Parser XLSX | openpyxl |
| API | FastAPI + Uvicorn |
| Interface | Streamlit |
| Serialização | Parquet (pyarrow) |

---

## 12. Limitações conhecidas

1. **Anos disponíveis:** apenas 2016, 2021 e 2022. Atos de outros anos não estão na base.
2. **Cloudflare:** ~25.202 de ~27k arquivos foram baixados. Documentos ausentes são retornados via fallback "modo menção" — o sistema cita atos que *mencionam* o documento não disponível.
3. **Rate limit Groq:** plano gratuito — ~100k tokens/dia (~60–70 perguntas complexas/dia). Excedido, o sistema faz fallback automático para DeepSeek R1.
4. **Perguntas sobre artigos específicos:** requerem os PDFs completos. Perguntas sobre o tema e escopo dos atos ("do que trata tal resolução") são respondidas com alta precisão.
5. **Parquets não versionados:** `chunks_pdf_completo.parquet` e `chunks_completo_unificado.parquet` não estão no repositório (111 MB). Necessário rodar o pipeline ou solicitá-los à equipe da P1.
