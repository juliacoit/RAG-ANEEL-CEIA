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

### 1.3 7-Zip

Necessário para extrair atos normativos distribuídos em `.rar` e `.7z`. 

**Nota:** O projeto já inclui uma versão portátil na pasta `7-Zip/`. O pipeline utilizará esta versão automaticamente se encontrar o executável `7-Zip/7z.exe`.

Se desejar instalar no sistema: https://www.7-zip.org/


### 1.6 Chaves de API

| Chave | Onde obter | Uso |
|---|---|---|
| `QWEN_API_KEY` | https://dashscope.console.aliyun.com/apiKey | **Obrigatório** — LLM principal (Qwen3-30B-A3B) |
| `GROQ_API_KEY` | https://console.groq.com | Gratuito (sem cartão) — fallback automático |
| `GEMINI_API_KEY` | https://aistudio.google.com | Gratuito — fallback opcional (segundo nível) |

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
# LLM principal
LLM_PROVIDER=qwen
LLM_MODEL=qwen3-30b-a3b
QWEN_API_KEY=sk-...                   # obrigatório
QWEN_BASE_URL=https://dashscope-intl.aliyuncs.com/compatible-mode/v1

# Fallback 1 (gratuito)
GROQ_API_KEY=gsk_...                  # recomendado
GROQ_MODEL=llama-3.3-70b-versatile

# Fallback 2 (opcional)
GEMINI_API_KEY=                       # deixe vazio se não quiser usar
GEMINI_MODEL=gemini-2.0-flash

# Qdrant
QDRANT_URL=http://localhost:6333
QDRANT_COLLECTION=aneel_chunks
TOP_K_RETRIEVAL=10
OPTIMIZER_TIMEOUT=12
```

> **Cadeia de fallback automático:**  
> `Qwen3-30B` → `Groq / Llama-3.3-70B` → `Gemini Flash` → mensagem de erro clara

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

> Esses arquivos **estão versionados** no repositório para facilitar o setup inicial.


### Modo alternativo (sem os JSONs brutos)

Se os arquivos Parquet já foram gerados e compartilhados (`chunks_completo_unificado.parquet`, 111 MB), colocá-los em `data/processed/` e pular diretamente para o [Passo 5.3 — Indexação](#53-só-a-indexação-parquets-já-prontos).

---

## 5. Executar o pipeline

O script `setup_pipeline.py` orquestra todas as fases. O pipeline completo faz:

1. **P1a** — Limpeza dos JSONs brutos
2. **P1b** — Download dos PDFs, HTMLs, XLSXs e RARs (25.202 arquivos)
3. **P1c** — Chunking das ementas
4. **P1d** — Extração de texto dos PDFs/HTMLs/XLSXs
5. **P1e** — União dos parquets (ementas + PDFs → 562k chunks)
6. **P2** — Indexação no Qdrant + criação do índice BM25

### 5.1 Pipeline completo (primeira vez — ~2–4 horas)

```bash
python setup_pipeline.py
```


### 5.2 Modo teste rápido (50 documentos — ~5 min)

Ideal para verificar se o ambiente está configurado corretamente:

```bash
python setup_pipeline.py --limite 50 --testar
```

### 5.3 Só a indexação (parquets já prontos)

```bash
python setup_pipeline.py --apenas-p2
```


Para reindexar do zero (apaga a coleção existente):

```bash
python setup_pipeline.py --apenas-p2 --resetar
```

### 5.4 Outras opções úteis

```bash
# Só download de PDFs (P1b)
python setup_pipeline.py --apenas-download

# Só parser (PDFs já baixados, P1d)
python setup_pipeline.py --apenas-parser

# Sem OCR (mais rápido, mas perde PDFs escaneados)
python setup_pipeline.py --sem-ocr

# Controlar workers do parser (padrão: 16)
python setup_pipeline.py --apenas-parser --workers 8
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
from src.search.indexar import conectar_qdrant, carregar_indices, buscar

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
from src.search.indexar import conectar_qdrant, carregar_indices, buscar
qclient = conectar_qdrant()
bm25, ids = carregar_indices(qclient)
r = buscar('tarifa TUSD distribuição', qclient, bm25, ids, n_resultados=1)
print('Busca OK:', r[0]['titulo'] if r else 'sem resultado')
"
```

---

## 9. Avaliação RAGAS

O sistema inclui um banco de perguntas em categorias temáticas para avaliação automática com métricas RAGAS.

```bash
# Avaliar 5 perguntas (rápido, para testar)
python src/evaluation/eval_runner.py --limite 5 --verbose

# Avaliar por categoria
python src/evaluation/eval_runner.py --categoria microgeracao

# Avaliação completa
python src/evaluation/eval_runner.py
```

Métricas avaliadas:
- **Faithfulness** — a resposta é fiel aos chunks recuperados?
- **Response Relevancy** — a resposta resolve a dúvida?
- **Context Precision** — os chunks recuperados são relevantes?

O resultado é salvo automaticamente em `docs/ragas_resultado_<timestamp>.csv`.

---

## 10. Estrutura de pastas

```
RAG-ANEEL-CEIA/
├── 7-Zip/              # Binários portáteis do 7-Zip (Windows)
├── data/
│   ├── raw/            # JSONs de metadados originais (versionados)
│   └── processed/      # Parquets gerados (ignorados no git)
├── db/                 # Armazenamento do Qdrant e BM25 (ignorados)
├── docs/               # Documentação e resultados de avaliação
├── index_manual/       # Embeddings pré-calculados (ver README interno)
├── pdfs/               # PDFs baixados (ignorados, ~27k arquivos)
├── src/                # Código fonte organizado por módulo
│   ├── api/            # FastAPI, LLM Chain e Analytics
│   ├── evaluation/     # Scripts de avaliação RAGAS e banco de perguntas
│   ├── ingestion/      # Pipeline P1 (limpeza, download, parser, união)
│   ├── search/         # Indexação e lógica de busca híbrida
│   └── utils/          # Funções utilitárias (Windows helpers, logs)
├── tests/              # Testes unitários e de integração
├── app.py              # Interface Streamlit
├── docker-compose.yml  # Configuração do Qdrant
├── requirements.txt    # Dependências Python
└── setup_pipeline.py   # Orquestrador do projeto
```

---

## 11. Arquitetura

O sistema utiliza uma arquitetura de **Busca Híbrida** combinada com **Geração Aumentada (RAG)**:

1. **Recuperação Semântica:** Vetores gerados pelo modelo `paraphrase-multilingual-MiniLM-L12-v2` armazenados no Qdrant.
2. **Recuperação Lexical:** Algoritmo BM25 para termos técnicos específicos e números de resoluções.
3. **Reranking:** Combinação de scores (0.7 semântico + 0.3 BM25) para selecionar os top-k chunks.
4. **Geração:** O modelo Qwen3-30B (ou fallback Groq/Llama-3) processa a pergunta com os chunks recuperados.

### Stack completa

| Componente | Tecnologia |
|---|---|
| Vector Store | Qdrant (Docker local, porta 6333) |
| Embeddings | MiniLM-L12-v2 384 dims (local) / BGE-M3 1024 dims (opcional) |
| Busca esparsa | BM25Okapi (`rank-bm25`) |
| LLM principal | Qwen-30B via Dashscope |
| LLM fallback | Groq / Llama-3.3-70B |
| Download | `curl_cffi` (contorna Cloudflare) |
| Parser PDF | PyMuPDF + pdfplumber |
| API | FastAPI + Uvicorn |
| Interface | Streamlit |

---

## 12. Limitações conhecidas

1. **Anos disponíveis:** apenas 2016, 2021 e 2022.
2. **Cloudflare:** Alguns arquivos podem falhar no download; o sistema usa o "modo menção" como fallback.
3. **Rate limit Groq:** Plano gratuito tem limites diários (usado apenas como fallback).
4. **Parquets não versionados:** Os arquivos processados finais (`.parquet` unificado) não estão no git devido ao tamanho.

