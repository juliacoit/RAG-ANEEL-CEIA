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

# Estrutura de Pastas

## Visão Geral

```
rag-aneel-2016/
├── data/
│   ├── raw/
│   ├── processed/
│   └── samples/
├── src/
│   ├── ingestion/
│   │   ├── downloader.py
│   │   ├── parser.py
│   │   └── chunker.py
│   ├── retrieval/
│   │   ├── vector_db.py
│   │   └── hybrid_search.py
│   ├── api/
│   │   ├── main.py
│   │   └── llm_chain.py
│   └── utils/
├── tests/
├── docs/
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```

---

## `data/` — Gestão de Dados (Pessoa 1)

Repositório central de todos os dados do projeto. Separado em três subpastas para garantir rastreabilidade e reprodutibilidade: sempre é possível remontar os dados processados a partir dos dados brutos.

### `data/raw/`

Contém os arquivos na sua forma **mais crua e original**, exatamente como vieram da fonte:

- O arquivo JSON de metadados original (`biblioteca_aneel_gov_br_legislacao_2016_metadados.json`), com as URLs, títulos, datas e categorias de cada ato normativo.
- Os **PDFs baixados** diretamente do site da ANEEL/MME (Resoluções, Portarias, Despachos de 2016).

> ⚠️ **Atenção:** Os PDFs **não devem ser versionados no Git** (são pesados e contêm dados públicos já disponíveis na fonte). O `.gitignore` deve excluir `data/raw/*.pdf` e `data/raw/**/*.pdf`. Somente o JSON de metadados pode entrar no repositório.

### `data/processed/`

Armazena os arquivos `.parquet` gerados pela pipeline de ingestão (Fase 1). Cada linha do Parquet representa um **chunk de texto** extraído de um PDF, já limpo, fatiado e enriquecido com seus metadados (número do ato, data, tipo, URL de origem, posição no documento).

O formato Parquet foi escolhido por ser **colunar e binário**: leitura muito mais rápida que CSV, suporte nativo a tipos de dados (strings, inteiros, datas) e compressão eficiente. É o formato de entrada da Fase 2 (indexação no Qdrant).

### `data/samples/`

Uma amostra pequena e representativa dos dados — geralmente 50 a 200 chunks — usada para **testes rápidos** sem precisar processar toda a base. Permite que qualquer membro da equipe valide mudanças no código de retrieval ou no prompt sem aguardar o pipeline completo.

---

## `src/` — Código-Fonte (A Alma do Projeto)

Todo o código executável está aqui, organizado pelas três fases da pipeline. A separação por fase facilita o trabalho paralelo entre as três pessoas e isola responsabilidades.

---

### `src/ingestion/` — Fase 1: Ingestão (Pessoa 1)

Responsável por transformar o JSON de metadados e os PDFs brutos em chunks limpos e estruturados prontos para indexação.

#### `downloader.py`

Script de download assíncrono dos PDFs. Usa `asyncio` + `aiohttp` para baixar dezenas (ou centenas) de documentos em paralelo, sem bloquear o processo principal.

Responsabilidades:
- Ler a lista de URLs do JSON de metadados.
- Baixar cada PDF de forma assíncrona com controle de concorrência (semáforo para não sobrecarregar o servidor).
- Salvar os PDFs em `data/raw/` com nomes padronizados (ex: `resolucao_normativa_687_2016.pdf`).
- Tratar erros de rede (timeout, 404, redirect) com retentativas.

#### `parser.py`

Extrator de texto bruto a partir dos PDFs usando **PyMuPDF (fitz)**. É aqui que a qualidade da extração determina tudo o que vem depois.

Responsabilidades:
- Abrir cada PDF e iterar página a página.
- Extrair texto preservando a ordem de leitura correta (crítico para PDFs com layout em duas colunas).
- Aplicar limpeza de texto: remover cabeçalhos/rodapés repetidos, hifenizações incorretas, caracteres especiais, espaços duplos e linhas em branco excessivas.
- Detectar e sinalizar páginas com baixa qualidade de extração (PDFs escaneados sem OCR).

#### `chunker.py`

Implementa a lógica de **fatiamento do texto limpo** em chunks menores, usando o `RecursiveCharacterTextSplitter` do LangChain.

Responsabilidades:
- Receber o texto limpo e os metadados de cada documento.
- Fatiar o texto em chunks de tamanho controlado (ex: 600 tokens, com overlap de ~15%).
- Juntar cada chunk com seus metadados de origem (número do ato, tipo, data, URL, número da página).
- Serializar o resultado em arquivos `.parquet` em `data/processed/`.

> **Decisão de design:** O overlap entre chunks garante que informações que caem na fronteira entre dois chunks não se percam durante o retrieval.

---

### `src/retrieval/` — Fase 2: Indexação e Busca Híbrida (Pessoa 2)

Responsável por transformar os chunks do Parquet em vetores indexados no Qdrant e implementar a lógica de busca híbrida.

#### `vector_db.py`

Configura e gerencia a conexão com o **Qdrant** e o processo de geração de embeddings.

Responsabilidades:
- Conectar ao Qdrant (local via Docker ou instância cloud).
- Criar a collection com as configurações corretas: dimensão do vetor (`text-embedding-3-small` → 1536 dimensões), métrica de distância (cosine), e índice BM25 para busca lexical.
- Ler o Parquet e gerar os vetores chamando a API da OpenAI (`text-embedding-3-small`) em lotes (batches) para otimizar custo e velocidade.
- Fazer o upsert dos vetores + payloads (metadados) no Qdrant.
- Expor funções auxiliares: verificar se a collection existe, contar pontos indexados, apagar e reindexar.

#### `hybrid_search.py`

Implementa a **busca híbrida** combinando busca vetorial densa e busca lexical BM25 — o coração do sistema de retrieval.

Responsabilidades:
- Receber a pergunta do usuário e gerar seu embedding de consulta.
- Executar as duas buscas em paralelo: vetorial (semântica) e BM25 (lexical/palavras-chave).
- Combinar os resultados usando **Reciprocal Rank Fusion (RRF)**: cada documento recebe uma pontuação que integra sua posição nos dois rankings, eliminando a necessidade de normalizar scores de origens diferentes.
- Retornar os Top-K chunks com seus scores e metadados completos.

> **Por que híbrida?** A busca vetorial captura semântica ("qual a regra sobre pequenas centrais hidrelétricas") mas pode perder siglas exatas. O BM25 garante que termos como **PCH**, **TUSD**, **CCEE** e **ANEEL** sejam recuperados com precisão lexical. A combinação elimina os pontos cegos de cada abordagem isolada.

---

### `src/api/` — Fase 3: Backend e Geração (Pessoa 3)

Expõe o sistema como uma API REST e implementa a cadeia de geração de respostas com LLM.

#### `main.py`

Aplicação **FastAPI** que serve como ponto de entrada do sistema.

Responsabilidades:
- Definir o endpoint principal: `POST /query` (recebe a pergunta, retorna a resposta com citações).
- Validar os dados de entrada com modelos Pydantic.
- Orquestrar o fluxo completo: chamar `hybrid_search.py` → montar o contexto → chamar `llm_chain.py` → retornar a resposta estruturada.
- Implementar rate limiting para proteger os custos de API.
- Expor um endpoint de health check (`GET /health`) para monitoramento.
- Gerar documentação automática via Swagger UI (`/docs`).

#### `llm_chain.py`

Implementa o **Prompt Engineering** e a chamada ao LLM final (Claude 3.5 Sonnet ou GPT-4o).

Responsabilidades:
- Montar o prompt completo: system prompt com instruções anti-alucinação + contexto (chunks recuperados) + pergunta do usuário.
- Forçar citação da fonte exata (número da Resolução/Portaria, artigo, inciso) em toda resposta.
- Implementar a instrução de recusa: quando o contexto recuperado não contém a informação, o modelo deve responder `"Não encontrado nos atos normativos consultados"` em vez de inferir.
- Chamar a API do LLM escolhido e parsear a resposta.
- Logar cada chamada (pergunta, tokens usados, latência, modelo) para análise de custo e benchmark.

> **Exemplo de system prompt base:**
> ```
> Você é um especialista em regulação do setor elétrico brasileiro.
> Responda EXCLUSIVAMENTE com base nos trechos fornecidos abaixo.
> Toda afirmação deve citar o ato normativo de origem (ex: "conforme o Art. 3º
> da Resolução Normativa ANEEL nº 687/2016").
> Se a informação não constar nos documentos, responda:
> "Não encontrado nos atos normativos consultados."
> Nunca infira, suponha ou complete com conhecimento externo.
> ```

---

### `src/utils/`

Funções e classes auxiliares compartilhadas entre os módulos das três fases. Evita duplicação de código.

O que costuma ficar aqui:
- Configuração e setup do **logger** (formato padronizado, níveis de log, saída para arquivo e console).
- Funções de leitura/escrita de Parquet.
- Formatadores de texto (normalização de strings, remoção de acentos para comparação).
- Helper de configuração de variáveis de ambiente (leitura do `.env`).
- Constantes do projeto (nomes de collections, dimensões de embedding, limites de chunk).

---

## `tests/` — Testes Unitários e de Integração

Testes automatizados para garantir que cada componente funciona corretamente de forma isolada (unitários) e que as fases funcionam em conjunto (integração).

Estrutura recomendada dentro de `tests/`:
```
tests/
├── test_parser.py        # Testa extração de texto de PDFs de exemplo
├── test_chunker.py       # Valida tamanho e overlap dos chunks gerados
├── test_hybrid_search.py # Testa retrieval com perguntas de referência
├── test_llm_chain.py     # Testa montagem do prompt e parsing da resposta
└── test_api.py           # Testa os endpoints FastAPI (com TestClient)
```

---

## `docs/` — Documentação e Relatórios de Benchmark

Documentação técnica complementar e os relatórios gerados durante o processo de avaliação do sistema.

O que colocar aqui:
- **Relatórios de benchmark:** resultados por pergunta do conjunto de avaliação (acurácia, precisão da citação, taxa de alucinação).
- **ADRs (Architecture Decision Records):** decisões técnicas importantes e o raciocínio por trás delas (ex: "por que Qdrant e não Pinecone", "por que `text-embedding-3-small` e não `large`").
- Análises de custo de API (tokens consumidos por fase).
- Guia de configuração do ambiente para novos membros.

---

## Arquivos Raiz

### `requirements.txt`

Lista todas as dependências Python com versões fixadas (`==`), garantindo que o ambiente de qualquer membro da equipe seja idêntico.

Dependências principais esperadas:
```
aiohttp==3.9.x
pymupdf==1.24.x
pandas==2.2.x
pyarrow==16.x
langchain==0.2.x
qdrant-client==1.9.x
openai==1.x
anthropic==0.x
fastapi==0.111.x
uvicorn==0.30.x
python-dotenv==1.0.x
```

### `.env.example`

Modelo do arquivo de variáveis de ambiente. O arquivo `.env` real (com as chaves secretas) **nunca entra no repositório** — apenas este exemplo com os nomes das variáveis e valores fictícios.

```env
# APIs de IA
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...

# Qdrant
QDRANT_URL=http://localhost:6333
QDRANT_API_KEY=

# Configurações da Pipeline
CHUNK_SIZE=600
CHUNK_OVERLAP=90
TOP_K_RETRIEVAL=5
LLM_MODEL=claude-3-5-sonnet-20241022
```

### `.gitignore`

Define o que **não deve ser versionado** no Git. Itens críticos para este projeto:

```gitignore
# Dados brutos pesados
data/raw/*.pdf
data/raw/pdfs/

# Dados processados (reproduzíveis a partir dos raws)
data/processed/

# Secrets
.env

# Ambiente virtual Python
venv/
.venv/
__pycache__/
*.pyc

# Logs e temporários
*.log
.DS_Store
```

### `README.md`

Documento de entrada do repositório. Deve conter: descrição do projeto, diagrama da arquitetura, instruções de setup (clonar → instalar dependências → configurar `.env` → executar cada fase), e referência ao benchmark.

---

## Fluxo de Execução Completo

```
JSON (metadados)
      │
      ▼
downloader.py  ──►  data/raw/*.pdf
      │
      ▼
parser.py      ──►  texto limpo (em memória)
      │
      ▼
chunker.py     ──►  data/processed/chunks.parquet
      │
      ▼
vector_db.py   ──►  Qdrant (vetores + BM25 indexados)
      │
      ▼ (em produção)
hybrid_search.py  ◄──  pergunta do usuário
      │
      ▼
llm_chain.py   ──►  resposta com citação
      │
      ▼
main.py (FastAPI)  ──►  POST /query → JSON de resposta
```

---

*Documento gerado para o projeto RAG-ANEEL 2016 · Sprint Ágil 12 Dias*


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
