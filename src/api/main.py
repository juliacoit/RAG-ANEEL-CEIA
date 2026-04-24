"""
main.py
=======
Pessoa 3 — LLM Engineer

API FastAPI do sistema RAG ANEEL.
Recebe perguntas em linguagem natural e retorna respostas com citações
dos atos normativos, sem alucinação.

Pré-requisitos:
  - Docker rodando com Qdrant: docker compose up -d
  - python src/p2_search/p2_indexar.py  (indexação feita ao menos uma vez)
  - .env com GROQ_API_KEY

Uso:
  uvicorn src.api.main:app --reload
  uvicorn src.api.main:app --host 0.0.0.0 --port 8000

Endpoints:
  POST /query   — pergunta → resposta com fontes
  GET  /health  — verifica Qdrant e modelo
  GET  /docs    — Swagger UI automático
"""

import logging
import os
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

load_dotenv()

# Garante que src/ está no path independente de onde o uvicorn é chamado
RAIZ = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(RAIZ))

from src.p2_search.p2_indexar import (
    buscar,
    carregar_indices,
    conectar_qdrant,
    NOME_COLECAO,
)
from src.api.llm_chain import gerar_resposta

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Estado global (inicializado no lifespan)
# ---------------------------------------------------------------------------

_estado: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Conecta ao Qdrant e carrega índices BM25 na inicialização."""
    log.info("Iniciando RAG ANEEL API...")
    try:
        qclient = conectar_qdrant()
        bm25, bm25_ids = carregar_indices(qclient)
        _estado["qclient"] = qclient
        _estado["bm25"] = bm25
        _estado["bm25_ids"] = bm25_ids
        count = qclient.count(NOME_COLECAO).count
        log.info(f"Pronto — {count} chunks indexados no Qdrant.")
    except Exception as e:
        log.error(f"Falha na inicialização: {e}")
        log.error("Verifique se o Qdrant está rodando e a indexação foi feita.")
        raise

    yield

    log.info("Encerrando API.")
    _estado.clear()


app = FastAPI(
    title="RAG ANEEL",
    description=(
        "Consulte atos normativos da ANEEL (Resoluções, Portarias, Despachos) "
        "em linguagem natural. Respostas fundamentadas com citação da fonte."
    ),
    version="0.2.0",
    lifespan=lifespan,
)

# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

TOP_K_DEFAULT = int(os.getenv("TOP_K_RETRIEVAL", "5"))


class QueryRequest(BaseModel):
    pergunta: str = Field(..., min_length=5, max_length=1000, examples=["O que é microgeração distribuída?"])
    n_resultados: int = Field(default=TOP_K_DEFAULT, ge=1, le=20)
    filtros: Optional[dict] = Field(
        default=None,
        description="Filtrar por metadados, ex: {'tipo_codigo': 'REH'} ou {'ano_fonte': ['2021', '2022']}",
        examples=[{"tipo_codigo": "REH"}, {"ano_fonte": ["2021", "2022"]}],
    )


class FonteCitada(BaseModel):
    chunk_id: str
    titulo: str
    tipo_nome: str
    numero: str
    ano: str
    assunto: str
    url_pdf: str
    score_final: float
    score_semantico: float
    score_bm25: float
    trecho: str = Field(description="Primeiros 300 caracteres do chunk")


class QueryResponse(BaseModel):
    resposta: str
    fontes: list[FonteCitada]
    modelo: str
    tokens_prompt: int
    tokens_resposta: int
    latencia_total_ms: int
    latencia_llm_ms: int
    system_prompt: str


class HealthResponse(BaseModel):
    status: str
    qdrant_chunks: int
    bm25_docs: int
    modelo_llm: str


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest):
    """
    Recebe uma pergunta e retorna uma resposta fundamentada nos atos normativos
    da ANEEL, com citação das fontes utilizadas.
    """
    if "qclient" not in _estado:
        raise HTTPException(status_code=503, detail="Serviço não inicializado.")

    inicio = time.monotonic()

    # --- Recuperação (P2) ---
    try:
        chunks = buscar(
            query=req.pergunta,
            qclient=_estado["qclient"],
            bm25=_estado["bm25"],
            bm25_ids=_estado["bm25_ids"],
            n_resultados=req.n_resultados,
            filtros=req.filtros,
        )
    except Exception as e:
        log.error(f"Erro na busca: {e}")
        raise HTTPException(status_code=500, detail=f"Erro na recuperação: {e}")

    # Remove chunks com texto vazio (indexados sem conteúdo)
    chunks = [c for c in chunks if c.get("texto", "").strip()]

    log.info(f"Busca retornou {len(chunks)} chunks válidos para: '{req.pergunta[:60]}'")

    # --- Geração (P3) ---
    try:
        resultado = gerar_resposta(req.pergunta, chunks)
    except EnvironmentError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        log.error(f"Erro na geração: {e}")
        raise HTTPException(status_code=500, detail=f"Erro na geração de resposta: {e}")

    latencia_total = int((time.monotonic() - inicio) * 1000)

    fontes = [
        FonteCitada(
            chunk_id=c["chunk_id"],
            titulo=c.get("titulo", ""),
            tipo_nome=c.get("tipo_nome", ""),
            numero=c.get("numero", ""),
            ano=c.get("ano", ""),
            assunto=c.get("assunto", ""),
            url_pdf=c.get("url_pdf", ""),
            score_final=c["score_final"],
            score_semantico=c["score_semantico"],
            score_bm25=c["score_bm25"],
            trecho=c.get("texto", "")[:300],
        )
        for c in chunks
    ]

    return QueryResponse(
        resposta=resultado.texto,
        fontes=fontes,
        modelo=resultado.modelo,
        tokens_prompt=resultado.tokens_prompt,
        tokens_resposta=resultado.tokens_resposta,
        latencia_total_ms=latencia_total,
        latencia_llm_ms=resultado.latencia_ms,
        system_prompt=resultado.system_prompt,
    )


@app.get("/health", response_model=HealthResponse)
def health():
    """Verifica se o sistema está operacional."""
    if "qclient" not in _estado:
        raise HTTPException(status_code=503, detail="Serviço não inicializado.")

    try:
        count = _estado["qclient"].count(NOME_COLECAO).count
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Qdrant inacessível: {e}")

    return HealthResponse(
        status="ok",
        qdrant_chunks=count,
        bm25_docs=len(_estado["bm25_ids"]),
        modelo_llm=os.getenv("LLM_MODEL", "llama-3.3-70b-versatile"),
    )
