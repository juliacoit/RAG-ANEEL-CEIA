"""
main.py — API FastAPI RAG ANEEL
Otimizacoes de latencia:
  1. Cache LRU — evita repetir optimizer+busca para perguntas identicas
  2. Paralelismo — optimizer e embedding rodam simultaneamente
  3. Timeout agressivo no optimizer — fallback rapido se LLM demorar
"""
import hashlib
import logging
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError as FuturesTimeout
from contextlib import asynccontextmanager
from collections import OrderedDict
from pathlib import Path
from typing import Optional, List

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

load_dotenv()

RAIZ = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(RAIZ))

from src.p2_search.p2_indexar import buscar, carregar_indices, conectar_qdrant, NOME_COLECAO
from src.api.llm_chain import gerar_resposta
from src.api.query_optimizer import otimizar_query, QueryOtimizada
from src.api.analytics import responder_analitico, is_pergunta_analitica

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

_estado = {}

# ---------------------------------------------------------------------------
# Cache LRU simples para resultados de busca
# ---------------------------------------------------------------------------
CACHE_MAX    = 100   # maximo de entradas no cache
CACHE_TTL_S  = 300   # 5 minutos de validade

class CacheBusca:
    def __init__(self, max_size=CACHE_MAX, ttl=CACHE_TTL_S):
        self._cache = OrderedDict()
        self._max   = max_size
        self._ttl   = ttl

    def _chave(self, pergunta, filtros):
        raw = "%s|%s" % (pergunta.lower().strip(), str(sorted((filtros or {}).items())))
        return hashlib.md5(raw.encode()).hexdigest()

    def get(self, pergunta, filtros):
        k = self._chave(pergunta, filtros)
        if k not in self._cache:
            return None
        ts, valor = self._cache[k]
        if time.monotonic() - ts > self._ttl:
            del self._cache[k]
            return None
        self._cache.move_to_end(k)
        log.info("Cache hit para: '%s'", pergunta[:50])
        return valor

    def set(self, pergunta, filtros, valor):
        k = self._chave(pergunta, filtros)
        self._cache[k] = (time.monotonic(), valor)
        self._cache.move_to_end(k)
        if len(self._cache) > self._max:
            self._cache.popitem(last=False)

_cache_busca = CacheBusca()

# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("Iniciando RAG ANEEL API...")
    try:
        qclient = conectar_qdrant()
        bm25, bm25_ids = carregar_indices(qclient)
        _estado["qclient"]  = qclient
        _estado["bm25"]     = bm25
        _estado["bm25_ids"] = bm25_ids
        _estado["executor"] = ThreadPoolExecutor(max_workers=4)
        count = qclient.count(NOME_COLECAO).count
        log.info("Pronto — %d chunks indexados no Qdrant.", count)
    except Exception as e:
        log.error("Falha na inicializacao: %s", e)
        raise
    yield
    log.info("Encerrando API.")
    _estado.get("executor", None) and _estado["executor"].shutdown(wait=False)
    _estado.clear()


app = FastAPI(
    title="RAG ANEEL",
    description="Consulte atos normativos da ANEEL em linguagem natural.",
    version="0.5.0",
    lifespan=lifespan,
)

TOP_K_DEFAULT = int(os.getenv("TOP_K_RETRIEVAL", "10"))
OPTIMIZER_TIMEOUT = float(os.getenv("OPTIMIZER_TIMEOUT", "12"))  # segundos


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class HistoricoTurno(BaseModel):
    pergunta:          str
    resposta:          str
    n_esclarecimentos: int = 0

class QueryRequest(BaseModel):
    pergunta:     str = Field(..., min_length=5, max_length=1000)
    n_resultados: int = Field(default=TOP_K_DEFAULT, ge=1, le=20)
    filtros:      Optional[dict] = None
    historico:    Optional[List[HistoricoTurno]] = None

class FonteCitada(BaseModel):
    chunk_id:        str
    titulo:          str
    tipo_nome:       str
    numero:          str
    ano:             str
    assunto:         str
    url_pdf:         str
    score_final:     float
    score_semantico: float
    score_bm25:      float
    trecho:          str

class QueryResponse(BaseModel):
    resposta:          str
    fontes:            list
    modelo:            str
    tokens_prompt:     int
    tokens_resposta:   int
    latencia_total_ms: int
    latencia_llm_ms:   int
    system_prompt:     str
    tipo_resposta:     str = "normal"
    n_esclarecimentos: int = 0
    cache_hit:         bool = False

class HealthResponse(BaseModel):
    status:        str
    qdrant_chunks: int
    bm25_docs:     int
    modelo_llm:    str
    cache_entries: int


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _executar_optimizer(pergunta, historico_dicts, n_esclarecimentos):
    """Roda o optimizer em thread separada para paralelizar com embedding."""
    return otimizar_query(
        pergunta,
        historico=historico_dicts,
        n_esclarecimentos=n_esclarecimentos,
    )


def _executar_busca(otimizado, filtros_finais, n_resultados):
    """Executa todas as queries e retorna chunks deduplicados."""
    log.info("Executando %d queries: %s", len(otimizado.queries), otimizado.queries)
    todos_chunks = []
    vistos = set()

    # Ajusta filtros por tipo de pergunta
    if otimizado.tipo_pergunta in ("comparacao_temporal",):
        # Comparação temporal: remove filtro de ano para não restringir a um único período
        filtros_finais.pop("ano", None)
        filtros_finais.pop("tipo_codigo", None)
    elif otimizado.tipo_pergunta == "comparacao":
        filtros_finais.pop("ano", None)
        filtros_finais.pop("tipo_codigo", None)
    elif otimizado.tipo_pergunta == "agregacao":
        # Agregação: delega para analytics — busca retorna vazio intencionalmente
        log.warning("tipo=agregacao em _executar_busca — deveria ter sido tratado pelo analytics antes")
    elif otimizado.tipo_pergunta == "autoria":
        # Autoria: garante que o filtro de autor está presente se extraído pelo optimizer
        autor = (otimizado.filtros or {}).get("autor", "")
        if autor and "autor" not in filtros_finais:
            filtros_finais["autor"] = autor
    elif otimizado.tipo_pergunta == "hibrida_complexa":
        # Híbrida complexa: remove filtros restritivos para ampliar o escopo da busca
        filtros_finais.pop("numero", None)

    for i, query_str in enumerate(otimizado.queries):
        chunks_q = buscar(
            query=query_str,
            qclient=_estado["qclient"],
            bm25=_estado["bm25"],
            bm25_ids=_estado["bm25_ids"],
            n_resultados=n_resultados,
            filtros=filtros_finais if filtros_finais else None,
            tipo_pergunta=otimizado.tipo_pergunta,
            hyde_texto=otimizado.hyde_texto if i == 0 else None,
            query_original=otimizado.query_original,
        )
        for c in chunks_q:
            if c["chunk_id"] not in vistos:
                vistos.add(c["chunk_id"])
                todos_chunks.append(c)

    return sorted(todos_chunks, key=lambda x: x["score_final"], reverse=True)[:n_resultados]


# ---------------------------------------------------------------------------
# Endpoint principal
# ---------------------------------------------------------------------------

@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest):
    if "qclient" not in _estado:
        raise HTTPException(status_code=503, detail="Servico nao inicializado.")

    inicio = time.monotonic()

    # --- Analitico (instantaneo, sem cache necessario) ---
    if is_pergunta_analitica(req.pergunta):
        log.info("Pergunta analitica: '%s'", req.pergunta[:60])
        resultado = responder_analitico(req.pergunta)
        return QueryResponse(
            resposta=resultado["resposta"],
            fontes=[],
            modelo="analytics",
            tokens_prompt=0,
            tokens_resposta=0,
            latencia_total_ms=int((time.monotonic() - inicio) * 1000),
            latencia_llm_ms=0,
            system_prompt="Analise agregada de metadados",
            tipo_resposta="analitico",
        )

    # --- Autoria: detecta antes do optimizer para redirecionar com filtro certo ---
    _p_lower = req.pergunta.lower()
    if any(t in _p_lower for t in ["quem assinou", "quem publicou", "quem emitiu",
                                    "atos do relator", "atos do diretor", "assinados por"]):
        import re as _re
        # Tenta extrair nome do autor da pergunta (palavras capitalizadas após "por"/"do")
        m_autor = _re.search(r'(?:por|do|da|relator|diretor)\s+([A-ZÀ-Ú][a-zà-ú]+(?:\s+[A-ZÀ-Ú][a-zà-ú]+)*)', req.pergunta)
        if m_autor:
            filtro_autor = {"autor": m_autor.group(1)}
            log.info("Autoria detectada — filtro: %s", filtro_autor)
            req = req.model_copy(update={"filtros": {**(req.filtros or {}), **filtro_autor}})

    # --- Prepara historico ---
    historico_dicts = [
        {"pergunta": t.pergunta, "resposta": t.resposta}
        for t in (req.historico or [])
    ]
    n_esclarecimentos = (req.historico[-1].n_esclarecimentos if req.historico else 0)

    # --- Cache: checa antes de qualquer processamento ---
    cached = _cache_busca.get(req.pergunta, req.filtros)
    if cached:
        chunks_cacheados, otimizado_cacheado = cached
        # Ainda precisa gerar a resposta (LLM nao e cacheado)
        try:
            resultado = gerar_resposta(req.pergunta, chunks_cacheados)
        except Exception as e:
            log.error("Erro na geracao (cache): %s", e)
            raise HTTPException(status_code=500, detail=str(e))

        lat_total = int((time.monotonic() - inicio) * 1000)
        fontes = [
            FonteCitada(
                chunk_id=c["chunk_id"], titulo=c.get("titulo",""),
                tipo_nome=c.get("tipo_nome",""), numero=c.get("numero",""),
                ano=c.get("ano",""), assunto=c.get("assunto",""),
                url_pdf=c.get("url_pdf",""), score_final=c["score_final"],
                score_semantico=c["score_semantico"], score_bm25=c["score_bm25"],
                trecho=c.get("texto","")[:300],
            ) for c in chunks_cacheados
        ]
        return QueryResponse(
            resposta=resultado.texto, fontes=fontes,
            modelo=resultado.modelo, tokens_prompt=resultado.tokens_prompt,
            tokens_resposta=resultado.tokens_resposta, latencia_total_ms=lat_total,
            latencia_llm_ms=resultado.latencia_ms, system_prompt=resultado.system_prompt,
            tipo_resposta="normal", n_esclarecimentos=n_esclarecimentos, cache_hit=True,
        )

    # --- Paralelismo: optimizer e pre-embedding simultaneos ---
    executor = _estado["executor"]
    otimizado = None

    # Submete optimizer em thread
    future_optimizer = executor.submit(
        _executar_optimizer, req.pergunta, historico_dicts, n_esclarecimentos
    )

    # Aguarda optimizer com timeout
    try:
        otimizado = future_optimizer.result(timeout=OPTIMIZER_TIMEOUT)
        log.info("Optimizer | tipo=%s | filtros=%s | ambigua=%s | lat=%dms",
                 otimizado.tipo_pergunta, otimizado.filtros,
                 otimizado.ambigua, otimizado.latencia_ms)

        if otimizado.contexto_mudou:
            n_esclarecimentos = 0
            log.info("Contexto mudou — historico descartado para esta busca")

    except (FuturesTimeout, Exception) as e:
        log.warning("Optimizer timeout/erro (%s) — usando query original.", e)
        otimizado = QueryOtimizada(
            query_original=req.pergunta, queries=[req.pergunta],
            hyde_texto=req.pergunta, filtros=req.filtros or {},
            tipo_pergunta="busca", sub_queries=[], termos_chave=[],
            requer_multiplos_docs=False,
        )

    # Combina filtros
    filtros_finais = {**(otimizado.filtros or {}), **(req.filtros or {})}

    # --- Busca (com filtros otimizados) ---
    try:
        t_busca = time.monotonic()
        chunks = _executar_busca(otimizado, filtros_finais, req.n_resultados)
        log.info("Busca: %d chunks em %.0fms", len(chunks),
                 (time.monotonic() - t_busca) * 1000)
    except Exception as e:
        log.error("Erro na busca: %s", e)
        raise HTTPException(status_code=500, detail="Erro na recuperacao: %s" % e)

    chunks = [c for c in chunks if c.get("texto", "").strip()]
    log.info("Chunks validos: %d para '%s'", len(chunks), req.pergunta[:50])

    # Salva no cache (chunks + otimizado para possivel reuso)
    if chunks:
        _cache_busca.set(req.pergunta, req.filtros, (chunks, otimizado))

    # --- Geracao LLM ---
    try:
        t_llm = time.monotonic()
        resultado = gerar_resposta(req.pergunta, chunks, tipo_forcado=otimizado.tipo_pergunta)
        log.info("LLM: %.0fms", (time.monotonic() - t_llm) * 1000)
    except EnvironmentError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        log.error("Erro na geracao: %s", e)
        raise HTTPException(status_code=500, detail="Erro na geracao: %s" % e)

    lat_total = int((time.monotonic() - inicio) * 1000)
    log.info("TOTAL: %dms", lat_total)

    fontes = [
        FonteCitada(
            chunk_id=c["chunk_id"], titulo=c.get("titulo",""),
            tipo_nome=c.get("tipo_nome",""), numero=c.get("numero",""),
            ano=c.get("ano",""), assunto=c.get("assunto",""),
            url_pdf=c.get("url_pdf",""), score_final=c["score_final"],
            score_semantico=c["score_semantico"], score_bm25=c["score_bm25"],
            trecho=c.get("texto","")[:300],
        ) for c in chunks
    ]

    return QueryResponse(
        resposta=resultado.texto, fontes=fontes,
        modelo=resultado.modelo, tokens_prompt=resultado.tokens_prompt,
        tokens_resposta=resultado.tokens_resposta, latencia_total_ms=lat_total,
        latencia_llm_ms=resultado.latencia_ms, system_prompt=resultado.system_prompt,
        tipo_resposta="normal", n_esclarecimentos=n_esclarecimentos,
    )


@app.get("/health", response_model=HealthResponse)
def health():
    if "qclient" not in _estado:
        raise HTTPException(status_code=503, detail="Servico nao inicializado.")
    try:
        count = _estado["qclient"].count(NOME_COLECAO).count
    except Exception as e:
        raise HTTPException(status_code=503, detail="Qdrant inacessivel: %s" % e)
    return HealthResponse(
        status="ok",
        qdrant_chunks=count,
        bm25_docs=len(_estado["bm25_ids"]),
        modelo_llm=os.getenv("LLM_MODEL", "llama-3.3-70b-versatile"),
        cache_entries=len(_cache_busca._cache),
    )