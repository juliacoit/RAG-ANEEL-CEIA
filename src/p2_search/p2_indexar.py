"""
p2_indexar.py
=============
Pessoa 2 — Search Engineer

Lê os chunks gerados pela P1 (chunks_json_todos.parquet),
gera embeddings via sentence-transformers (local, gratuito)
e indexa no Qdrant local (Docker).

Pré-requisitos:
  - Docker rodando com Qdrant: docker compose up -d
  - pip install qdrant-client sentence-transformers pandas pyarrow python-dotenv rank-bm25 tqdm
  - Arquivo .env com ANTHROPIC_API_KEY (para geração — a P3 usa)

Uso — rodar da RAIZ do projeto:
  python src/p2_search/p2_indexar.py
  python src/p2_search/p2_indexar.py --limite 100   # teste rápido
  python src/p2_search/p2_indexar.py --resetar       # apaga e recria coleção
  python src/p2_search/p2_indexar.py --testar        # roda perguntas de teste
"""

import os
import pickle
import argparse
import time
import re
import logging
from pathlib import Path

import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, PointStruct,
    Filter, FieldCondition, MatchValue, MatchAny,
)
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

try:
    from tqdm import tqdm
    USA_TQDM = True
except ImportError:
    USA_TQDM = False

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

load_dotenv()

RAIZ = Path(__file__).resolve().parent.parent.parent

PARQUET         = RAIZ / "data" / "processed" / "chunks_completo_unificado.parquet"
PASTA_BM25      = RAIZ / "db" / "bm25"
ARQUIVO_BM25    = PASTA_BM25 / "bm25_index.pkl"
ARQUIVO_IDS     = PASTA_BM25 / "bm25_ids.pkl"

NOME_COLECAO    = "aneel_chunks"
QDRANT_URL      = "http://localhost:6333"

# Verifica se existe um arquivo de embeddings pré-calculado pelo Colab
PASTA_MANUAL    = RAIZ / "index_manual"
ARQUIVO_NPY     = PASTA_MANUAL / "embeddings_bge_m3.npy"

if ARQUIVO_NPY.exists():
    MODELO_EMBED = "BAAI/bge-m3"
    DIM_EMBED    = 1024
else:
    MODELO_EMBED = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    DIM_EMBED    = 384

# Pool de clientes Qdrant para paralelismo real nos scrolls
_qdrant_pool = []
POOL_SIZE = 4

def _get_qdrant_pool():
    """Retorna pool de clientes Qdrant (criado na primeira chamada)."""
    global _qdrant_pool
    if not _qdrant_pool:
        for _ in range(POOL_SIZE):
            _qdrant_pool.append(QdrantClient(url=QDRANT_URL, timeout=10))
        log.info("Pool Qdrant criado: %d conexoes", POOL_SIZE)
    return _qdrant_pool

LOTE_EMBED      = 64
LOTE_QDRANT     = 200

PESO_SEMANTICO  = 0.30
PESO_BM25       = 0.70

# Tamanho mínimo de chunk para retornar ao LLM
MIN_CHARS_CHUNK = 150

_modelo_cache = None
_embedding_cache = {}   # cache query -> vetor (max 200 entradas)
MAX_EMBEDDING_CACHE = 200


# ---------------------------------------------------------------------------
# Embeddings
# ---------------------------------------------------------------------------

def carregar_modelo() -> SentenceTransformer:
    global _modelo_cache
    if _modelo_cache is None:
        log.info(f"Carregando modelo de embedding: {MODELO_EMBED}")
        log.info("(Primeira execução baixa ~120MB — aguarde)")
        _modelo_cache = SentenceTransformer(MODELO_EMBED)
        log.info("Modelo carregado.")
    return _modelo_cache


def gerar_embeddings(textos: list[str]) -> list[list[float]]:
    modelo = carregar_modelo()
    todos = []

    iterador = range(0, len(textos), LOTE_EMBED)
    if USA_TQDM:
        iterador = tqdm(list(iterador), desc="  Embeddings", unit="lote")

    for i in iterador:
        lote = textos[i: i + LOTE_EMBED]
        vetores = modelo.encode(
            lote,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        todos.extend(vetores.tolist())

    return todos


# ---------------------------------------------------------------------------
# Qdrant
# ---------------------------------------------------------------------------

def conectar_qdrant() -> QdrantClient:
    try:
        client = QdrantClient(url=QDRANT_URL, timeout=30)
        client.get_collections()
        log.info(f"Qdrant conectado em {QDRANT_URL}")
        return client
    except Exception as e:
        log.error(f"Não foi possível conectar ao Qdrant: {e}")
        log.error("Verifique se o Docker está rodando: docker compose up -d")
        raise


def criar_colecao(qclient: QdrantClient, resetar: bool = False) -> None:
    colecoes = [c.name for c in qclient.get_collections().collections]

    if NOME_COLECAO in colecoes:
        if resetar:
            log.info(f"Deletando coleção existente: {NOME_COLECAO}")
            qclient.delete_collection(NOME_COLECAO)
        else:
            count = qclient.count(NOME_COLECAO).count
            log.info(f"Coleção '{NOME_COLECAO}' já existe com {count} pontos.")
            return

    log.info(f"Criando coleção '{NOME_COLECAO}' (dim={DIM_EMBED}, cosine)...")
    qclient.create_collection(
        collection_name=NOME_COLECAO,
        vectors_config=VectorParams(size=DIM_EMBED, distance=Distance.COSINE),
    )
    log.info("Coleção criada.")


def indexar_qdrant(
    df: pd.DataFrame,
    embeddings: list[list[float]],
    qclient: QdrantClient,
) -> None:
    log.info(f"Indexando {len(df)} chunks no Qdrant...")

    total = 0
    for i in range(0, len(df), LOTE_QDRANT):
        lote_df  = df.iloc[i: i + LOTE_QDRANT]
        lote_emb = embeddings[i: i + LOTE_QDRANT]

        def _s(v):
            if v is None:
                return ""
            try:
                if pd.isna(v):
                    return ""
            except Exception:
                pass
            s = str(v)
            return "" if s == "nan" else s

        pontos = []
        for j, (_, row) in enumerate(lote_df.iterrows()):
            payload = {
                "chunk_id":        _s(row.get("chunk_id")),
                "doc_id":          _s(row.get("doc_id")),
                "tipo_codigo":     _s(row.get("tipo_codigo")),
                "tipo_nome":       _s(row.get("tipo_nome")),
                "numero":          _s(row.get("numero")),
                "ano":             _s(row.get("ano")),
                "ano_fonte":       _s(row.get("ano_fonte")),
                "autor":           _s(row.get("autor")),
                "assunto":         _s(row.get("assunto")),
                "data_publicacao": _s(row.get("data_publicacao")),
                "data_assinatura": _s(row.get("data_assinatura")),
                "url_pdf":         _s(row.get("url_pdf")),
                "fonte_texto":     _s(row.get("fonte_texto")),
                "chunk_index":     int(row.get("chunk_index") or 0),
                "chunk_total":     int(row.get("chunk_total") or 1),
                "texto":           _s(row.get("texto")),
            }

            ponto_id = abs(hash(payload["chunk_id"])) % (2 ** 53)
            pontos.append(PointStruct(
                id=ponto_id,
                vector=lote_emb[j],
                payload=payload,
            ))

        qclient.upsert(collection_name=NOME_COLECAO, points=pontos)
        total += len(pontos)
        pct = min(100, int(total / len(df) * 100))
        log.info(f"  Qdrant: {pct}% ({total}/{len(df)})")

    log.info(f"Qdrant: {total} pontos indexados.")


# ---------------------------------------------------------------------------
# BM25
# ---------------------------------------------------------------------------

def tokenizar(texto: str) -> list[str]:
    return re.findall(r"\b\w+\b", texto.lower())


def criar_bm25(df: pd.DataFrame) -> tuple:
    PASTA_BM25.mkdir(parents=True, exist_ok=True)

    if ARQUIVO_BM25.exists() and ARQUIVO_IDS.exists():
        log.info("Índice BM25 já existe. Carregando do disco...")
        with open(ARQUIVO_BM25, "rb") as f:
            bm25 = pickle.load(f)
        with open(ARQUIVO_IDS, "rb") as f:
            ids = pickle.load(f)
        log.info(f"BM25 carregado: {len(ids)} documentos.")
        return bm25, ids

    log.info("Criando índice BM25...")
    textos = df["texto"].tolist()
    ids    = df["chunk_id"].tolist()

    corpus = [tokenizar(t) for t in textos]
    bm25   = BM25Okapi(corpus)

    with open(ARQUIVO_BM25, "wb") as f:
        pickle.dump(bm25, f)
    with open(ARQUIVO_IDS, "wb") as f:
        pickle.dump(ids, f)

    log.info(f"BM25 salvo: {len(ids)} documentos.")
    return bm25, ids


# ---------------------------------------------------------------------------
# BUSCA HÍBRIDA
# ---------------------------------------------------------------------------

def _detectar_filtros_automaticos(query: str, filtros: dict, query_original: str = None) -> dict:
    """
    Detecta número de ato e ano na query e adiciona como filtro automático.
    Exemplos:
      "Despacho 1380 de 2021" → {"numero": "1380", "ano": "2021"}
      "REN 687/2015"          → {"numero": "687",  "ano": "2015"}
    Só aplica se o usuário não passou filtros manuais para esses campos.
    """
    # Nao aplica filtros automaticos se a query mencionar multiplos anos
    # pois provavelmente e uma pergunta de comparacao temporal
   # Usa query original para detectar comparacao temporal
    query_para_filtro = query_original if query_original else query
    anos_na_query = re.findall(r'\b(19\d{2}|20\d{2})\b', query_para_filtro)
    if len(set(anos_na_query)) >= 2:
        return filtros if filtros else None
    filtros = dict(filtros) if filtros else {}

    # Padrão NNNN/AAAA ou "NNNN de AAAA"
    m = re.search(r"\b(\d{3,5})[/\s](?:de\s)?(\d{4})\b", query)
    if m:
        numero, ano = m.group(1), m.group(2)
        if "numero" not in filtros:
            filtros["numero"] = numero
        if "ano" not in filtros:
            filtros["ano"] = ano
        log.info(f"Filtro automático: numero={numero}, ano={ano}")
    else:
        # Só número (4+ dígitos que não sejam um ano)
        m = re.search(r"\b(\d{4,5})\b", query)
        if m and "numero" not in filtros:
            numero = m.group(1)
            if not (1990 <= int(numero) <= 2030):
                filtros["numero"] = numero
                log.info(f"Filtro automático: numero={numero}")

    return filtros if filtros else None


def _scroll_lote(cliente, lote_ids):
    """Executa um scroll para um lote de chunk_ids usando cliente dedicado."""
    condicoes = [FieldCondition(key="chunk_id", match=MatchAny(any=lote_ids))]
    try:
        resultado = cliente.scroll(
            collection_name=NOME_COLECAO,
            scroll_filter=Filter(must=condicoes),
            limit=len(lote_ids),
            with_payload=True,
            with_vectors=False,
        )
        return {p.payload["chunk_id"]: p.payload for p in resultado[0] if p.payload.get("chunk_id")}
    except Exception as e:
        log.warning("Erro no scroll: %s", e)
        return {}


def _recuperar_payloads_bm25(qclient, chunk_ids):
    """
    Recupera payloads do Qdrant para chunk_ids do BM25.
    Usa ThreadPoolExecutor para paralelizar os scrolls.
    """
    if not chunk_ids:
        return {}

    LOTE = 30
    lotes = [chunk_ids[i:i+LOTE] for i in range(0, len(chunk_ids), LOTE)]
    payloads = {}

    pool = _get_qdrant_pool()
    with ThreadPoolExecutor(max_workers=min(POOL_SIZE, len(lotes))) as ex:
        futuros = {
            ex.submit(_scroll_lote, pool[i % POOL_SIZE], lote): lote
            for i, lote in enumerate(lotes)
        }
        for futuro in as_completed(futuros, timeout=10):
            try:
                payloads.update(futuro.result())
            except Exception as e:
                log.warning("Scroll paralelo falhou: %s", e)

    return payloads




def _expandir_contexto(qclient, chunks_encontrados, n_vizinhos=2):
    """
    Para cada chunk encontrado, busca chunks vizinhos em paralelo.
    Resolve o caso onde o titulo esta num chunk mas o conteudo nos seguintes.
    """
    # Monta todos os lotes de vizinhos de uma vez
    lotes = []
    for chunk in chunks_encontrados:
        chunk_id = chunk.get("chunk_id", "")
        if not chunk_id:
            continue
        try:
            partes = chunk_id.rsplit("_", 1)
            idx_atual = int(partes[-1])
        except (ValueError, IndexError):
            continue

        prefixo = partes[0]
        vizinhos = [
            "%s_%d" % (prefixo, i)
            for i in range(max(0, idx_atual - n_vizinhos), idx_atual + n_vizinhos + 1)
            if i != idx_atual
        ]
        if vizinhos:
            lotes.append(vizinhos)

    if not lotes:
        return {}

    payloads_extras = {}
    pool = _get_qdrant_pool()
    with ThreadPoolExecutor(max_workers=min(POOL_SIZE, len(lotes))) as ex:
        futuros = {
            ex.submit(_scroll_lote, pool[i % POOL_SIZE], lote): lote
            for i, lote in enumerate(lotes)
        }
        for futuro in as_completed(futuros, timeout=10):
            try:
                for cid, payload in futuro.result().items():
                    if len(payload.get("texto", "")) >= MIN_CHARS_CHUNK:
                        payloads_extras[cid] = payload
            except Exception as e:
                log.warning("Expansao paralela falhou: %s", e)

    return payloads_extras

def buscar(
    query: str,
    qclient: QdrantClient,
    bm25: BM25Okapi,
    bm25_ids: list[str],
    n_resultados: int = 5,
    filtros: dict = None,
    peso_semantico: float = PESO_SEMANTICO,
    peso_bm25: float = PESO_BM25,
    tipo_pergunta: str = "busca",
    hyde_texto: str = None,
    _e_fallback: bool = False,
    query_original: str = None,
) -> list[dict]:
    """
    Busca híbrida: semântica (Qdrant) + palavras exatas (BM25).

    Fluxo:
      1. Qdrant retorna top candidatos semânticos
      2. BM25 retorna top candidatos lexicais
      3. Chunks do BM25 sem payload são buscados diretamente no Qdrant
      4. Scores são combinados e filtrados por tamanho mínimo
    """
    n_candidatos = n_resultados * 10

    # Aplica filtros automaticos por numero/ano detectados na query
    # Nao aplica no fallback de mencoes para evitar recursao infinita
    if not _e_fallback:
        filtros = _detectar_filtros_automaticos(query, filtros, query_original)

    # --- 1. Embedding da query ---
    # HyDE: usa o trecho hipotético gerado pelo LLM para busca semântica
    # mais precisa — o vetor do HyDE está mais próximo dos documentos reais
    modelo = carregar_modelo()
    texto_embedding = hyde_texto if hyde_texto else query

    # Cache de embedding — evita rodar o modelo para queries repetidas
    if texto_embedding in _embedding_cache:
        vetor_query = _embedding_cache[texto_embedding]
        log.info("Embedding cache hit")
    else:
        vetor_query = modelo.encode(
            [texto_embedding],
            normalize_embeddings=True,
            show_progress_bar=False,
        )[0].tolist()
        _embedding_cache[texto_embedding] = vetor_query
        if len(_embedding_cache) > MAX_EMBEDDING_CACHE:
            # Remove entrada mais antiga
            _embedding_cache.pop(next(iter(_embedding_cache)))

    # --- 2. Busca semântica no Qdrant ---
    qdrant_filter = None
    if filtros:
        condicoes = []
        for campo, valor in filtros.items():
            if isinstance(valor, list):
                condicoes.append(FieldCondition(
                    key=campo, match=MatchAny(any=valor)
                ))
            else:
                condicoes.append(FieldCondition(
                    key=campo, match=MatchValue(value=valor)
                ))
        qdrant_filter = Filter(must=condicoes)

    try:
        resultado = qclient.query_points(
            collection_name=NOME_COLECAO,
            query=vetor_query,
            limit=n_candidatos,
            query_filter=qdrant_filter,
            with_payload=True,
        )
        hits = resultado.points
    except AttributeError:
        hits = qclient.search(
            collection_name=NOME_COLECAO,
            query_vector=vetor_query,
            limit=n_candidatos,
            query_filter=qdrant_filter,
            with_payload=True,
        )

    hits = [h for h in hits if len(h.payload.get("texto", "")) >= MIN_CHARS_CHUNK]
    scores_semanticos = {h.payload["chunk_id"]: h.score for h in hits}
    payloads          = {h.payload["chunk_id"]: h.payload for h in hits}

    # --- 3. Busca BM25 ---
    # ── Pesos dinâmicos por tipo de pergunta ──────────────────────────────
    # Corpus jurídico ANEEL: vocabulário controlado favorece BM25 na maioria
    # dos casos. Semântico só lidera quando o usuário usa vocabulário diferente
    # do corpus (perguntas vagas, definições, comparações temporais).
    #
    # Tipos suportados:
    #   "especifica"             — busca por número de ato exato
    #   "comparacao_referencia"  — referência cruzada entre documentos
    #   "tabela"                 — valores, tarifas, planilhas
    #   "vigencia"               — se ato está em vigor / quando foi revogado
    #   "busca"                  — fallback geral (padrão)
    #   "resumo"                 — resumir um ato específico
    #   "procedimento"           — fluxo/processo descrito em artigos
    #   "comparacao_temporal"    — o que mudou entre anos
    #   "definicao"              — o que significa um termo técnico
    #   "semantica"              — query vaga sem termos técnicos exatos
    #   "agregacao"              — sinaliza para a P3 usar query no Parquet
    #   "autoria"                — sinaliza para a P3 filtrar por campo autor
    #   "hibrida_complexa"       — sinaliza para a P3 fazer busca multi-step

    _PESOS = {
        "especifica":            (0.10, 0.90),
        "comparacao_referencia": (0.15, 0.85),
        "tabela":                (0.15, 0.85),
        "vigencia":              (0.20, 0.80),
        "busca":                 (0.30, 0.70),
        "resumo":                (0.30, 0.70),
        "procedimento":          (0.40, 0.60),
        "comparacao_temporal":   (0.50, 0.50),
        "definicao":             (0.55, 0.45),
        "semantica":             (0.55, 0.45),
        # Tipos complexos — P3 deve tratar antes de chamar buscar()
        # mas se chegarem aqui, usam pesos conservadores
        "agregacao":             (0.30, 0.70),
        "autoria":               (0.10, 0.90),
        "hibrida_complexa":      (0.40, 0.60),
    }
    peso_semantico, peso_bm25 = _PESOS.get(tipo_pergunta, (peso_semantico, peso_bm25))

    # Tipos complexos: loga aviso para a P3 tratar adequadamente
    if tipo_pergunta == "agregacao":
        log.warning(
            "tipo_pergunta='agregacao': RAG não é ideal para contagens/listagens. "
            "Considere query direta no Parquet na P3."
        )
    elif tipo_pergunta == "autoria":
        log.warning(
            "tipo_pergunta='autoria': considere passar filtros={'autor': nome} "
            "diretamente para buscar() em vez de depender dos pesos."
        )
    elif tipo_pergunta == "hibrida_complexa":
        log.warning(
            "tipo_pergunta='hibrida_complexa': recomenda-se busca multi-step na P3 "
            "(múltiplas chamadas a buscar() com queries decompostas)."
        )

    # Expande a query BM25 com termos específicos por tipo de pergunta
    query_bm25 = query
    if tipo_pergunta == "comparacao":
        query_bm25 += " revogado revoga fica revogado tornada sem efeito altera substituído cancelado"
    elif tipo_pergunta == "tabela":
        query_bm25 += " valor tarifa percentual tabela planilha"
    elif tipo_pergunta == "resumo":
        query_bm25 += " introdução prefácio objetivo contexto"

    tokens = tokenizar(query_bm25)
    scores_raw = bm25.get_scores(tokens)
    max_bm25 = max(scores_raw) if max(scores_raw) > 0 else 1.0

    # Top chunks do BM25 por score
    bm25_top = sorted(
        [(bm25_ids[i], score / max_bm25) for i, score in enumerate(scores_raw) if score > 0],
        key=lambda x: x[1],
        reverse=True,
    )[:n_candidatos]

    scores_bm25 = {cid: score for cid, score in bm25_top}

    # --- 4. Recupera payloads do BM25 que não vieram do Qdrant ---
    ids_sem_payload = [cid for cid in scores_bm25 if cid not in payloads]
    if ids_sem_payload:
        limite_bm25 = 15 if _e_fallback else 30
        payloads_bm25 = _recuperar_payloads_bm25(qclient, ids_sem_payload[:limite_bm25])
        # Filtra por tamanho mínimo
        for cid, payload in payloads_bm25.items():
            if len(payload.get("texto", "")) >= MIN_CHARS_CHUNK:
                payloads[cid] = payload

    # --- 5. Expansão de contexto por chunks vizinhos ---
    # Ativada apenas para resumos e perguntas especificas (vale a latencia extra)
    # Para busca geral desativada para manter latencia baixa
    if tipo_pergunta in ("resumo", "especifica", "tabela",
                         "procedimento", "comparacao_temporal",
                         "comparacao_referencia", "vigencia") and not _e_fallback:
        chunks_base = [
            {"chunk_id": cid, "doc_id": payloads[cid].get("doc_id", ""), "texto": payloads[cid].get("texto", "")}
            for cid in list(payloads.keys())[:5]
        ]
        payloads_vizinhos = _expandir_contexto(qclient, chunks_base, n_vizinhos=2)
        for cid, payload in payloads_vizinhos.items():
            if cid not in payloads:
                payloads[cid] = payload
                scores_semanticos[cid] = 0.3
    else:
        log.info("Expansao de vizinhos desativada para tipo=%s fallback=%s", tipo_pergunta, _e_fallback)

    # --- 6. Combinação híbrida ---
    todos_ids = set(scores_semanticos) | set(scores_bm25)
    todos_ids = {id_ for id_ in todos_ids if id_ in payloads}

    scores_finais = {
        id_: (peso_semantico * scores_semanticos.get(id_, 0.0))
           + (peso_bm25      * scores_bm25.get(id_, 0.0))
        for id_ in todos_ids
    }

    # Boost de score para chunks XLSX em perguntas de tabela
    if tipo_pergunta == "tabela" and not _e_fallback:
        for id_ in todos_ids:
            if payloads.get(id_, {}).get("fonte_texto") == "xlsx":
                scores_finais[id_] = min(1.0, scores_finais[id_] + 0.3)

    top_ids = sorted(scores_finais, key=scores_finais.get, reverse=True)[:n_resultados]

    # --- 7. Monta resultado ---
    resultados = []
    for id_ in top_ids:
        payload = payloads.get(id_, {})
        texto = payload.get("texto", "")

        if len(texto) < MIN_CHARS_CHUNK:
            continue

        resultados.append({
            "chunk_id":        id_,
            "doc_id":          payload.get("doc_id", ""),
            "score_final":     round(scores_finais[id_], 4),
            "score_semantico": round(scores_semanticos.get(id_, 0.0), 4),
            "score_bm25":      round(scores_bm25.get(id_, 0.0), 4),
            "titulo":          f"{payload.get('tipo_nome','')} nº {payload.get('numero','')} {payload.get('ano','')}",
            "tipo_codigo":     payload.get("tipo_codigo", ""),
            "tipo_nome":       payload.get("tipo_nome", ""),
            "numero":          payload.get("numero", ""),
            "ano":             payload.get("ano", ""),
            "autor":           payload.get("autor", ""),
            "assunto":         payload.get("assunto", ""),
            "data_publicacao": payload.get("data_publicacao", ""),
            "fonte_texto":     payload.get("fonte_texto", ""),
            "url_pdf":         payload.get("url_pdf", ""),
            "texto":           texto,
            "chunk_index":     payload.get("chunk_index", 0),
        })

    # --- 8. Fallback sem filtro quando score baixo ---
    # Se filtros automáticos foram aplicados mas os resultados têm score baixo
    # (documento não está na base), faz nova busca sem filtro para encontrar
    # documentos que MENCIONAM o ato solicitado
    SCORE_MIN = 0.5
    tem_filtros = bool(filtros)
    score_max = max((r["score_final"] for r in resultados), default=0)

    if tem_filtros and score_max < SCORE_MIN and resultados:
        log.info(
            f"Score máximo {score_max:.2f} < {SCORE_MIN} com filtros ativos — "
            f"buscando menções sem filtro"
        )
        resultados_mencoes = buscar(
            query=query,
            qclient=qclient,
            bm25=bm25,
            bm25_ids=bm25_ids,
            n_resultados=n_resultados,
            filtros=None,
            peso_semantico=peso_semantico,
            peso_bm25=peso_bm25,
            tipo_pergunta=tipo_pergunta,
            hyde_texto=hyde_texto,
            _e_fallback=True,
        )
        score_max_mencoes = max((r["score_final"] for r in resultados_mencoes), default=0)
        if score_max_mencoes > SCORE_MIN:
            log.info(f"Menções encontradas com score {score_max_mencoes:.2f} — usando resultado sem filtro")
            # Marca os chunks como sendo menções, não o documento principal
            for r in resultados_mencoes:
                r["busca_fallback"] = True
            return resultados_mencoes
    resultados = _injetar_cabecalhos_xlsx(qclient, resultados)
    return resultados




def _injetar_cabecalhos_xlsx(qclient, resultados):
    precisam = [
        r for r in resultados
        if r.get("fonte_texto") == "xlsx" and r.get("chunk_index", 0) > 0
    ]
    if not precisam:
        return resultados

    pool = _get_qdrant_pool()
    cabecalhos = {}

    for r in precisam:
        cid = r["chunk_id"]
        partes = cid.rsplit("_", 1)
        if len(partes) != 2:
            continue
        cid_cab = partes[0] + "_0"
        if cid_cab in cabecalhos:
            continue
        resultado = _scroll_lote(pool[0], [cid_cab])
        if resultado:
            payload = list(resultado.values())[0]
            texto_cab = payload.get("texto", "")
            linhas = texto_cab.split("\n")
            cab_linhas = []
            for linha in linhas:
                cab_linhas.append(linha)
                if not linha.strip():
                    continue
                partes = [p.strip() for p in linha.split("|")]
                # Se tem mais de 2 colunas e a maioria das colunas
                # (exceto a primeira) são números ou traço — é linha de dado
                colunas_valor = [p for p in partes[1:] if p]
                if len(colunas_valor) >= 2:
                    n_numericos = sum(
                        1 for p in colunas_valor
                        if p.replace(".", "").replace(",", "").replace("-", "").strip().isdigit()
                        or p == "-"
                    )
                    if n_numericos >= len(colunas_valor) * 0.6:
                        break
            cabecalhos[cid_cab] = "\n".join(cab_linhas)

    novos = []
    for r in resultados:
        if (r.get("fonte_texto") == "xlsx"
                and r.get("chunk_index", 0) > 0):
            cid = r["chunk_id"]
            partes = cid.rsplit("_", 1)
            cid_cab = partes[0] + "_0"
            if cid_cab in cabecalhos:
                r = dict(r)
                r["texto"] = "[Cabecalho]\n" + cabecalhos[cid_cab] + "\n\n[Dados]\n" + r["texto"]
        novos.append(r)

    # Garante que o chunk 0 de cada doc XLSX esta nos resultados
    # pois ele tem o cabecalho + primeiros dados completos
    doc_ids_presentes = set(r["doc_id"] for r in novos if r.get("fonte_texto") == "xlsx")
    chunk0_ids = []
    for r in novos:
        if r.get("fonte_texto") == "xlsx" and r.get("chunk_index", 0) > 0:
            cid = r["chunk_id"]
            partes = cid.rsplit("_", 1)
            if len(partes) == 2:
                chunk0_ids.append(partes[0] + "_0")

    if chunk0_ids:
        chunk0_ids = list(set(chunk0_ids))
        payloads_0 = _scroll_lote(pool[0], chunk0_ids)
        vistos = {r["chunk_id"] for r in novos}
        for cid, payload in payloads_0.items():
            if cid not in vistos and len(payload.get("texto","")) >= 150:
                novos.append({
                    "chunk_id":        cid,
                    "doc_id":          payload.get("doc_id",""),
                    "score_final":     0.75,
                    "score_semantico": 0.75,
                    "score_bm25":      0.0,
                    "titulo":          "%s n %s/%s" % (payload.get("tipo_nome",""), payload.get("numero",""), payload.get("ano","")),
                    "tipo_codigo":     payload.get("tipo_codigo",""),
                    "tipo_nome":       payload.get("tipo_nome",""),
                    "numero":          payload.get("numero",""),
                    "ano":             payload.get("ano",""),
                    "autor":           payload.get("autor",""),
                    "assunto":         payload.get("assunto",""),
                    "data_publicacao": payload.get("data_publicacao",""),
                    "fonte_texto":     payload.get("fonte_texto",""),
                    "url_pdf":         payload.get("url_pdf",""),
                    "texto":           payload.get("texto",""),
                    "chunk_index":     0,
                })
        log.info("Chunk 0 XLSX adicionado: %d chunks", len(payloads_0))
    log.info("Cabecalhos XLSX injetados em %d chunks", len(precisam))
    return novos

def carregar_indices(qclient: QdrantClient) -> tuple:
    """
    Carrega os índices BM25 do disco.
    """
    if not ARQUIVO_BM25.exists():
        raise FileNotFoundError(
            f"Índice BM25 não encontrado em {ARQUIVO_BM25}. "
            "Rode p2_indexar.py primeiro."
        )
    with open(ARQUIVO_BM25, "rb") as f:
        bm25 = pickle.load(f)
    with open(ARQUIVO_IDS, "rb") as f:
        bm25_ids = pickle.load(f)
    return bm25, bm25_ids


# ---------------------------------------------------------------------------
# Teste de busca
# ---------------------------------------------------------------------------

def testar(qclient: QdrantClient, bm25: BM25Okapi, bm25_ids: list) -> None:
    perguntas = [
        # busca geral               — pesos 0.30/0.70
        ("qual a tarifa de uso do sistema de distribuição TUSD",         "busca"),
        # específica por número     — pesos 0.10/0.90
        ("Despacho 3409 percentual de redução TUSD",                     "especifica"),
        # tabela/planilha           — pesos 0.15/0.85
        ("valor TUSD distribuidora CEMIG 2021",                          "tabela"),
        # vigência                  — pesos 0.20/0.80
        ("a Resolução Normativa 482 ainda está em vigor",                "vigencia"),
        # comparação referência     — pesos 0.15/0.85
        ("quais documentos de 2016 citam a Lei 8.987/1995",             "comparacao_referencia"),
        # comparação temporal       — pesos 0.50/0.50
        ("o que mudou na regulação de TUSD entre 2016 e 2021",          "comparacao_temporal"),
        # procedimento              — pesos 0.40/0.60
        ("como solicitar autorização para pequena central hidrelétrica", "procedimento"),
        # definição                 — pesos 0.55/0.45
        ("o que é microgeração distribuída",                             "definicao"),
        # semântica pura            — pesos 0.55/0.45
        ("qual a regra sobre geração de energia em residências",         "semantica"),
        # multa auto de infração    — busca geral
        ("multa auto de infração distribuidora penalidade",              "busca"),
    ]

    log.info("\n" + "=" * 60)
    log.info("TESTE DE BUSCA HÍBRIDA")
    log.info("=" * 60)

    for pergunta, tipo in perguntas:
        log.info(f"\nQuery: '{pergunta}' [tipo={tipo}]")
        log.info("-" * 50)
        resultados = buscar(pergunta, qclient, bm25, bm25_ids, n_resultados=3, tipo_pergunta=tipo)
        for i, r in enumerate(resultados, 1):
            log.info(
                f"  {i}. [{r['score_final']:.3f}] "
                f"{r['tipo_nome']} nº {r['numero']}/{r['ano']} "
                f"| fonte: {r.get('fonte_texto','?')} | {len(r['texto'])} chars"
            )
            log.info(f"     {r['texto'][:120]}...")

    log.info("=" * 60)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Indexa chunks da ANEEL no Qdrant + BM25 (embeddings locais)"
    )
    parser.add_argument(
        "--parquet", default=str(PARQUET),
        help="Caminho do parquet gerado pela P1",
    )
    parser.add_argument(
        "--limite", type=int, default=None,
        help="Indexar apenas N chunks (modo teste)",
    )
    parser.add_argument(
        "--resetar", action="store_true",
        help="Apaga e recria a coleção do Qdrant",
    )
    parser.add_argument(
        "--testar", action="store_true",
        help="Após indexar, roda perguntas de teste",
    )
    args = parser.parse_args()

    qclient = conectar_qdrant()

    parquet_path = Path(args.parquet)
    if not parquet_path.exists():
        log.error(f"Parquet não encontrado: {parquet_path}")
        return

    log.info(f"Lendo {parquet_path.name}...")
    df = pd.read_parquet(parquet_path)
    log.info(f"  {len(df)} chunks carregados")

    if args.limite:
        df = df.head(args.limite)
        log.info(f"  Modo teste: usando apenas {len(df)} chunks")

    if "chunk_id" not in df.columns:
        df["chunk_id"] = df.index.astype(str)

    criar_colecao(qclient, resetar=args.resetar)

    count = qclient.count(NOME_COLECAO).count
    if count >= len(df) and not args.resetar:
        log.info(f"Qdrant já tem {count} pontos. Pulando embeddings.")
    else:
        if ARQUIVO_NPY.exists():
            log.info(f"Matriz do Colab detectada: {ARQUIVO_NPY.name}")
            log.info(f"Pulando modelo local e carregando arquivo pré-processado...")
            embeddings_array = np.load(ARQUIVO_NPY)
            embeddings = embeddings_array.tolist()
            if len(embeddings) != len(df):
                log.error(f"Erro: Parquet tem {len(df)} linhas, mas NPY tem {len(embeddings)}!")
                return
        else:
            log.info(f"Gerando embeddings para {len(df)} chunks (local, sem custo)...")
            textos     = df["texto"].tolist()
            embeddings = gerar_embeddings(textos)
            
        indexar_qdrant(df, embeddings, qclient)

    bm25, bm25_ids = criar_bm25(df)

    log.info("\n" + "=" * 55)
    log.info("INDEXAÇÃO CONCLUÍDA")
    log.info("=" * 55)
    log.info(f"  Qdrant  : {qclient.count(NOME_COLECAO).count} pontos")
    log.info(f"  BM25    : {len(bm25_ids)} documentos")
    log.info(f"  Modelo  : {MODELO_EMBED}")
    log.info(f"  Coleção : {NOME_COLECAO}")
    log.info("=" * 55)
    log.info("\nA P3 pode usar a busca assim:")
    log.info("  from src.p2_search.p2_indexar import conectar_qdrant, carregar_indices, buscar")
    log.info("  qclient = conectar_qdrant()")
    log.info("  bm25, bm25_ids = carregar_indices(qclient)")
    log.info("  resultados = buscar('sua pergunta', qclient, bm25, bm25_ids)")

    if args.testar:
        testar(qclient, bm25, bm25_ids)


if __name__ == "__main__":
    main()