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

PARQUET         = RAIZ / "data" / "processed" / "chunks_json_todos.parquet"
PASTA_BM25      = RAIZ / "db" / "bm25"
ARQUIVO_BM25    = PASTA_BM25 / "bm25_index.pkl"
ARQUIVO_IDS     = PASTA_BM25 / "bm25_ids.pkl"

NOME_COLECAO    = "aneel_chunks"
QDRANT_URL      = "http://localhost:6333"

# Modelo local gratuito — multilingual, bom para português
# Baixa ~120MB na primeira execução e fica em cache
MODELO_EMBED    = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
DIM_EMBED       = 384   # dimensão deste modelo

LOTE_EMBED      = 64    # chunks por lote de embedding (menor que OpenAI pois é local)
LOTE_QDRANT     = 200   # pontos por upsert no Qdrant

PESO_SEMANTICO  = 0.6
PESO_BM25       = 0.4

# Modelo carregado uma vez e reutilizado
_modelo_cache = None


# ---------------------------------------------------------------------------
# Embeddings — sentence-transformers (local, gratuito)
# ---------------------------------------------------------------------------

def carregar_modelo() -> SentenceTransformer:
    """Carrega o modelo de embedding uma única vez e guarda em cache."""
    global _modelo_cache
    if _modelo_cache is None:
        log.info(f"Carregando modelo de embedding: {MODELO_EMBED}")
        log.info("(Primeira execução baixa ~120MB — aguarde)")
        _modelo_cache = SentenceTransformer(MODELO_EMBED)
        log.info("Modelo carregado.")
    return _modelo_cache


def gerar_embeddings(textos: list[str]) -> list[list[float]]:
    """
    Gera embeddings localmente com sentence-transformers.
    Sem custo de API, sem limite de quota.
    """
    modelo = carregar_modelo()
    todos = []

    iterador = range(0, len(textos), LOTE_EMBED)
    if USA_TQDM:
        iterador = tqdm(list(iterador), desc="  Embeddings", unit="lote")

    for i in iterador:
        lote = textos[i: i + LOTE_EMBED]
        vetores = modelo.encode(
            lote,
            normalize_embeddings=True,   # normaliza para distância cosseno
            show_progress_bar=False,
        )
        todos.extend(vetores.tolist())

    return todos


# ---------------------------------------------------------------------------
# Qdrant
# ---------------------------------------------------------------------------

def conectar_qdrant() -> QdrantClient:
    """Conecta ao Qdrant local via Docker."""
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
    """Cria a coleção no Qdrant (ou recria se --resetar)."""
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
    """Faz upsert dos pontos no Qdrant em lotes."""
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
    """Tokeniza o texto para o BM25."""
    return re.findall(r"\b\w+\b", texto.lower())


def criar_bm25(df: pd.DataFrame) -> tuple:
    """Cria e salva o índice BM25. Carrega do disco se já existir."""
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
# BUSCA HÍBRIDA — função principal para a P3 usar
# ---------------------------------------------------------------------------

def buscar(
    query: str,
    qclient: QdrantClient,
    bm25: BM25Okapi,
    bm25_ids: list[str],
    n_resultados: int = 5,
    filtros: dict = None,
    peso_semantico: float = PESO_SEMANTICO,
    peso_bm25: float = PESO_BM25,
) -> list[dict]:
    """
    Busca híbrida: semântica (Qdrant) + palavras exatas (BM25).

    Parâmetros:
      query          — pergunta do usuário em linguagem natural
      qclient        — cliente Qdrant (de conectar_qdrant())
      bm25           — índice BM25 (de carregar_indices())
      bm25_ids       — lista de chunk_ids na ordem do BM25
      n_resultados   — quantos chunks retornar (padrão 5)
      filtros        — filtrar por metadado, ex:
                         {"tipo_codigo": "REH"}
                         {"ano_fonte": ["2021", "2022"]}
      peso_semantico — peso da busca semântica (padrão 0.6)
      peso_bm25      — peso do BM25 (padrão 0.4)

    Retorna lista de dicts com chunks mais relevantes + scores.

    Para migrar para outro modelo de embedding:
      Só troca carregar_modelo() — a interface desta função não muda.
    """
    n_candidatos = n_resultados * 4

    # --- 1. Embedding da query ---
    modelo = carregar_modelo()
    vetor_query = modelo.encode(
        [query],
        normalize_embeddings=True,
        show_progress_bar=False,
    )[0].tolist()

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

    scores_semanticos = {h.payload["chunk_id"]: h.score for h in hits}
    payloads          = {h.payload["chunk_id"]: h.payload for h in hits}

    # --- 3. Busca BM25 ---
    tokens = tokenizar(query)
    scores_raw = bm25.get_scores(tokens)
    max_bm25 = max(scores_raw) if max(scores_raw) > 0 else 1.0

    scores_bm25 = {}
    for i, score in enumerate(scores_raw):
        if score > 0:
            scores_bm25[bm25_ids[i]] = score / max_bm25

    # --- 4. Combinação híbrida ---
    todos_ids = set(scores_semanticos) | set(scores_bm25)
    scores_finais = {
        id_: (peso_semantico * scores_semanticos.get(id_, 0.0))
           + (peso_bm25      * scores_bm25.get(id_, 0.0))
        for id_ in todos_ids
    }

    top_ids = sorted(scores_finais, key=scores_finais.get, reverse=True)[:n_resultados]

    # --- 5. Monta resultado ---
    resultados = []
    for id_ in top_ids:
        payload = payloads.get(id_, {})
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
            "url_pdf":         payload.get("url_pdf", ""),
            "texto":           payload.get("texto", ""),
        })

    return resultados


def carregar_indices(qclient: QdrantClient) -> tuple:
    """
    Carrega os índices BM25 do disco.
    Atalho para a P3 usar no agente:

      from src.p2_search.p2_indexar import conectar_qdrant, carregar_indices, buscar
      qclient = conectar_qdrant()
      bm25, bm25_ids = carregar_indices(qclient)
      resultados = buscar("qual a tarifa TUSD?", qclient, bm25, bm25_ids)
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
    """Roda perguntas de teste para validar o sistema."""
    perguntas = [
        "qual a tarifa de uso do sistema de distribuição TUSD",
        "pequena central hidrelétrica PCH autorizada no rio",
        "liberação de unidade geradora para operação comercial",
        "Despacho 3409 percentual de redução TUSD",
        "multa auto de infração distribuidora penalidade",
    ]

    log.info("\n" + "=" * 60)
    log.info("TESTE DE BUSCA HÍBRIDA")
    log.info("=" * 60)

    for pergunta in perguntas:
        log.info(f"\nQuery: '{pergunta}'")
        log.info("-" * 50)
        resultados = buscar(pergunta, qclient, bm25, bm25_ids, n_resultados=3)
        for i, r in enumerate(resultados, 1):
            log.info(
                f"  {i}. [{r['score_final']:.3f}] "
                f"{r['tipo_nome']} nº {r['numero']}/{r['ano']}"
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

    # Lê parquet
    parquet_path = Path(args.parquet)
    if not parquet_path.exists():
        log.error(f"Parquet não encontrado: {parquet_path}")
        log.error("Rode primeiro: python src/p1_ingestion/chunker_json.py")
        return

    log.info(f"Lendo {parquet_path.name}...")
    df = pd.read_parquet(parquet_path)
    log.info(f"  {len(df)} chunks carregados")

    if args.limite:
        df = df.head(args.limite)
        log.info(f"  Modo teste: usando apenas {len(df)} chunks")

    if "chunk_id" not in df.columns:
        df["chunk_id"] = df.index.astype(str)

    # Cria coleção
    criar_colecao(qclient, resetar=args.resetar)

    # Verifica se já está indexado
    count = qclient.count(NOME_COLECAO).count
    if count >= len(df) and not args.resetar:
        log.info(f"Qdrant já tem {count} pontos. Pulando embeddings.")
    else:
        # Gera embeddings localmente
        log.info(f"Gerando embeddings para {len(df)} chunks (local, sem custo)...")
        textos     = df["texto"].tolist()
        embeddings = gerar_embeddings(textos)

        # Indexa no Qdrant
        indexar_qdrant(df, embeddings, qclient)

    # Cria BM25
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
