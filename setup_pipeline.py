"""
setup_pipeline.py
=================
Script de setup completo — roda P1 e P2 em sequência.

Executa:
  1. P1 — limpeza dos JSONs (limpar_json_aneel.py)
  2. P1 — geração dos chunks em parquet (chunker_json.py)
  3. P2 — indexação no Qdrant + BM25 (p2_indexar.py)

Pré-requisitos:
  - Docker rodando com Qdrant: docker compose up -d
  - pip install -r requirements.txt
  - Arquivos JSON brutos em data/raw/ ou na raiz

Uso:
  # Setup completo (primeira vez):
  python setup_pipeline.py

  # Só P1 (gerar chunks novamente):
  python setup_pipeline.py --apenas-p1

  # Só P2 (reindexar — parquet já existe):
  python setup_pipeline.py --apenas-p2

  # Reindexar do zero (apaga coleção Qdrant existente):
  python setup_pipeline.py --apenas-p2 --resetar

  # Modo teste rápido (50 chunks):
  python setup_pipeline.py --limite 50 --testar
"""

import argparse
import logging
import sys
import time
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

RAIZ = Path(__file__).resolve().parent

# ---------------------------------------------------------------------------
# Caminhos esperados
# ---------------------------------------------------------------------------

JSON_2016 = RAIZ / "data" / "raw" / "biblioteca_aneel_gov_br_legislacao_2016_metadados.json"
JSON_2021 = RAIZ / "data" / "raw" / "biblioteca_aneel_gov_br_legislacao_2021_metadados.json"
JSON_2022 = RAIZ / "data" / "raw" / "biblioteca_aneel_gov_br_legislacao_2022_metadados.json"

JSON_VIGENTES = RAIZ / "data" / "aneel_vigentes_completo.json"
PARQUET       = RAIZ / "data" / "processed" / "chunks_json_todos.parquet"


# ---------------------------------------------------------------------------
# Verificações iniciais
# ---------------------------------------------------------------------------

def verificar_docker():
    """Verifica se o Qdrant está acessível."""
    import urllib.request
    try:
        urllib.request.urlopen("http://localhost:6333/collections", timeout=5)
        log.info("Qdrant OK — rodando em http://localhost:6333")
        return True
    except Exception:
        log.error("Qdrant não está acessível em http://localhost:6333")
        log.error("Suba o container antes: docker compose up -d")
        return False


def verificar_jsons():
    """Verifica se os JSONs de entrada existem."""
    faltando = []
    for path in [JSON_2016, JSON_2021, JSON_2022]:
        if not path.exists():
            faltando.append(str(path))

    if faltando:
        log.warning("JSONs brutos não encontrados em data/raw/:")
        for f in faltando:
            log.warning(f"  {f}")

        # Tenta encontrar na raiz como fallback
        fallbacks = {
            JSON_2016: RAIZ / "biblioteca_aneel_gov_br_legislacao_2016_metadados.json",
            JSON_2021: RAIZ / "biblioteca_aneel_gov_br_legislacao_2021_metadados.json",
            JSON_2022: RAIZ / "biblioteca_aneel_gov_br_legislacao_2022_metadados.json",
        }
        encontrados = {k: v for k, v in fallbacks.items() if v.exists()}
        if encontrados:
            log.info(f"Encontrados {len(encontrados)} JSONs na raiz do projeto como fallback.")
            return encontrados
        return None

    return {JSON_2016: JSON_2016, JSON_2021: JSON_2021, JSON_2022: JSON_2022}


# ---------------------------------------------------------------------------
# Fase 1 — P1
# ---------------------------------------------------------------------------

def rodar_p1_limpeza(jsons: dict) -> bool:
    """Roda limpar_json_aneel.py para gerar o JSON limpo."""
    log.info("=" * 55)
    log.info("FASE 1a — Limpeza dos JSONs (P1)")
    log.info("=" * 55)

    if JSON_VIGENTES.exists():
        log.info(f"JSON limpo já existe: {JSON_VIGENTES.name} — pulando limpeza.")
        return True

    sys.path.insert(0, str(RAIZ / "src" / "p1_ingestion"))
    try:
        from limpar_json_aneel import processar, salvar, imprimir_resumo
    except ImportError as e:
        log.error(f"Não foi possível importar limpar_json_aneel: {e}")
        return False

    SITUACOES_INATIVAS = {"REVOGADA", "TORNADA SEM EFEITO", "ANULADA", "CADUCADA", "SUSPENSA"}

    import json, re
    from datetime import datetime

    todos = []
    for ano_fonte, path in [("2016", jsons.get(JSON_2016, JSON_2016)),
                             ("2021", jsons.get(JSON_2021, JSON_2021)),
                             ("2022", jsons.get(JSON_2022, JSON_2022))]:
        if not path.exists():
            log.warning(f"JSON {ano_fonte} não encontrado, pulando.")
            continue
        registros = processar(path)
        # Adiciona ano_fonte a cada registro
        for r in registros:
            r["ano_fonte"] = ano_fonte
        todos.extend(registros)
        log.info(f"  {ano_fonte}: {len(registros)} registros")

    if not todos:
        log.error("Nenhum registro processado.")
        return False

    imprimir_resumo(todos)

    pasta_saida = RAIZ / "data"
    pasta_saida.mkdir(exist_ok=True)

    import json as _json
    with open(RAIZ / "data" / "aneel_limpo_completo.json", "w", encoding="utf-8") as f:
        _json.dump(todos, f, ensure_ascii=False, indent=2)

    vigentes = [r for r in todos if r.get("vigente", True)]
    with open(JSON_VIGENTES, "w", encoding="utf-8") as f:
        _json.dump(vigentes, f, ensure_ascii=False, indent=2)

    log.info(f"JSON limpo salvo: {len(vigentes)} registros vigentes")
    return True


def rodar_p1_chunks(limite: int = None) -> bool:
    """Roda chunker_json.py para gerar o parquet."""
    log.info("=" * 55)
    log.info("FASE 1b — Chunking (P1)")
    log.info("=" * 55)

    if PARQUET.exists() and limite is None:
        log.info(f"Parquet já existe: {PARQUET.name} — pulando chunking.")
        return True

    if not JSON_VIGENTES.exists():
        log.error(f"JSON vigentes não encontrado: {JSON_VIGENTES}")
        log.error("Execute a limpeza primeiro.")
        return False

    sys.path.insert(0, str(RAIZ / "src" / "p1_ingestion"))
    try:
        from chunker_json import gerar_chunks, salvar_parquet, mostrar_preview
        from langchain_text_splitters import RecursiveCharacterTextSplitter
    except ImportError:
        try:
            from langchain.text_splitter import RecursiveCharacterTextSplitter
        except ImportError as e:
            log.error(f"Erro ao importar chunker: {e}")
            log.error("Instale: pip install langchain-text-splitters")
            return False

    import json as _json
    with open(JSON_VIGENTES, encoding="utf-8") as f:
        registros = _json.load(f)

    log.info(f"Gerando chunks de {len(registros)} registros...")
    inicio = time.time()

    chunks = gerar_chunks(
        registros=registros,
        anos=[],
        incluir_sem_ementa=False,
        limite=limite,
    )

    if not chunks:
        log.error("Nenhum chunk gerado.")
        return False

    mostrar_preview(chunks)

    PARQUET.parent.mkdir(parents=True, exist_ok=True)
    salvar_parquet(chunks, PARQUET)

    log.info(f"Chunking concluído em {time.time()-inicio:.0f}s — {len(chunks)} chunks")
    return True


# ---------------------------------------------------------------------------
# Fase 2 — P2
# ---------------------------------------------------------------------------

def rodar_p2(limite: int = None, resetar: bool = False, testar: bool = False) -> bool:
    """Roda p2_indexar.py para indexar no Qdrant + BM25."""
    log.info("=" * 55)
    log.info("FASE 2 — Indexação Qdrant + BM25 (P2)")
    log.info("=" * 55)

    if not PARQUET.exists():
        log.error(f"Parquet não encontrado: {PARQUET}")
        log.error("Execute a fase 1 primeiro.")
        return False

    sys.path.insert(0, str(RAIZ / "src" / "p2_search"))
    try:
        from p2_indexar import (
            conectar_qdrant, criar_colecao, gerar_embeddings,
            indexar_qdrant, criar_bm25, testar as testar_busca,
        )
    except ImportError as e:
        log.error(f"Não foi possível importar p2_indexar: {e}")
        return False

    import pandas as pd

    qclient = conectar_qdrant()

    log.info(f"Lendo {PARQUET.name}...")
    df = pd.read_parquet(PARQUET)
    log.info(f"  {len(df)} chunks carregados")

    if limite:
        df = df.head(limite)
        log.info(f"  Modo teste: {len(df)} chunks")

    if "chunk_id" not in df.columns:
        df["chunk_id"] = df.index.astype(str)

    criar_colecao(qclient, resetar=resetar)

    from p2_indexar import NOME_COLECAO
    count = qclient.count(NOME_COLECAO).count
    if count >= len(df) and not resetar:
        log.info(f"Qdrant já tem {count} pontos — pulando embeddings.")
    else:
        log.info(f"Gerando embeddings para {len(df)} chunks...")
        embeddings = gerar_embeddings(df["texto"].tolist())
        indexar_qdrant(df, embeddings, qclient)

    bm25, bm25_ids = criar_bm25(df)

    log.info("=" * 55)
    log.info("INDEXAÇÃO CONCLUÍDA")
    log.info(f"  Qdrant : {qclient.count(NOME_COLECAO).count} pontos")
    log.info(f"  BM25   : {len(bm25_ids)} documentos")
    log.info("=" * 55)

    if testar:
        testar_busca(qclient, bm25, bm25_ids)

    return True


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Setup completo do pipeline RAG ANEEL (P1 + P2)"
    )
    parser.add_argument(
        "--apenas-p1", action="store_true",
        help="Roda só a fase 1 (limpeza + chunking)",
    )
    parser.add_argument(
        "--apenas-p2", action="store_true",
        help="Roda só a fase 2 (indexação Qdrant + BM25)",
    )
    parser.add_argument(
        "--limite", type=int, default=None,
        help="Processar apenas N registros/chunks (modo teste)",
    )
    parser.add_argument(
        "--resetar", action="store_true",
        help="Apaga e recria a coleção Qdrant",
    )
    parser.add_argument(
        "--testar", action="store_true",
        help="Roda perguntas de teste após indexar",
    )
    args = parser.parse_args()

    inicio_total = time.time()

    log.info("=" * 55)
    log.info("RAG ANEEL — SETUP PIPELINE")
    log.info("=" * 55)

    rodar_p1 = not args.apenas_p2
    rodar_p2_flag = not args.apenas_p1

    # Verifica Qdrant se vai rodar P2
    if rodar_p2_flag:
        if not verificar_docker():
            log.error("Abortando. Suba o Qdrant antes de continuar.")
            sys.exit(1)

    # Fase 1
    if rodar_p1:
        jsons = verificar_jsons()

        if not PARQUET.exists():
            if not jsons:
                log.error("JSONs brutos não encontrados. Coloque-os em data/raw/")
                sys.exit(1)

            ok = rodar_p1_limpeza(jsons)
            if not ok:
                log.error("Fase 1a falhou.")
                sys.exit(1)

        ok = rodar_p1_chunks(limite=args.limite)
        if not ok:
            log.error("Fase 1b falhou.")
            sys.exit(1)

    # Fase 2
    if rodar_p2_flag:
        ok = rodar_p2(
            limite=args.limite,
            resetar=args.resetar,
            testar=args.testar,
        )
        if not ok:
            log.error("Fase 2 falhou.")
            sys.exit(1)

    log.info("=" * 55)
    log.info(f"PIPELINE COMPLETO em {time.time()-inicio_total:.0f}s")
    log.info("=" * 55)
    log.info("Próximo passo: implementar o agente da P3")
    log.info("  from src.p2_search.p2_indexar import conectar_qdrant, carregar_indices, buscar")


if __name__ == "__main__":
    main()
