"""
setup_pipeline.py
=================
Script de setup completo do projeto RAG ANEEL.

Executa em sequência:
  1. P1a — Limpeza dos JSONs (limpar_json_aneel.py)
  2. P1b — Download dos PDFs (baixar_pdfs_aneel.py)
  3. P1c — Chunking das ementas (chunker_json.py)
  4. P1d — Parser dos PDFs/HTML/XLSX (parser.py)
  5. P1e — União dos parquets (unir_parquets.py)
  6. P2  — Indexação Qdrant + BM25 (p2_indexar.py)

Pré-requisitos:
  - Docker rodando com Qdrant: docker compose up -d
  - pip install -r requirements.txt
  - Arquivos JSON brutos em data/raw/
  - Poppler em poppler/Library/bin/ (para OCR)

Uso:
  # Setup completo (primeira vez):
  python scripts/setup_pipeline.py

  # Só download de PDFs:
  python scripts/setup_pipeline.py --apenas-download

  # Só parser (PDFs já baixados):
  python scripts/setup_pipeline.py --apenas-parser

  # Só P2 (parquets já gerados):
  python scripts/setup_pipeline.py --apenas-p2

  # Reindexar do zero:
  python scripts/setup_pipeline.py --apenas-p2 --resetar

  # Modo teste rápido:
  python scripts/setup_pipeline.py --limite 50 --testar

  # Sem OCR (mais rápido):
  python scripts/setup_pipeline.py --sem-ocr
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

RAIZ = Path(__file__).resolve().parent.parent

# ---------------------------------------------------------------------------
# Caminhos
# ---------------------------------------------------------------------------

# JSONs brutos
JSON_2016 = RAIZ / "data" / "raw" / "biblioteca_aneel_gov_br_legislacao_2016_metadados.json"
JSON_2021 = RAIZ / "data" / "raw" / "biblioteca_aneel_gov_br_legislacao_2021_metadados.json"
JSON_2022 = RAIZ / "data" / "raw" / "biblioteca_aneel_gov_br_legislacao_2022_metadados.json"

# Saídas P1
JSON_VIGENTES      = RAIZ / "data" / "aneel_vigentes_completo.json"
PARQUET_EMENTAS    = RAIZ / "data" / "processed" / "chunks_json_todos.parquet"
PARQUET_PDFS       = RAIZ / "data" / "processed" / "chunks_pdf_completo.parquet"
PARQUET_UNIFICADO  = RAIZ / "data" / "processed" / "chunks_completo_unificado.parquet"

# Pasta de PDFs baixados
PASTA_PDFS = RAIZ / "pdfs"


# ---------------------------------------------------------------------------
# Verificações
# ---------------------------------------------------------------------------

def verificar_docker() -> bool:
    import urllib.request
    try:
        urllib.request.urlopen("http://localhost:6333/collections", timeout=5)
        log.info("Qdrant OK — http://localhost:6333")
        return True
    except Exception:
        log.error("Qdrant não está acessível.")
        log.error("Suba o container: docker compose up -d")
        return False


def verificar_jsons() -> dict | None:
    """Procura JSONs brutos em data/raw/ ou na raiz."""
    todos = {JSON_2016: JSON_2016, JSON_2021: JSON_2021, JSON_2022: JSON_2022}
    faltando = [p for p in todos if not p.exists()]

    if not faltando:
        return todos

    # Fallback: raiz do projeto
    fallbacks = {
        JSON_2016: RAIZ / "biblioteca_aneel_gov_br_legislacao_2016_metadados.json",
        JSON_2021: RAIZ / "biblioteca_aneel_gov_br_legislacao_2021_metadados.json",
        JSON_2022: RAIZ / "biblioteca_aneel_gov_br_legislacao_2022_metadados.json",
    }
    encontrados = {k: v for k, v in fallbacks.items() if v.exists()}
    if encontrados:
        log.info(f"JSONs encontrados na raiz: {len(encontrados)}")
        return {**todos, **encontrados}

    log.error("JSONs brutos não encontrados em data/raw/ nem na raiz.")
    return None


# ---------------------------------------------------------------------------
# FASE 1a — Limpeza dos JSONs
# ---------------------------------------------------------------------------

def rodar_limpeza(jsons: dict) -> bool:
    log.info("=" * 55)
    log.info("FASE 1a — Limpeza dos JSONs")
    log.info("=" * 55)

    if JSON_VIGENTES.exists():
        log.info(f"JSON limpo já existe — pulando.")
        return True

    sys.path.insert(0, str(RAIZ / "src" / "p1_ingestion"))
    try:
        from limpar_json_aneel import processar, imprimir_resumo
    except ImportError as e:
        log.error(f"Erro ao importar limpar_json_aneel: {e}")
        return False

    import json as _json

    todos = []
    for ano_fonte, path in [
        ("2016", jsons.get(JSON_2016, JSON_2016)),
        ("2021", jsons.get(JSON_2021, JSON_2021)),
        ("2022", jsons.get(JSON_2022, JSON_2022)),
    ]:
        if not path.exists():
            log.warning(f"JSON {ano_fonte} não encontrado, pulando.")
            continue
        registros = processar(path)
        for r in registros:
            r["ano_fonte"] = ano_fonte
        todos.extend(registros)
        log.info(f"  {ano_fonte}: {len(registros)} registros")

    if not todos:
        log.error("Nenhum registro processado.")
        return False

    imprimir_resumo(todos)

    pasta = RAIZ / "data"
    pasta.mkdir(exist_ok=True)

    with open(RAIZ / "data" / "aneel_limpo_completo.json", "w", encoding="utf-8") as f:
        _json.dump(todos, f, ensure_ascii=False, indent=2)

    vigentes = [r for r in todos if r.get("vigente", True)]
    with open(JSON_VIGENTES, "w", encoding="utf-8") as f:
        _json.dump(vigentes, f, ensure_ascii=False, indent=2)

    log.info(f"JSON limpo salvo: {len(vigentes)} registros vigentes")
    return True


# ---------------------------------------------------------------------------
# FASE 1b — Download dos PDFs
# ---------------------------------------------------------------------------

def rodar_download(categorias: list, limite: int = None) -> bool:
    log.info("=" * 55)
    log.info("FASE 1b — Download dos PDFs")
    log.info("=" * 55)

    if not JSON_VIGENTES.exists():
        log.error("JSON vigentes não encontrado. Rode a limpeza primeiro.")
        return False

    # Verifica se já tem PDFs baixados
    pdfs_existentes = list(PASTA_PDFS.rglob("*.pdf")) if PASTA_PDFS.exists() else []
    if pdfs_existentes and not limite:
        log.info(f"  {len(pdfs_existentes)} PDFs já existem em disco.")
        resposta = input("  Redownload? (s/N): ").strip().lower()
        if resposta != "s":
            log.info("  Pulando download.")
            return True

    sys.path.insert(0, str(RAIZ / "src" / "p1_ingestion"))
    try:
        from baixar_pdfs_aneel import coletar_downloads, baixar_todos, salvar_falhas
    except ImportError as e:
        log.error(f"Erro ao importar baixar_pdfs_aneel: {e}")
        return False

    downloads = coletar_downloads(JSON_VIGENTES, categorias, [], limite)
    if not downloads:
        log.info("  Nada a baixar.")
        return True

    log.info(f"  {len(downloads)} arquivos para baixar...")
    resultados, falhas = baixar_todos(downloads, workers=5)
    salvar_falhas(falhas)

    log.info(f"  Baixados: {resultados.get('ok', 0)} | Erros: {len(falhas)}")

    # Segunda rodada — baixa PDFs referenciados nos HTMLs (inclui 2015)
    # O baixar_pdfs_aneel.py segue links dentro dos HTMLs e encontra
    # documentos de anos não cobertos pelo JSON (ex: 2015)
    log.info("  Verificando PDFs referenciados nos HTMLs...")
    try:
        from baixar_pdfs_aneel import baixar_referenciados_html
        baixar_referenciados_html()
    except (ImportError, AttributeError):
        # Função não disponível — rodar manualmente se necessário:
        # python src\p1_ingestion\baixar_pdfs_aneel.py
        log.info("  Para baixar referenciados: python src/p1_ingestion/baixar_pdfs_aneel.py")

    return True


# ---------------------------------------------------------------------------
# FASE 1c — Chunking das ementas
# ---------------------------------------------------------------------------

def rodar_chunking_ementas(limite: int = None) -> bool:
    log.info("=" * 55)
    log.info("FASE 1c — Chunking das ementas")
    log.info("=" * 55)

    if PARQUET_EMENTAS.exists() and not limite:
        log.info(f"Parquet de ementas já existe — pulando.")
        return True

    if not JSON_VIGENTES.exists():
        log.error("JSON vigentes não encontrado.")
        return False

    sys.path.insert(0, str(RAIZ / "src" / "p1_ingestion"))
    try:
        from chunker_json import gerar_chunks, salvar_parquet
    except ImportError as e:
        log.error(f"Erro ao importar chunker_json: {e}")
        return False

    import json as _json
    with open(JSON_VIGENTES, encoding="utf-8") as f:
        registros = _json.load(f)

    log.info(f"  Gerando chunks de {len(registros)} registros...")
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

    PARQUET_EMENTAS.parent.mkdir(parents=True, exist_ok=True)
    salvar_parquet(chunks, PARQUET_EMENTAS)
    log.info(f"  {len(chunks)} chunks em {time.time()-inicio:.0f}s")
    return True


# ---------------------------------------------------------------------------
# FASE 1d — Parser dos PDFs/HTML/XLSX
# ---------------------------------------------------------------------------

def rodar_parser(limite: int = None, usar_ocr: bool = True, workers: int = 16) -> bool:
    log.info("=" * 55)
    log.info("FASE 1d — Parser PDFs/HTML/XLSX")
    log.info("=" * 55)

    if PARQUET_PDFS.exists() and not limite:
        log.info(f"Parquet de PDFs já existe — pulando.")
        return True

    if not JSON_VIGENTES.exists():
        log.error("JSON vigentes não encontrado.")
        return False

    sys.path.insert(0, str(RAIZ / "src" / "p1_ingestion"))
    try:
        from parser import (
            coletar_tarefas, worker, ParquetWriter,
            SCHEMA, BATCH_WRITE,
        )
        from concurrent.futures import ProcessPoolExecutor, as_completed
    except ImportError as e:
        log.error(f"Erro ao importar parser: {e}")
        log.error("Instale: pip install pymupdf pdfplumber")
        return False

    try:
        from tqdm import tqdm
        usa_tqdm = True
    except ImportError:
        usa_tqdm = False

    tarefas = coletar_tarefas(JSON_VIGENTES, [], [], limite)
    log.info(f"  {len(tarefas)} arquivos para processar")

    worker_args = [
        (str(cam), meta, 800, 120)
        for cam, meta in tarefas
    ]

    writer = ParquetWriter(PARQUET_PDFS)
    buffer = []
    erros  = 0

    barra = tqdm(total=len(tarefas), desc="Parser",
                 unit="arq", dynamic_ncols=True) if usa_tqdm else None

    with ProcessPoolExecutor(max_workers=workers) as ex:
        futuros = {ex.submit(worker, a): a for a in worker_args}
        for fut in as_completed(futuros):
            try:
                chunks = fut.result()
                buffer.extend(chunks)
                if len(buffer) >= BATCH_WRITE:
                    writer.write(buffer)
                    buffer = []
            except Exception as e:
                erros += 1
            if barra:
                barra.update(1)

    if buffer:
        writer.write(buffer)
    total = writer.close()
    if barra:
        barra.close()

    log.info(f"  {total:,} chunks gerados | {erros} erros")
    return True


# ---------------------------------------------------------------------------
# FASE 1e — União dos parquets
# ---------------------------------------------------------------------------

def rodar_uniao() -> bool:
    log.info("=" * 55)
    log.info("FASE 1e — União dos parquets")
    log.info("=" * 55)

    if not PARQUET_EMENTAS.exists():
        log.error(f"Parquet de ementas não encontrado: {PARQUET_EMENTAS}")
        return False

    if not PARQUET_PDFS.exists():
        log.error(f"Parquet de PDFs não encontrado: {PARQUET_PDFS}")
        return False

    sys.path.insert(0, str(RAIZ / "src" / "p1_ingestion"))
    try:
        from unir_parquets import carregar, alinhar_colunas, unir
    except ImportError as e:
        log.error(f"Erro ao importar unir_parquets: {e}")
        return False

    import pandas as pd

    df_ementas = carregar(PARQUET_EMENTAS, "ementas")
    df_pdfs    = carregar(PARQUET_PDFS,    "PDFs")

    df_ementas, df_pdfs = alinhar_colunas(df_ementas, df_pdfs)
    df_final = unir(df_ementas, df_pdfs)

    PARQUET_UNIFICADO.parent.mkdir(parents=True, exist_ok=True)
    df_final.to_parquet(PARQUET_UNIFICADO, index=False, compression="snappy")

    log.info(f"  {len(df_final):,} chunks unificados")
    log.info(f"  Salvo: {PARQUET_UNIFICADO.name}")
    return True


# ---------------------------------------------------------------------------
# FASE 2 — Indexação Qdrant + BM25
# ---------------------------------------------------------------------------

def rodar_p2(
    parquet: Path = None,
    limite: int = None,
    resetar: bool = False,
    testar: bool = False,
) -> bool:
    log.info("=" * 55)
    log.info("FASE 2 — Indexação Qdrant + BM25")
    log.info("=" * 55)

    # Usa o parquet unificado se existir, senão o de ementas
    if parquet is None:
        if PARQUET_UNIFICADO.exists():
            parquet = PARQUET_UNIFICADO
            log.info(f"  Usando parquet unificado.")
        elif PARQUET_EMENTAS.exists():
            parquet = PARQUET_EMENTAS
            log.info(f"  Parquet unificado não encontrado — usando só ementas.")
        else:
            log.error("Nenhum parquet encontrado. Rode as fases anteriores primeiro.")
            return False

    sys.path.insert(0, str(RAIZ / "src" / "p2_search"))
    try:
        from p2_indexar import (
            conectar_qdrant, criar_colecao, gerar_embeddings,
            indexar_qdrant, criar_bm25, NOME_COLECAO,
        )
        from p2_indexar import testar as testar_busca
    except ImportError as e:
        log.error(f"Erro ao importar p2_indexar: {e}")
        return False

    import pandas as pd

    qclient = conectar_qdrant()

    log.info(f"  Lendo {parquet.name}...")
    df = pd.read_parquet(parquet)
    log.info(f"  {len(df):,} chunks")

    if limite:
        df = df.head(limite)
        log.info(f"  Modo teste: {len(df)} chunks")

    if "chunk_id" not in df.columns:
        df["chunk_id"] = df.index.astype(str)

    criar_colecao(qclient, resetar=resetar)

    count = qclient.count(NOME_COLECAO).count
    if count >= len(df) and not resetar:
        log.info(f"  Qdrant já tem {count:,} pontos — pulando embeddings.")
    else:
        log.info(f"  Gerando embeddings...")
        embeddings = gerar_embeddings(df["texto"].tolist())
        indexar_qdrant(df, embeddings, qclient)

    bm25, bm25_ids = criar_bm25(df)

    log.info("=" * 55)
    log.info("INDEXAÇÃO CONCLUÍDA")
    log.info(f"  Qdrant : {qclient.count(NOME_COLECAO).count:,} pontos")
    log.info(f"  BM25   : {len(bm25_ids):,} documentos")
    log.info("=" * 55)

    if testar:
        testar_busca(qclient, bm25, bm25_ids)

    return True


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Setup completo do pipeline RAG ANEEL"
    )
    ap.add_argument("--apenas-download", action="store_true",
                    help="Só baixa os PDFs")
    ap.add_argument("--apenas-parser",   action="store_true",
                    help="Só roda o parser (PDFs já baixados)")
    ap.add_argument("--apenas-p1",       action="store_true",
                    help="Só P1 completo (limpeza + download + chunking + parser + união)")
    ap.add_argument("--apenas-p2",       action="store_true",
                    help="Só P2 (indexação Qdrant + BM25)")
    ap.add_argument("--limite",          type=int, default=None,
                    help="Processar só N itens (teste)")
    ap.add_argument("--resetar",         action="store_true",
                    help="Apaga e recria coleção Qdrant")
    ap.add_argument("--testar",          action="store_true",
                    help="Roda perguntas de teste após indexar")
    ap.add_argument("--sem-ocr",         action="store_true",
                    help="Desabilita OCR no parser (mais rápido)")
    ap.add_argument("--workers",         type=int, default=16,
                    help="Workers do parser (padrão: 16)")
    ap.add_argument(
        "--categorias", nargs="+",
        default=["texto_integral", "voto", "nota_tecnica", "decisao", "anexo", "outro"],
        help="Categorias de PDF para download",
    )
    args = ap.parse_args()

    usar_ocr  = not args.sem_ocr
    inicio    = time.time()

    log.info("=" * 55)
    log.info("RAG ANEEL — SETUP PIPELINE COMPLETO")
    log.info("=" * 55)

    # Determina quais fases rodar
    rodar_download_flag = not (args.apenas_p2 or args.apenas_parser)
    rodar_parser_flag   = not (args.apenas_p2 or args.apenas_download)
    rodar_p2_flag       = not (args.apenas_p1 or args.apenas_download or args.apenas_parser)

    # Verifica Qdrant se vai rodar P2
    if rodar_p2_flag:
        if not verificar_docker():
            log.error("Suba o Qdrant antes: docker compose up -d")
            sys.exit(1)

    # ── P1a: Limpeza ─────────────────────────────────────────────────
    if not args.apenas_p2:
        jsons = verificar_jsons()
        if not JSON_VIGENTES.exists():
            if not jsons:
                log.error("JSONs brutos não encontrados em data/raw/")
                sys.exit(1)
            if not rodar_limpeza(jsons):
                log.error("Limpeza falhou.")
                sys.exit(1)
        else:
            log.info("JSON limpo já existe — pulando limpeza.")

    # ── P1b: Download ────────────────────────────────────────────────
    if rodar_download_flag:
        if not rodar_download(args.categorias, args.limite):
            log.error("Download falhou.")
            sys.exit(1)

    # ── P1c: Chunking ementas ────────────────────────────────────────
    if rodar_parser_flag or not args.apenas_p2:
        if not rodar_chunking_ementas(args.limite):
            log.error("Chunking de ementas falhou.")
            sys.exit(1)

    # ── P1d: Parser PDFs ─────────────────────────────────────────────
    if rodar_parser_flag:
        if not rodar_parser(args.limite, usar_ocr, args.workers):
            log.error("Parser falhou.")
            sys.exit(1)

    # ── P1e: União ───────────────────────────────────────────────────
    if rodar_parser_flag and PARQUET_PDFS.exists():
        if not rodar_uniao():
            log.error("União dos parquets falhou.")
            sys.exit(1)

    # ── P2: Indexação ────────────────────────────────────────────────
    if rodar_p2_flag:
        if not rodar_p2(
            limite=args.limite,
            resetar=args.resetar,
            testar=args.testar,
        ):
            log.error("Indexação falhou.")
            sys.exit(1)

    elapsed = time.time() - inicio
    h = int(elapsed // 3600)
    m = int((elapsed % 3600) // 60)

    log.info("=" * 55)
    log.info(f"PIPELINE COMPLETO em {h}h {m}min")
    log.info("=" * 55)
    log.info("Próximo passo: implementar o agente da P3")
    log.info("  from src.p2_search.p2_indexar import conectar_qdrant, carregar_indices, buscar")


if __name__ == "__main__":
    main()