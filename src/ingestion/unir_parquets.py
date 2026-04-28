"""
unir_parquets.py
================
Pessoa 1 — Data Engineer

Une os dois parquets gerados:
  1. chunks_json_todos.parquet    — chunks das ementas (cobertura ampla)
  2. chunks_pdf_completo.parquet  — chunks dos PDFs completos (conteúdo rico)

Estratégia de deduplicação:
  - Para cada doc_id que tem PDF completo → usa chunks do PDF (mais rico)
  - Para cada doc_id que SÓ tem ementa → usa chunks da ementa (única fonte)
  - Resultado: cobertura máxima com qualidade máxima

Uso:
  python src/ingestion/unir_parquets.py
  python src/ingestion/unir_parquets.py --preview   # só mostra estatísticas
"""

import argparse
import logging
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

RAIZ          = Path(__file__).resolve().parent.parent.parent
PASTA_DATA    = RAIZ / "data" / "processed"

PARQUET_EMENTAS = PASTA_DATA / "chunks_json_todos.parquet"
PARQUET_PDFS    = PASTA_DATA / "chunks_pdf_completo.parquet"
PARQUET_FINAL   = PASTA_DATA / "chunks_completo_unificado.parquet"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def carregar(path: Path, nome: str) -> pd.DataFrame:
    """Carrega parquet e reporta estatísticas."""
    log.info(f"Carregando {nome}...")
    df = pd.read_parquet(path)
    log.info(f"  {len(df):,} chunks | {df['doc_id'].nunique():,} documentos únicos")
    log.info(f"  Colunas: {list(df.columns)}")
    return df


def alinhar_colunas(df_ementas: pd.DataFrame, df_pdfs: pd.DataFrame) -> tuple:
    """
    Garante que os dois DataFrames têm as mesmas colunas antes de concatenar.
    O parquet de PDFs tem colunas extras (tem_tabela, tem_tachado, etc.)
    que não existem no de ementas — preenche com valores padrão.
    """
    # Colunas que existem no PDF mas não na ementa
    colunas_pdf_only = set(df_pdfs.columns) - set(df_ementas.columns)

    # Colunas que existem na ementa mas não no PDF
    colunas_ementa_only = set(df_ementas.columns) - set(df_pdfs.columns)

    if colunas_pdf_only:
        log.info(f"  Colunas só no PDF (adicionando à ementa): {colunas_pdf_only}")
        for col in colunas_pdf_only:
            dtype = df_pdfs[col].dtype
            if dtype == bool:
                df_ementas[col] = False
            elif dtype in ("int32", "int64"):
                df_ementas[col] = 0
            else:
                df_ementas[col] = ""

    if colunas_ementa_only:
        log.info(f"  Colunas só na ementa (adicionando ao PDF): {colunas_ementa_only}")
        for col in colunas_ementa_only:
            dtype = df_ementas[col].dtype
            if dtype == bool:
                df_pdfs[col] = False
            elif dtype in ("int32", "int64"):
                df_pdfs[col] = 0
            else:
                df_pdfs[col] = ""

    # Garante mesma ordem de colunas
    colunas_ordem = list(df_pdfs.columns)
    df_ementas = df_ementas[colunas_ordem]
    df_pdfs    = df_pdfs[colunas_ordem]

    return df_ementas, df_pdfs


# ---------------------------------------------------------------------------
# Lógica de deduplicação
# ---------------------------------------------------------------------------

def unir(
    df_ementas: pd.DataFrame,
    df_pdfs: pd.DataFrame,
) -> pd.DataFrame:
    """
    Para cada doc_id:
      - Se tem chunks de PDF completo → usa PDF (descarta ementa do mesmo doc)
      - Se só tem ementa → mantém ementa

    Isso garante:
      ✓ Conteúdo rico (PDF) quando disponível
      ✓ Cobertura ampla (ementa) para documentos sem PDF
    """
    # doc_ids que têm PDF completo
    docs_com_pdf = set(df_pdfs["doc_id"].unique())

    # Ementas de docs que NÃO têm PDF — esses precisamos manter
    ementas_sem_pdf = df_ementas[~df_ementas["doc_id"].isin(docs_com_pdf)].copy()

    # Marca a fonte para rastreabilidade
    ementas_sem_pdf["fonte_texto"] = "ementa"
    df_pdfs_final = df_pdfs.copy()

    log.info(f"\n  Documentos com PDF completo : {len(docs_com_pdf):,}")
    log.info(f"  Documentos só com ementa    : {df_ementas['doc_id'].nunique() - len(docs_com_pdf.intersection(set(df_ementas['doc_id'].unique()))):,}")
    log.info(f"  Chunks de PDF               : {len(df_pdfs_final):,}")
    log.info(f"  Chunks de ementa (sem PDF)  : {len(ementas_sem_pdf):,}")

    # Concatena
    df_final = pd.concat([df_pdfs_final, ementas_sem_pdf], ignore_index=True)

    return df_final


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Une parquets de ementas e PDFs completos com deduplicação"
    )
    ap.add_argument(
        "--ementas", default=str(PARQUET_EMENTAS),
        help="Parquet das ementas (chunks_json_todos.parquet)",
    )
    ap.add_argument(
        "--pdfs", default=str(PARQUET_PDFS),
        help="Parquet dos PDFs completos (chunks_pdf_completo.parquet)",
    )
    ap.add_argument(
        "--saida", default=str(PARQUET_FINAL),
        help="Parquet de saída unificado",
    )
    ap.add_argument(
        "--preview", action="store_true",
        help="Só mostra estatísticas sem salvar",
    )
    args = ap.parse_args()

    path_ementas = Path(args.ementas)
    path_pdfs    = Path(args.pdfs)
    path_saida   = Path(args.saida)

    # Verifica arquivos
    if not path_ementas.exists():
        log.error(f"Parquet de ementas não encontrado: {path_ementas}")
        return
    if not path_pdfs.exists():
        log.error(f"Parquet de PDFs não encontrado: {path_pdfs}")
        log.error("Rode primeiro: python src/ingestion/parser.py")
        return

    log.info("=" * 55)
    log.info("UNINDO PARQUETS")
    log.info("=" * 55)

    # Carrega
    df_ementas = carregar(path_ementas, "ementas")
    df_pdfs    = carregar(path_pdfs,    "PDFs completos")

    # Alinha colunas
    log.info("\nAlinhando colunas...")
    df_ementas, df_pdfs = alinhar_colunas(df_ementas, df_pdfs)

    # Une com deduplicação
    log.info("\nAplicando deduplicação...")
    df_final = unir(df_ementas, df_pdfs)

    # Estatísticas finais
    log.info("\n" + "=" * 55)
    log.info("RESULTADO")
    log.info("=" * 55)
    # Chunks que vieram de PDF, HTML, XLSX ou OCR (tudo exceto ementa pura)
    fontes_pdf = ['pdf_completo', 'pdf_ocr', 'pdf_imagem', 'html', 'xlsx', 'html_ren', '']
    n_nao_ementa = (df_final['fonte_texto'] != 'ementa').sum()
    n_ementa     = (df_final['fonte_texto'] == 'ementa').sum()

    log.info(f"  Total de chunks      : {len(df_final):,}")
    log.info(f"  Documentos únicos    : {df_final['doc_id'].nunique():,}")
    log.info(f"  Chunks de arquivos   : {n_nao_ementa:,}  (PDF/HTML/XLSX/OCR)")
    log.info(f"  Chunks de ementa     : {n_ementa:,}")
    log.info(f"  Por fonte_texto:")
    for fonte, count in df_final['fonte_texto'].value_counts().items():
        log.info(f"    {fonte or '(vazio)'}: {count:,}")
    if "tem_tabela" in df_final.columns:
        log.info(f"  Chunks com tabela    : {df_final['tem_tabela'].sum():,}")
    if "tem_tachado" in df_final.columns:
        log.info(f"  Chunks com tachado   : {df_final['tem_tachado'].sum():,}")
    if "tem_imagem" in df_final.columns:
        log.info(f"  Chunks com imagem    : {df_final['tem_imagem'].sum():,}")
    log.info(f"  Tamanho médio chunk  : {df_final['texto'].str.len().mean():.0f} chars")
    log.info("=" * 55)

    if args.preview:
        log.info("\nModo --preview: arquivo não salvo.")
        return

    # Salva
    path_saida.parent.mkdir(parents=True, exist_ok=True)
    log.info(f"\nSalvando em {path_saida}...")
    df_final.to_parquet(path_saida, index=False, compression="snappy")
    tamanho = path_saida.stat().st_size / 1024 / 1024
    log.info(f"Salvo! ({tamanho:.1f} MB)")

    log.info("\nProximo passo — reindexar no Qdrant com o parquet unificado:")
    log.info(f"  python src/search/indexar.py \\")
    log.info(f"    --parquet {path_saida} \\")
    log.info(f"    --resetar")


if __name__ == "__main__":
    main()