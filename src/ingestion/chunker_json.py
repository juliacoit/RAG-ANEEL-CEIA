"""
chunker_json.py
===============
Pessoa 1 — Data Engineer · Sprint (fallback sem PDFs)

Gera chunks diretamente do JSON de metadados da ANEEL,
usando ementa + metadados como texto — sem precisar dos PDFs.

Quando os PDFs estiverem disponíveis, o chunker.py completo
(que usa o parser.py) substitui este script.

Uso — rodar da RAIZ do projeto:
  python src/ingestion/chunker_json.py
  python src/ingestion/chunker_json.py --ano 2016
  python src/ingestion/chunker_json.py --ano 2016 --limite 50   # teste
  python src/ingestion/chunker_json.py --incluir-sem-ementa      # inclui todos

Saída:
  data/processed/chunks_json_<ano>.parquet

Estrutura de cada chunk gerado:
  {
    "chunk_id":        "DSP_3386_2016_0",   # id único do chunk
    "doc_id":          "DSP_3386_2016",     # id do ato normativo
    "tipo_codigo":     "DSP",
    "tipo_nome":       "Despacho",
    "numero":          "3386",
    "ano":             "2016",
    "autor":           "SCG/ANEEL",
    "assunto":         "Registro",
    "data_assinatura": "2016-12-26",
    "data_publicacao": "2016-12-30",
    "url_pdf":         "https://...",
    "fonte_texto":     "ementa" | "metadados",  # de onde veio o texto
    "chunk_index":     0,                        # posição do chunk no doc
    "chunk_total":     1,                        # total de chunks do doc
    "texto":           "..."                     # texto do chunk
  }
"""

import json
import logging
import argparse
from pathlib import Path

import pandas as pd
try:
    from langchain.text_splitter import RecursiveCharacterTextSplitter
except ImportError:
    from langchain_text_splitters import RecursiveCharacterTextSplitter

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Caminhos
# ---------------------------------------------------------------------------

RAIZ_PROJETO = Path(__file__).resolve().parent.parent.parent
ARQUIVO_JSON = RAIZ_PROJETO / "data" / "aneel_vigentes_completo.json"
PASTA_SAIDA  = RAIZ_PROJETO / "data" / "processed"

# ---------------------------------------------------------------------------
# Configurações de chunking
# ---------------------------------------------------------------------------

CHUNK_SIZE    = 600    # tokens aproximados (characters / 4 ≈ tokens)
CHUNK_OVERLAP = 90     # ~15% de overlap entre chunks
CHUNK_SIZE_CHARS    = CHUNK_SIZE * 4      # 2400 caracteres
CHUNK_OVERLAP_CHARS = CHUNK_OVERLAP * 4  # 360 caracteres

# ---------------------------------------------------------------------------
# Monta o texto de cada registro
# ---------------------------------------------------------------------------

def montar_texto(registro: dict) -> tuple[str, str]:
    """
    Constrói o texto mais rico possível a partir dos metadados do registro.

    Estratégia:
      - Se tem ementa: usa título + assunto + ementa (texto principal)
      - Se não tem ementa: usa título + assunto + situação (texto mínimo)

    Retorna:
      texto     (str) — texto montado
      fonte     (str) — "ementa" ou "metadados"
    """
    tipo_nome      = registro.get("tipo_nome", "")
    numero         = registro.get("numero", "")
    ano            = registro.get("ano", "")
    autor          = registro.get("autor", "")
    assunto        = registro.get("assunto", "")
    situacao       = registro.get("situacao", "")
    ementa         = registro.get("ementa") or ""
    data_assinatura = registro.get("data_assinatura", "")

    # Cabeçalho padronizado — sempre presente
    cabecalho = (
        f"{tipo_nome} nº {numero}/{ano}\n"
        f"Autor: {autor}\n"
        f"Data: {data_assinatura}\n"
        f"Assunto: {assunto}\n"
        f"Situação: {situacao}\n"
    )

    if ementa.strip():
        texto  = cabecalho + f"\nEmenta:\n{ementa.strip()}"
        fonte  = "ementa"
    else:
        texto  = cabecalho
        fonte  = "metadados"

    return texto.strip(), fonte


# ---------------------------------------------------------------------------
# Chunking de um único registro
# ---------------------------------------------------------------------------

def chunkear_registro(
    registro: dict,
    splitter: RecursiveCharacterTextSplitter,
    incluir_sem_ementa: bool = False,
) -> list[dict]:
    """
    Gera os chunks de um único registro.

    Retorna lista de dicts (um por chunk).
    Retorna lista vazia se o registro for descartado.
    """
    tem_ementa = bool(registro.get("ementa"))

    if not tem_ementa and not incluir_sem_ementa:
        return []  # descarta registros sem ementa (padrão)

    texto, fonte = montar_texto(registro)

    if not texto.strip():
        return []

    # Divide em chunks
    pedacos = splitter.split_text(texto)

    if not pedacos:
        return []

    # URL do PDF principal (texto_integral, se existir)
    url_pdf = ""
    for pdf in registro.get("pdfs") or []:
        if pdf.get("categoria") == "texto_integral":
            url_pdf = pdf.get("url", "").replace("http://", "https://")
            break

    doc_id = registro.get("id", "")

    chunks = []
    for idx, pedaco in enumerate(pedacos):
        chunks.append({
            "chunk_id":        f"{doc_id}_{idx}",
            "doc_id":          doc_id,
            "tipo_codigo":     registro.get("tipo_codigo", ""),
            "tipo_nome":       registro.get("tipo_nome", ""),
            "numero":          registro.get("numero", ""),
            "ano":             registro.get("ano", ""),
            "autor":           registro.get("autor", ""),
            "assunto":         registro.get("assunto", ""),
            "data_assinatura": registro.get("data_assinatura", ""),
            "data_publicacao": registro.get("data_publicacao", ""),
            "url_pdf":         url_pdf,
            "fonte_texto":     fonte,
            "chunk_index":     idx,
            "chunk_total":     len(pedacos),
            "texto":           pedaco.strip(),
        })

    return chunks


# ---------------------------------------------------------------------------
# Pipeline principal
# ---------------------------------------------------------------------------

def gerar_chunks(
    registros: list,
    anos: list,
    incluir_sem_ementa: bool = False,
    limite: int = None,
) -> list[dict]:
    """
    Processa todos os registros e retorna lista de chunks.
    """
    # Filtra por ano
    if anos:
        registros = [r for r in registros if r.get("ano_fonte") in anos]
        log.info(f"Filtro por ano {anos}: {len(registros)} registros")

    # Aplica limite (modo teste)
    if limite:
        registros = registros[:limite]
        log.info(f"Modo teste: {limite} registros")

    # Inicializa o splitter
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE_CHARS,
        chunk_overlap=CHUNK_OVERLAP_CHARS,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
    )

    todos_chunks = []
    descartados  = 0
    sem_ementa   = 0

    for reg in registros:
        if not reg.get("ementa"):
            sem_ementa += 1

        chunks = chunkear_registro(reg, splitter, incluir_sem_ementa)

        if not chunks:
            descartados += 1
        else:
            todos_chunks.extend(chunks)

    log.info("=" * 55)
    log.info(f"Registros processados : {len(registros)}")
    log.info(f"Sem ementa            : {sem_ementa}")
    log.info(f"Descartados           : {descartados}")
    log.info(f"Chunks gerados        : {len(todos_chunks)}")
    if todos_chunks:
        docs_unicos = len({c["doc_id"] for c in todos_chunks})
        media_chunks = len(todos_chunks) / docs_unicos
        log.info(f"Documentos únicos     : {docs_unicos}")
        log.info(f"Média chunks/doc      : {media_chunks:.1f}")
    log.info("=" * 55)

    return todos_chunks


# ---------------------------------------------------------------------------
# Salva em Parquet
# ---------------------------------------------------------------------------

def salvar_parquet(chunks: list[dict], caminho: Path) -> None:
    """
    Salva os chunks em Parquet (formato de entrada da Fase 2).

    Colunas com tipo garantido:
      chunk_index, chunk_total → int32
      todas as strings         → string (não object)
    """
    caminho.parent.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(chunks)

    # Garante tipos corretos
    for col in ["chunk_index", "chunk_total"]:
        df[col] = df[col].astype("int32")

    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].astype("string")

    df.to_parquet(caminho, index=False, engine="pyarrow", compression="snappy")

    tamanho_mb = caminho.stat().st_size / (1024 * 1024)
    log.info(f"Parquet salvo : {caminho}")
    log.info(f"Linhas        : {len(df)}")
    log.info(f"Tamanho       : {tamanho_mb:.1f} MB")
    log.info(f"Colunas       : {list(df.columns)}")


# ---------------------------------------------------------------------------
# Preview
# ---------------------------------------------------------------------------

def mostrar_preview(chunks: list[dict], n: int = 3) -> None:
    """Mostra os primeiros N chunks para inspeção visual."""
    log.info(f"\nPREVIEW — primeiros {n} chunks:")
    for chunk in chunks[:n]:
        log.info("-" * 50)
        log.info(f"  chunk_id   : {chunk['chunk_id']}")
        log.info(f"  tipo       : {chunk['tipo_nome']} nº {chunk['numero']}/{chunk['ano']}")
        log.info(f"  fonte      : {chunk['fonte_texto']}")
        log.info(f"  chunk      : {chunk['chunk_index']+1}/{chunk['chunk_total']}")
        log.info(f"  chars      : {len(chunk['texto'])}")
        log.info(f"  texto      : {chunk['texto'][:200].replace(chr(10), ' ')}...")
    log.info("-" * 50)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Gera chunks do JSON ANEEL (ementas) → Parquet"
    )
    parser.add_argument("--json",   default=str(ARQUIVO_JSON), help="Caminho do JSON")
    parser.add_argument("--saida",  default=str(PASTA_SAIDA),  help="Pasta de saída")
    parser.add_argument("--ano",    nargs="+", default=[],      help="Filtrar por ano (ex: 2016)")
    parser.add_argument("--limite", type=int,  default=None,    help="Só N registros (teste)")
    parser.add_argument(
        "--incluir-sem-ementa", action="store_true",
        help="Inclui registros sem ementa (só metadados)"
    )
    args = parser.parse_args()

    # Carrega JSON
    json_path = Path(args.json)
    if not json_path.exists():
        log.error(f"JSON não encontrado: {json_path}")
        return

    with open(json_path, encoding="utf-8") as f:
        registros = json.load(f)
    log.info(f"JSON carregado: {len(registros)} registros")

    # Gera chunks
    chunks = gerar_chunks(
        registros=registros,
        anos=args.ano,
        incluir_sem_ementa=args.incluir_sem_ementa,
        limite=args.limite,
    )

    if not chunks:
        log.warning("Nenhum chunk gerado. Verifique os filtros.")
        return

    # Preview
    mostrar_preview(chunks)

    # Define nome do arquivo de saída
    anos_str = "_".join(sorted(args.ano)) if args.ano else "todos"
    nome_arquivo = f"chunks_json_{anos_str}.parquet"
    caminho_saida = Path(args.saida) / nome_arquivo

    # Salva
    salvar_parquet(chunks, caminho_saida)

    log.info("\nPróximo passo: rodar a Fase 2 (vector_db.py) apontando para esse Parquet.")


if __name__ == "__main__":
    main()