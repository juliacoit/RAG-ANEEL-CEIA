"""
parser.py
=========
Pessoa 1 — Data Engineer

Parser de alta performance para arquivos da ANEEL.
Configurado para usar ~2GB de RAM com máximo paralelismo.

Estratégia de memória (~2GB):
  - 12 workers ProcessPool = 12 PDFs simultâneos (~1.8GB)
  - Buffer de 5000 chunks em RAM antes de gravar (~10MB)
  - Cache de 30k arquivos já processados em memória (~3MB)
  - Overhead libs (PyMuPDF, pdfplumber, PyArrow): ~300MB
  - TOTAL: ~2.1GB

Para cada PDF, extrai:
  - Texto completo com posição de página
  - Texto tachado: marcado como [TACHADO: ...]
  - Imagens: marcadas como [IMAGEM]
  - Tabelas: convertidas para Markdown com pdfplumber
  - Cabeçalhos/rodapés: removidos automaticamente
  - Caracteres especiais preservados (§, º, ≤, ≥, etc.)

Pré-requisitos:
  pip install pymupdf pdfplumber pyarrow pandas langchain-text-splitters tqdm

Uso:
  # Processar tudo (~2GB RAM, recomendado):
  python src/p1_ingestion/parser.py

  # Testar com 100 arquivos:
  python src/p1_ingestion/parser.py --limite 100

  # Só um ano e categoria:
  python src/p1_ingestion/parser.py --ano 2022 --categoria texto_integral

  # Ajustar workers manualmente:
  python src/p1_ingestion/parser.py --workers 8
"""

import re
import json
import logging
import argparse
import unicodedata
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, asdict

import pyarrow as pa
import pyarrow.parquet as pq
import pandas as pd

try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:
    from langchain.text_splitter import RecursiveCharacterTextSplitter

try:
    from tqdm import tqdm
    USA_TQDM = True
except ImportError:
    USA_TQDM = False

# ---------------------------------------------------------------------------
# Configurações — calibradas para ~2GB de RAM
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

RAIZ       = Path(__file__).resolve().parent.parent.parent
PASTA_PDFS = RAIZ / "pdfs"
PASTA_DATA = RAIZ / "data" / "processed"
JSON_LIMPO = RAIZ / "data" / "aneel_vigentes_completo.json"
PARQUET_OUT= PASTA_DATA / "chunks_pdf_completo.parquet"

# Chunking
CHUNK_SIZE    = 800
CHUNK_OVERLAP = 120

# Performance — ~2GB RAM
WORKERS       = 12     # PDFs processados em paralelo
BATCH_WRITE   = 5000   # chunks em RAM antes de gravar no Parquet
MIN_CHARS_PAG = 50     # abaixo = PDF escaneado

CATEGORIAS = ["texto_integral", "voto", "nota_tecnica", "decisao", "anexo", "outro"]

# ---------------------------------------------------------------------------
# Schema Parquet
# ---------------------------------------------------------------------------

SCHEMA = pa.schema([
    pa.field("chunk_id",        pa.string()),
    pa.field("doc_id",          pa.string()),
    pa.field("arquivo",         pa.string()),
    pa.field("categoria_pdf",   pa.string()),
    pa.field("tipo_codigo",     pa.string()),
    pa.field("tipo_nome",       pa.string()),
    pa.field("numero",          pa.string()),
    pa.field("ano",             pa.string()),
    pa.field("ano_fonte",       pa.string()),
    pa.field("autor",           pa.string()),
    pa.field("assunto",         pa.string()),
    pa.field("data_assinatura", pa.string()),
    pa.field("data_publicacao", pa.string()),
    pa.field("url_pdf",         pa.string()),
    pa.field("fonte_texto",     pa.string()),
    pa.field("pagina_inicio",   pa.int32()),
    pa.field("pagina_fim",      pa.int32()),
    pa.field("chunk_index",     pa.int32()),
    pa.field("chunk_total",     pa.int32()),
    pa.field("tem_tabela",      pa.bool_()),
    pa.field("tem_imagem",      pa.bool_()),
    pa.field("tem_tachado",     pa.bool_()),
    pa.field("qualidade",       pa.string()),
    pa.field("texto",           pa.string()),
])

DEFAULTS = {
    "chunk_id": "", "doc_id": "", "arquivo": "", "categoria_pdf": "",
    "tipo_codigo": "", "tipo_nome": "", "numero": "", "ano": "",
    "ano_fonte": "", "autor": "", "assunto": "", "data_assinatura": "",
    "data_publicacao": "", "url_pdf": "", "fonte_texto": "pdf_completo",
    "pagina_inicio": 0, "pagina_fim": 0, "chunk_index": 0, "chunk_total": 0,
    "tem_tabela": False, "tem_imagem": False, "tem_tachado": False,
    "qualidade": "ok", "texto": "",
}


# ---------------------------------------------------------------------------
# Funções de extração (rodam dentro dos workers)
# ---------------------------------------------------------------------------

# Regex compilados uma vez — mais performático com 27k arquivos
_RE_HIFENIZACAO    = re.compile(r"-\s*\n\s*")
_RE_MULTIPLOS_NL   = re.compile(r"\n{3,}")
_RE_ESPACOS_DUPLOS = re.compile(r"[ \t]{2,}")
_RE_PAGINA_NUM     = re.compile(r"(?m)^\s*(?:Página|página|Pg\.?|pg\.?|p\.)?\s*\d{1,4}\s*$")
_RE_LINHA_PONTUA   = re.compile(r"(?m)^\s*[-–—=_*·•]{3,}\s*$")
_RE_CTRL           = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


def _limpar_texto(texto: str) -> str:
    """
    Limpeza completa preservando caracteres especiais jurídicos/técnicos.
    (§, º, ª, ≤, ≥, ×, ÷ — comuns em normas da ANEEL)
    """
    texto = unicodedata.normalize("NFC", texto)   # normaliza acentos de PDFs antigos
    texto = _RE_CTRL.sub("", texto)               # remove controle (exceto \n\t)
    texto = _RE_HIFENIZACAO.sub("", texto)        # "condi-\nções" → "condições"
    texto = _RE_PAGINA_NUM.sub("", texto)         # remove números de página isolados
    texto = _RE_LINHA_PONTUA.sub("", texto)       # remove linhas só com traços/bullets
    texto = _RE_ESPACOS_DUPLOS.sub(" ", texto)    # colapsa espaços múltiplos
    texto = _RE_MULTIPLOS_NL.sub("\n\n", texto)   # máx 2 quebras seguidas
    return texto.strip()


def _detectar_cabecalho_rodape(paginas: list[str]) -> set[str]:
    """
    Linhas que aparecem em 80%+ das páginas = cabeçalho/rodapé.
    Inspeciona 5 linhas do topo e 5 do rodapé de cada página.
    """
    from collections import Counter
    n = len(paginas)
    if n < 3:
        return set()
    ctr = Counter()
    for pag in paginas:
        linhas = pag.split("\n")
        candidatas = set(
            l.strip() for l in linhas[:5] + linhas[-5:]
            if len(l.strip()) > 5
        )
        for l in candidatas:
            ctr[l] += 1
    thr = max(3, int(n * 0.8))
    return {l for l, c in ctr.items() if c >= thr}


def _extrair_tabelas(pagina_plumber) -> list[str]:
    """Extrai tabelas em Markdown com pdfplumber."""
    tabelas = []
    try:
        for tbl in pagina_plumber.extract_tables():
            if not tbl:
                continue
            linhas = []
            for i, row in enumerate(tbl):
                cells = [str(c or "").strip() for c in row]
                linhas.append("| " + " | ".join(cells) + " |")
                if i == 0:
                    linhas.append("|" + "|".join(["---"] * len(cells)) + "|")
            if linhas:
                tabelas.append("\n".join(linhas))
    except Exception:
        pass
    return tabelas


def _extrair_pagina_fitz(pagina, fitz_mod=None) -> tuple[str, bool, bool]:
    """
    Extrai texto de uma página com PyMuPDF.
    Retorna (texto, tem_tachado, tem_imagem).

    Ordem de tentativas:
      1. get_text("dict", flags=~0) — texto rico: detecta tachado e imagens
      2. get_text("text") simples  — fallback para PDFs com estrutura malformada
         (PDFs da ANEEL com warning "No common ancestor" caem aqui)
      3. Retorna vazio → OCR em _processar_pdf

    fitz_mod: módulo fitz já importado (evita re-import em multiprocessing Windows)
    """
    blocos  = []
    tachado = False
    imagem  = False

    # Verifica imagens
    try:
        if pagina.get_images():
            imagem = True
    except Exception:
        pass

    # ── Tentativa 1: modo dict — texto rico com tachado ───────────────────
    chars_dict = 0
    try:
        dic = pagina.get_text("dict", flags=~0)
        for bloco in dic.get("blocks", []):
            if bloco.get("type") == 1:
                imagem = True
                blocos.append("[IMAGEM]")
                continue
            for linha in bloco.get("lines", []):
                linha_txt = []
                for span in linha.get("spans", []):
                    t = span.get("text", "")
                    if not t.strip():
                        continue
                    flags = span.get("flags", 0)
                    eh_tachado = bool(flags & 8)
                    if not eh_tachado:
                        cor = span.get("color", 0)
                        if isinstance(cor, int) and cor > 8_000_000:
                            eh_tachado = True
                    if eh_tachado:
                        tachado = True
                        linha_txt.append(f"[TACHADO: {t}]")
                    else:
                        linha_txt.append(t)
                if linha_txt:
                    blocos.append("".join(linha_txt))
        chars_dict = sum(len(b) for b in blocos if b != "[IMAGEM]")
    except Exception:
        pass

    # ── Tentativa 2: get_text simples — fallback robusto ──────────────────
    if chars_dict < 50:
        try:
            # Usa flags numéricas para evitar dependência do módulo fitz
            # TEXT_PRESERVE_READING_ORDER = 4, TEXT_DEHYPHENATE = 8
            texto_simples = pagina.get_text("text", flags=4 | 8).strip()
            if len(texto_simples) > chars_dict:
                imagens_marcadas = [b for b in blocos if b == "[IMAGEM]"]
                blocos = imagens_marcadas + ([texto_simples] if texto_simples else [])
        except Exception:
            pass

    return "\n".join(blocos), tachado, imagem


def _processar_pdf(caminho: Path) -> dict:
    """
    Processa um PDF completo.
    Retorna dict com texto_por_pagina, tem_tachado, tem_imagem, qualidade.
    """
    import fitz
    res = {
        "texto_por_pagina": [],
        "tem_tachado": False,
        "tem_imagem":  False,
        "qualidade":   "ok",
        "n_paginas":   0,
    }

    try:
        doc = fitz.open(str(caminho))
        res["n_paginas"] = len(doc)

        # Tenta abrir pdfplumber para tabelas
        doc_pl = None
        try:
            import pdfplumber
            doc_pl = pdfplumber.open(str(caminho))
        except Exception:
            pass

        paginas_raw = []
        chars_total = 0

        for i, pag in enumerate(doc):
            txt, tach, img = _extrair_pagina_fitz(pag, fitz)
            if tach: res["tem_tachado"] = True
            if img:  res["tem_imagem"]  = True

            # Adiciona tabelas da mesma página
            if doc_pl and i < len(doc_pl.pages):
                try:
                    tbls = _extrair_tabelas(doc_pl.pages[i])
                    if tbls:
                        txt += "\n\n" + "\n\n".join(tbls)
                except Exception:
                    pass

            txt_limpo = _limpar_texto(txt)
            paginas_raw.append(txt_limpo)
            chars_total += len(txt_limpo)

        doc.close()
        if doc_pl:
            doc_pl.close()

        # Qualidade
        cpp = chars_total / max(len(paginas_raw), 1)
        if cpp < MIN_CHARS_PAG:   res["qualidade"] = "imagem"
        elif cpp < 200:           res["qualidade"] = "baixa"

        # Remove cabeçalho/rodapé
        cab = _detectar_cabecalho_rodape(paginas_raw)
        paginas = []
        for pag in paginas_raw:
            linhas = [l for l in pag.split("\n") if l.strip() not in cab]
            paginas.append("\n".join(linhas))

        res["texto_por_pagina"] = paginas

    except Exception as e:
        res["qualidade"] = "erro"

    return res


# ---------------------------------------------------------------------------
# Worker (roda em processo separado)
# ---------------------------------------------------------------------------

def worker(args: tuple) -> list[dict]:
    """
    Processa um arquivo e retorna lista de dicts de chunks.
    Cada processo tem seu próprio splitter e imports.
    """
    caminho_str, meta, chunk_size, chunk_overlap = args
    caminho = Path(caminho_str)
    ext = caminho.suffix.lower()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    out = []

    # ── PDF ──────────────────────────────────────────────────────────────
    if ext == ".pdf":
        extr = _processar_pdf(caminho)

        if extr["qualidade"] in ("imagem", "erro"):
            c = {**DEFAULTS, **meta,
                 "chunk_id":    f"{meta.get('doc_id','')}_{caminho.name}_img",
                 "arquivo":     caminho.name,
                 "fonte_texto": "pdf_imagem",
                 "qualidade":   extr["qualidade"],
                 "tem_imagem":  True,
                 "chunk_total": 1,
                 "texto":       f"[PDF ESCANEADO: {caminho.name}]"}
            return [c]

        # Monta texto com marcadores de página para rastreamento
        blocos = []
        for i, pag in enumerate(extr["texto_por_pagina"]):
            if pag.strip():
                blocos.append(f"[Página {i+1}]\n{pag}")
        texto_completo = "\n\n".join(blocos)
        if not texto_completo.strip():
            return out

        pedacos = splitter.split_text(texto_completo)
        total   = len(pedacos)

        for idx, pedaco in enumerate(pedacos):
            pags = re.findall(r"\[Página (\d+)\]", pedaco)
            p_ini = int(pags[0])  if pags else 1
            p_fim = int(pags[-1]) if pags else 1
            txt   = re.sub(r"\[Página \d+\]\n?", "", pedaco).strip()
            if not txt:
                continue

            c = {**DEFAULTS, **meta,
                 "chunk_id":      f"{meta.get('doc_id','')}_{caminho.name}_{idx}",
                 "arquivo":       caminho.name,
                 "fonte_texto":   "pdf_completo",
                 "qualidade":     extr["qualidade"],
                 "tem_tachado":   extr["tem_tachado"],
                 "tem_imagem":    extr["tem_imagem"],
                 "tem_tabela":    "|" in txt,
                 "pagina_inicio": p_ini,
                 "pagina_fim":    p_fim,
                 "chunk_index":   idx,
                 "chunk_total":   total,
                 "texto":         txt}
            out.append(c)

    # ── TXT (HTML ou XLSX já extraídos) ──────────────────────────────────
    elif ext == ".txt":
        try:
            texto = _limpar_texto(
                caminho.read_text(encoding="utf-8", errors="replace")
            )
            if not texto.strip():
                return out
            fonte = "html" if "html_ren" in str(caminho) else "xlsx"
            pedacos = splitter.split_text(texto)
            total   = len(pedacos)
            for idx, pedaco in enumerate(pedacos):
                if not pedaco.strip():
                    continue
                c = {**DEFAULTS, **meta,
                     "chunk_id":    f"{meta.get('doc_id','')}_{caminho.name}_{idx}",
                     "arquivo":     caminho.name,
                     "fonte_texto": fonte,
                     "qualidade":   "ok",
                     "tem_tabela":  "|" in pedaco,
                     "chunk_index": idx,
                     "chunk_total": total,
                     "texto":       pedaco.strip()}
                out.append(c)
        except Exception:
            pass

    return out


# ---------------------------------------------------------------------------
# Escrita incremental no Parquet
# ---------------------------------------------------------------------------

class ParquetWriter:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self._w     = pq.ParquetWriter(str(path), SCHEMA, compression="snappy")
        self._total = 0

    def write(self, chunks: list[dict]) -> None:
        if not chunks:
            return
        df = pd.DataFrame(chunks)
        for col in SCHEMA.names:
            if col not in df.columns:
                dtype = SCHEMA.field(col).type
                df[col] = False if dtype == pa.bool_() else (0 if dtype == pa.int32() else "")
        # Cast explícito para int32 nas colunas inteiras
        for col in ["pagina_inicio","pagina_fim","chunk_index","chunk_total"]:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype("int32")
        # Corrige colunas booleanas — string vazia vira False
        for col in ["tem_tabela","tem_imagem","tem_tachado"]:
            if col in df.columns:
                df[col] = df[col].apply(lambda x: bool(x) if not isinstance(x, str) else x.lower() not in ("", "false", "0"))
        tbl = pa.Table.from_pandas(df[SCHEMA.names], schema=SCHEMA)
        self._w.write_table(tbl)
        self._total += len(chunks)

    def close(self) -> int:
        self._w.close()
        return self._total


# ---------------------------------------------------------------------------
# Coleta de tarefas
# ---------------------------------------------------------------------------

def coletar_tarefas(
    json_path: Path,
    categorias: list[str],
    anos: list[str],
    limite: int = None,
) -> list[tuple[str, dict]]:
    """Retorna lista de (caminho_str, meta_dict)."""
    with open(json_path, encoding="utf-8") as f:
        registros = json.load(f)

    tarefas   = []
    vistos    = set()   # evita duplicatas — cache em memória ~3MB para 30k itens

    def _meta(reg: dict, cat: str, url: str) -> dict:
        return {
            "doc_id":          reg.get("id", ""),
            "categoria_pdf":   cat,
            "url_pdf":         url,
            "tipo_codigo":     reg.get("tipo_codigo", ""),
            "tipo_nome":       reg.get("tipo_nome", ""),
            "numero":          reg.get("numero") or "",
            "ano":             reg.get("ano") or "",
            "ano_fonte":       reg.get("ano_fonte", ""),
            "autor":           reg.get("autor") or "",
            "assunto":         reg.get("assunto") or "",
            "data_assinatura": reg.get("data_assinatura") or "",
            "data_publicacao": reg.get("data_publicacao") or "",
        }

    for reg in registros:
        ano = reg.get("ano_fonte", "")
        if anos and ano not in anos:
            continue

        for pdf in reg.get("pdfs") or []:
            cat     = pdf.get("categoria", "")
            arquivo = (pdf.get("arquivo") or "").strip()
            url     = pdf.get("url") or ""

            if not arquivo or (categorias and cat not in categorias):
                continue

            ext = Path(arquivo).suffix.lower().rstrip()

            if ext in (".pdf",):
                caminho = PASTA_PDFS / ano / cat / arquivo
            elif ext in (".htm", ".html"):
                # Usa .txt já extraído
                caminho = PASTA_PDFS / ano / "html_ren" / \
                          arquivo.replace(ext, ".txt")
            elif ext in (".xlsx", ".xlsm"):
                caminho = PASTA_PDFS / ano / "xlsx" / \
                          arquivo.replace(ext, ".txt")
            elif ext == ".zip":
                # PDFs dentro do ZIP extraído
                pasta = PASTA_PDFS / ano / "zip_extraido" / arquivo.replace(".zip","")
                if pasta.exists():
                    for pdf_int in pasta.rglob("*.pdf"):
                        k = str(pdf_int)
                        if k not in vistos:
                            vistos.add(k)
                            tarefas.append((k, _meta(reg, cat, url)))
                continue
            else:
                continue

            if not caminho.exists():
                continue

            k = str(caminho)
            if k not in vistos:
                vistos.add(k)
                tarefas.append((k, _meta(reg, cat, url)))

    # TXTs de html_ren que não estão no JSON
    for ano in (anos or ["2015","2016","2021","2022"]):
        pasta = PASTA_PDFS / ano / "html_ren"
        if pasta.exists():
            for txt in pasta.rglob("*.txt"):
                k = str(txt)
                if k not in vistos:
                    vistos.add(k)
                    meta = {k2: "" for k2 in DEFAULTS}
                    meta.update({"ano_fonte": ano, "categoria_pdf": "html_ren",
                                 "doc_id": txt.stem})
                    tarefas.append((k, meta))

    return tarefas[:limite] if limite else tarefas


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Parser ANEEL — extração de texto com ~2GB de RAM"
    )
    ap.add_argument("--json",      default=str(JSON_LIMPO))
    ap.add_argument("--saida",     default=str(PARQUET_OUT))
    ap.add_argument("--categoria", nargs="+", default=CATEGORIAS,
                    choices=CATEGORIAS + ["html_ren"])
    ap.add_argument("--ano",       nargs="+", default=[],
                    choices=["2015","2016","2021","2022"])
    ap.add_argument("--limite",    type=int, default=None)
    ap.add_argument("--workers",   type=int, default=WORKERS)
    ap.add_argument("--chunk-size",    type=int, default=CHUNK_SIZE)
    ap.add_argument("--chunk-overlap", type=int, default=CHUNK_OVERLAP)
    args = ap.parse_args()

    log.info("=" * 55)
    log.info("PARSER ANEEL — modo ~2GB RAM")
    log.info("=" * 55)

    tarefas = coletar_tarefas(
        Path(args.json), args.categoria, args.ano, args.limite
    )
    log.info(f"  Arquivos a processar: {len(tarefas)}")

    if not tarefas:
        log.error("Nenhum arquivo encontrado.")
        return

    tempo_est = len(tarefas) / args.workers / 3
    h = int(tempo_est // 3600)
    m = int((tempo_est % 3600) // 60)
    log.info(f"  Workers:              {args.workers}")
    log.info(f"  RAM estimada:         ~{args.workers * 150 + 500}MB")
    log.info(f"  Tempo estimado:       ~{h}h {m}min")
    log.info(f"  Saída:                {args.saida}")
    log.info("=" * 55)

    worker_args = [
        (cam, meta, args.chunk_size, args.chunk_overlap)
        for cam, meta in tarefas
    ]

    writer  = ParquetWriter(Path(args.saida))
    buffer  = []
    erros   = 0
    imagens = 0

    barra = tqdm(total=len(tarefas), desc="Parsing", unit="arq",
                 dynamic_ncols=True) if USA_TQDM else None

    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futuros = {ex.submit(worker, a): a for a in worker_args}

        for fut in as_completed(futuros):
            try:
                chunks = fut.result()
                imagens += sum(1 for c in chunks if c.get("qualidade") == "imagem")
                buffer.extend(chunks)

                # Grava quando buffer chega no limite — libera RAM
                if len(buffer) >= BATCH_WRITE:
                    writer.write(buffer)
                    buffer = []

            except Exception as e:
                erros += 1
                log.debug(f"Worker error: {e}")

            if barra:
                barra.set_postfix({
                    "buf": len(buffer),
                    "err": erros,
                    "img": imagens,
                })
                barra.update(1)

    # Grava restante
    if buffer:
        writer.write(buffer)

    total = writer.close()
    if barra:
        barra.close()

    log.info("\n" + "=" * 55)
    log.info("CONCLUÍDO")
    log.info("=" * 55)
    log.info(f"  Chunks gerados  : {total:,}")
    log.info(f"  PDFs escaneados : {imagens}")
    log.info(f"  Erros           : {erros}")
    log.info(f"  Parquet salvo   : {args.saida}")
    log.info("=" * 55)
    log.info("\nProximo passo: reindexar no Qdrant com o novo parquet")
    log.info("  python src/p2_search/p2_indexar.py --parquet data/processed/chunks_pdf_completo.parquet --resetar")


if __name__ == "__main__":
    main()