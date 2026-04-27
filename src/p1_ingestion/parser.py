"""
parser.py
=========
Pessoa 1 — Data Engineer

Parser de alta performance para arquivos da ANEEL.

Melhorias implementadas:
  ✓ BUG CORRIGIDO: regex _RE_ATO_NORMATIVO com escape correto
  ✓ Chunking semântico jurídico (Art., §, CAPÍTULO, incisos)
  ✓ Extração e salvamento de imagens em disco com IDs únicos
  ✓ Referências textuais entre atos normativos (refs_texto)
  ✓ Filtro de chunks inúteis (XLSX linhas vazias, densidade baixa)
  ✓ Deduplicação de chunks idênticos dentro do documento
  ✓ Checkpoint de progresso — retoma onde parou se interrompido
  ✓ Metadados de qualidade por chunk (n_tokens, densidade)
  ✓ Extração de valores numéricos e percentuais (valores_num)
  ✓ Extração de datas mencionadas no corpo do texto
  ✓ Filtro de assinaturas (imagens muito pequenas/estreitas)

Uso:
  python src/p1_ingestion/parser.py             # tudo
  python src/p1_ingestion/parser.py --limite 50 # teste rápido
  python src/p1_ingestion/parser.py --ano 2022  # só 2022
  python src/p1_ingestion/parser.py --workers 8 # menos RAM
  python src/p1_ingestion/parser.py --sem-checkpoint  # ignora checkpoint
"""

import re
import json
import logging
import argparse
import unicodedata
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

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
# Configurações
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

RAIZ        = Path(__file__).resolve().parent.parent.parent
PASTA_PDFS  = RAIZ / "pdfs"
PASTA_DATA  = RAIZ / "data" / "processed"
JSON_LIMPO  = RAIZ / "data" / "aneel_vigentes_completo.json"
PARQUET_OUT = PASTA_DATA / "chunks_pdf_completo.parquet"
CHECKPOINT  = PASTA_DATA / "parser_checkpoint.json"

# Chunking semântico
CHUNK_SIZE    = 800
CHUNK_OVERLAP = 120

# Performance
WORKERS       = 50
BATCH_WRITE   = 5000
MIN_CHARS_PAG = 50    # abaixo = PDF sem texto extraível
MIN_CHUNK_DENSIDADE = 0.3   # chunks com menos de 30% de chars úteis são descartados
MIN_CHUNK_CHARS     = 30    # chunks com menos de 30 chars são descartados

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
    pa.field("n_tokens",        pa.int32()),   # estimativa de tokens
    pa.field("densidade",       pa.float32()), # proporção de chars úteis
    pa.field("texto",           pa.string()),

    pa.field("refs_texto",      pa.string()),  # JSON: ["REN_482_2012", ...]
    pa.field("n_refs",          pa.int32()),
    pa.field("valores_num",     pa.string()),  # JSON: ["50%", "R$ 1.200,00"]
    pa.field("datas_texto",     pa.string()),  # JSON: datas mencionadas no corpo
])

DEFAULTS = {
    "chunk_id": "", "doc_id": "", "arquivo": "", "categoria_pdf": "",
    "tipo_codigo": "", "tipo_nome": "", "numero": "", "ano": "",
    "ano_fonte": "", "autor": "", "assunto": "", "data_assinatura": "",
    "data_publicacao": "", "url_pdf": "", "fonte_texto": "pdf_completo",
    "pagina_inicio": 0, "pagina_fim": 0, "chunk_index": 0, "chunk_total": 0,
    "tem_tabela": False, "tem_imagem": False, "tem_tachado": False,
    "qualidade": "ok", "n_tokens": 0, "densidade": 0.0, "texto": "",
    "refs_texto": "[]", "n_refs": 0,
    "valores_num": "[]", "datas_texto": "[]",
}


# ---------------------------------------------------------------------------
# Regex compilados (nível de módulo — compartilhados entre workers)
# ---------------------------------------------------------------------------

_RE_HIFENIZACAO    = re.compile(r"-\s*\n\s*")
_RE_MULTIPLOS_NL   = re.compile(r"\n{3,}")
_RE_ESPACOS_DUPLOS = re.compile(r"[ \t]{2,}")
_RE_PAGINA_NUM     = re.compile(r"(?m)^\s*(?:Página|página|Pg\.?|pg\.?|p\.)?\s*\d{1,4}\s*$")
_RE_LINHA_PONTUA   = re.compile(r"(?m)^\s*[-–—=_*·•]{3,}\s*$")
_RE_CTRL           = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")

# ✓ BUG CORRIGIDO: escape simples \b, não \\b
_RE_ATO_NORMATIVO = re.compile(
    r"\b(?:Resolu[cç][aã]o\s+Normativa|Resolu[cç][aã]o\s+Autorizativa|"
    r"Resolu[cç][aã]o\s+Homologatória|Despacho|Portaria|"
    r"Instru[cç][aã]o\s+Normativa|Decreto)\s+"
    r"(?:n[oº°]?\.?\s*)?(\d{1,5})[/,\s]+(?:de\s+)?(\d{4})",
    re.IGNORECASE,
)

_TIPO_PREFIXO = {
    "resolução normativa": "REN", "resolução autorizativa": "REA",
    "resolução homologatória": "REH", "despacho": "DSP",
    "portaria": "PRT", "instrução normativa": "INA", "decreto": "DEC",
}

# Valores numéricos: percentuais e monetários
_RE_VALORES = re.compile(
    r"(?:"
    r"\d{1,3}(?:\.\d{3})*(?:,\d{1,4})?\s*%"        # percentual: 50%, 1.200,50%
    r"|R\$\s*\d{1,3}(?:\.\d{3})*(?:,\d{2})?"        # monetário: R$ 1.200,00
    r"|\d{1,3}(?:\.\d{3})*(?:,\d{2})?\s*R\$/MWh"    # tarifa: 120,50 R$/MWh
    r"|\d+(?:,\d+)?\s*(?:MW|kW|GW|MWh|kWh)"         # energia: 100 MW
    r")",
    re.IGNORECASE,
)

# Datas mencionadas no corpo do texto
_RE_DATA = re.compile(
    r"\b(?:\d{1,2}/\d{1,2}/\d{4}"           # 01/01/2022
    r"|\d{1,2}\s+de\s+\w+\s+de\s+\d{4}"     # 1 de janeiro de 2022
    r"|\d{4}-\d{2}-\d{2})\b",               # 2022-01-01
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Funções auxiliares
# ---------------------------------------------------------------------------

def _limpar_texto(texto: str) -> str:
    texto = unicodedata.normalize("NFC", texto)
    texto = _RE_CTRL.sub("", texto)
    texto = _RE_HIFENIZACAO.sub("", texto)
    texto = _RE_PAGINA_NUM.sub("", texto)
    texto = _RE_LINHA_PONTUA.sub("", texto)
    texto = _RE_ESPACOS_DUPLOS.sub(" ", texto)
    texto = _RE_MULTIPLOS_NL.sub("\n\n", texto)
    return texto.strip()


def _calcular_densidade(texto: str) -> float:
    """Proporção de chars não-espaço sobre total. Chunks muito esparsos = ruído."""
    if not texto:
        return 0.0
    util = sum(1 for c in texto if not c.isspace())
    return round(util / len(texto), 3)


def _estimar_tokens(texto: str) -> int:
    """Estimativa rápida: ~4 chars por token para português."""
    return max(1, len(texto) // 4)


def _extrair_refs_texto(texto: str) -> list[str]:
    """Extrai doc_ids referenciados no texto via regex."""
    refs = []
    for m in _RE_ATO_NORMATIVO.finditer(texto):
        trecho = m.group(0).lower()
        numero = m.group(1)
        ano    = m.group(2)
        tipo   = next((v for k, v in _TIPO_PREFIXO.items() if k in trecho), None)
        if tipo:
            refs.append(f"{tipo}_{numero}_{ano}")
    return list(set(refs))


def _extrair_valores(texto: str) -> list[str]:
    """Extrai valores numéricos relevantes: percentuais, tarifas, potências."""
    return list(set(m.group(0).strip() for m in _RE_VALORES.finditer(texto)))[:20]


def _extrair_datas(texto: str) -> list[str]:
    """Extrai datas mencionadas no corpo do texto."""
    return list(set(m.group(0).strip() for m in _RE_DATA.finditer(texto)))[:10]


def _detectar_cabecalho_rodape(paginas: list[str]) -> set[str]:
    """Linhas que aparecem em 80%+ das páginas = cabeçalho/rodapé."""
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
        ctr.update(candidatas)
    return {l for l, c in ctr.items() if c / n >= 0.8}


def _chunk_util(txt: str) -> bool:
    """Retorna False se o chunk é inútil (muito curto, muito esparso ou vazio)."""
    if not txt or len(txt) < MIN_CHUNK_CHARS:
        return False
    if _calcular_densidade(txt) < MIN_CHUNK_DENSIDADE:
        return False
    return True


# ---------------------------------------------------------------------------
# Extração de texto com PyMuPDF
# ---------------------------------------------------------------------------

def _extrair_pagina_fitz(pagina) -> tuple[str, bool, bool]:
    """
    Extrai texto de uma página com PyMuPDF.
    Retorna (texto, tem_tachado, tem_imagem).

    Ordem de tentativas:
      1. get_text("dict", flags=~0) — rico: detecta tachado e imagens
      2. get_text("text") simples  — fallback para PDFs malformados
    """
    blocos  = []
    tachado = False
    imagem  = False

    try:
        if pagina.get_images():
            imagem = True
    except Exception:
        pass

    # Tentativa 1: modo dict — detecta tachado
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
                    flags     = span.get("flags", 0)
                    eh_tach   = bool(flags & 8)
                    if not eh_tach:
                        cor = span.get("color", 0)
                        if isinstance(cor, int) and cor > 8_000_000:
                            eh_tach = True
                    if eh_tach:
                        tachado = True
                        linha_txt.append(f"[TACHADO: {t}]")
                    else:
                        linha_txt.append(t)
                if linha_txt:
                    blocos.append("".join(linha_txt))
        chars_dict = sum(len(b) for b in blocos if b != "[IMAGEM]")
    except Exception:
        pass

    # Tentativa 2: get_text simples (fallback robusto)
    # TEXT_PRESERVE_READING_ORDER=4, TEXT_DEHYPHENATE=8
    if chars_dict < 50:
        try:
            texto_simples = pagina.get_text("text", flags=4 | 8).strip()
            if len(texto_simples) > chars_dict:
                imagens_marcadas = [b for b in blocos if b == "[IMAGEM]"]
                blocos = imagens_marcadas + ([texto_simples] if texto_simples else [])
        except Exception:
            pass

    return "\n".join(blocos), tachado, imagem


def _extrair_tabelas(pagina_plumber) -> list[str]:
    """Extrai tabelas em Markdown com pdfplumber."""
    tabelas = []
    try:
        for tbl in pagina_plumber.extract_tables():
            if not tbl:
                continue
            linhas = []
            for i, row in enumerate(tbl):
                cells = [str(c or "").strip().replace("\n", " ") for c in row]
                linhas.append("| " + " | ".join(cells) + " |")
                if i == 0:
                    linhas.append("|" + "|".join(["---"] * len(cells)) + "|")
            if len(linhas) > 2:  # pelo menos header + separator + 1 linha de dados
                tabelas.append("\n".join(linhas))
    except Exception:
        pass
    return tabelas


# ---------------------------------------------------------------------------
# Processamento de PDF
# ---------------------------------------------------------------------------

def _processar_pdf(caminho: Path) -> dict:
    """
    Processa um PDF completo.
    Retorna dict com texto_por_pagina, imagens_por_pagina, metadados.
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
        doc    = fitz.open(str(caminho))

        res["n_paginas"] = len(doc)

        doc_pl = None
        try:
            import pdfplumber
            doc_pl = pdfplumber.open(str(caminho))
        except Exception:
            pass

        paginas_raw = []
        chars_total = 0

        for i, pag in enumerate(doc):
            txt, tach, img = _extrair_pagina_fitz(pag)
            if tach: res["tem_tachado"] = True

            # Detecta se página tem imagens (sem salvar em disco)
            if img:
                res["tem_imagem"] = True

            # Tabelas
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

        cpp = chars_total / max(len(paginas_raw), 1)
        if cpp < MIN_CHARS_PAG:
            res["qualidade"] = "imagem"
        elif cpp < 200:
            res["qualidade"] = "baixa"


        cab     = _detectar_cabecalho_rodape(paginas_raw)
        paginas = []
        for pag in paginas_raw:
            linhas = [l for l in pag.split("\n") if l.strip() not in cab]
            paginas.append("\n".join(linhas))

        res["texto_por_pagina"] = paginas

    except Exception:
        res["qualidade"] = "erro"
    return res


# ---------------------------------------------------------------------------
# Worker (roda em processo separado)
# ---------------------------------------------------------------------------

def worker(args: tuple) -> list[dict]:
    caminho_str, meta, chunk_size, chunk_overlap = args
    caminho = Path(caminho_str)
    ext     = caminho.suffix.lower()

    # Chunking semântico — fronteiras jurídicas têm prioridade
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=[
            "\nCAPÍTULO ", "\nSEÇÃO ", "\nSECAO ",
            "\nArt. ", "\nArt.", "\nArtigo ",
            "\n§ ", "\nParágrafo único",
            "\nI - ", "\nII - ", "\nIII - ", "\nIV - ", "\nV - ",
            "\n\n", "\n", ". ", " ", "",
        ],
    )

    out = []

    # ── PDF ──────────────────────────────────────────────────────────────
    if ext == ".pdf":
        extr = _processar_pdf(caminho)

        if extr["qualidade"] in ("imagem", "erro"):
            return [{**DEFAULTS, **meta,
                     "chunk_id":    f"{meta.get('doc_id','')}_{caminho.name}_img",
                     "arquivo":     caminho.name,
                     "fonte_texto": "pdf_imagem",
                     "qualidade":   extr["qualidade"],
                     "tem_imagem":  True,
                     "chunk_total": 1,
                     "n_tokens":    5,
                     "densidade":   1.0,
                     "texto":       f"[PDF ESCANEADO: {caminho.name}]"}]

        blocos = []
        for i, pag in enumerate(extr["texto_por_pagina"]):
            if pag.strip():
                blocos.append(f"[Página {i+1}]\n{pag}")
        texto_completo = "\n\n".join(blocos)
        if not texto_completo.strip():
            return out

        import json as _json
        pedacos = splitter.split_text(texto_completo)
        total   = len(pedacos)

        for idx, pedaco in enumerate(pedacos):
            pags  = re.findall(r"\[Página (\d+)\]", pedaco)
            p_ini = int(pags[0])  if pags else 1
            p_fim = int(pags[-1]) if pags else 1
            txt   = re.sub(r"\[Página \d+\]\n?", "", pedaco).strip()

            refs   = _extrair_refs_texto(txt)
            vals   = _extrair_valores(txt)
            datas  = _extrair_datas(txt)
            dens   = _calcular_densidade(txt)
            ntok   = _estimar_tokens(txt)

            out.append({**DEFAULTS, **meta,
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
                "n_tokens":      ntok,
                "densidade":     dens,

                "refs_texto":    _json.dumps(refs, ensure_ascii=False),
                "n_refs":        len(refs),
                "valores_num":   _json.dumps(vals, ensure_ascii=False),
                "datas_texto":   _json.dumps(datas, ensure_ascii=False),
                "texto":         txt,
            })

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

            import json as _json
            for idx, pedaco in enumerate(pedacos):
                txt = pedaco.strip()

                if not txt:
                    continue

                refs  = _extrair_refs_texto(txt)
                vals  = _extrair_valores(txt)
                datas = _extrair_datas(txt)
                dens  = _calcular_densidade(txt)
                ntok  = _estimar_tokens(txt)

                out.append({**DEFAULTS, **meta,
                    "chunk_id":    f"{meta.get('doc_id','')}_{caminho.name}_{idx}",
                    "arquivo":     caminho.name,
                    "fonte_texto": fonte,
                    "qualidade":   "ok",
                    "tem_tabela":  "|" in txt,
                    "chunk_index": idx,
                    "chunk_total": total,
                    "n_tokens":    ntok,
                    "densidade":   dens,
                    "refs_texto":  _json.dumps(refs, ensure_ascii=False),
                    "n_refs":      len(refs),
                    "valores_num": _json.dumps(vals, ensure_ascii=False),
                    "datas_texto": _json.dumps(datas, ensure_ascii=False),
                    "texto":       txt,
                })
        except Exception:
            pass

    # ── Planilhas (CSV, XLSX) extraídas de ZIP/RAR ───────────────────────
    elif ext in (".csv", ".xlsx", ".xlsm"):
        try:
            if ext == ".csv":
                df = pd.read_csv(caminho, sep=None, engine="python", on_bad_lines="skip")
            else:
                df = pd.read_excel(caminho)

            linhas = []
            for _, row in df.iterrows():
                celulas = [str(c) for c in row if pd.notna(c) and str(c).strip()]
                if celulas:
                    linhas.append(" | ".join(celulas))

            texto = _limpar_texto("\n".join(linhas))
            if not texto.strip():
                return out

            pedacos = splitter.split_text(texto)
            total   = len(pedacos)

            import json as _json
            for idx, pedaco in enumerate(pedacos):
                txt = pedaco.strip()
                if not txt:
                    continue

                refs  = _extrair_refs_texto(txt)
                vals  = _extrair_valores(txt)
                datas = _extrair_datas(txt)
                dens  = _calcular_densidade(txt)
                ntok  = _estimar_tokens(txt)

                out.append({**DEFAULTS, **meta,
                    "chunk_id":    f"{meta.get('doc_id','')}_{caminho.name}_{idx}",
                    "arquivo":     caminho.name,
                    "fonte_texto": "planilha_zip",
                    "qualidade":   "ok",
                    "tem_tabela":  True,
                    "chunk_index": idx,
                    "chunk_total": total,
                    "n_tokens":    ntok,
                    "densidade":   dens,
                    "refs_texto":  _json.dumps(refs, ensure_ascii=False),
                    "n_refs":      len(refs),
                    "valores_num": _json.dumps(vals, ensure_ascii=False),
                    "datas_texto": _json.dumps(datas, ensure_ascii=False),
                    "texto":       txt,
                })
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
                ftype = SCHEMA.field(col).type
                df[col] = (False if ftype == pa.bool_() else
                           (0    if ftype == pa.int32() else
                           (0.0  if ftype == pa.float32() else "")))
        for col in ["pagina_inicio","pagina_fim","chunk_index","chunk_total","n_tokens","n_refs"]:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype("int32")
        df["densidade"] = pd.to_numeric(df["densidade"], errors="coerce").fillna(0.0).astype("float32")
        for col in ["tem_tabela","tem_imagem","tem_tachado"]:
            if col in df.columns:
                df[col] = df[col].apply(
                    lambda x: bool(x) if not isinstance(x, str)
                    else x.lower() not in ("", "false", "0")
                )
        tbl = pa.Table.from_pandas(df[SCHEMA.names], schema=SCHEMA)
        self._w.write_table(tbl)
        self._total += len(chunks)

    def close(self) -> int:
        self._w.close()
        return self._total


# ---------------------------------------------------------------------------
# Checkpoint — retoma onde parou
# ---------------------------------------------------------------------------

def _carregar_checkpoint() -> set[str]:
    if CHECKPOINT.exists():
        try:
            with open(CHECKPOINT, encoding="utf-8") as f:
                return set(json.load(f))
        except Exception:
            pass
    return set()


def _salvar_checkpoint(vistos: set[str]) -> None:
    CHECKPOINT.parent.mkdir(parents=True, exist_ok=True)
    with open(CHECKPOINT, "w", encoding="utf-8") as f:
        json.dump(list(vistos), f)


# ---------------------------------------------------------------------------
# Coleta de tarefas
# ---------------------------------------------------------------------------

def coletar_tarefas(
    json_path: Path,
    categorias: list[str],
    anos: list[str],
    limite: int = None,
) -> list[tuple[str, dict]]:
    with open(json_path, encoding="utf-8") as f:
        registros = json.load(f)

    tarefas = []
    vistos  = set()

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

            if ext == ".pdf":
                caminho = PASTA_PDFS / ano / cat / arquivo
            elif ext in (".htm", ".html"):
                caminho = PASTA_PDFS / ano / "html_ren" / arquivo.replace(ext, ".txt")
            elif ext in (".xlsx", ".xlsm"):
                caminho = PASTA_PDFS / ano / "xlsx" / arquivo.replace(ext, ".txt")
            elif ext == ".zip":
                pasta = PASTA_PDFS / ano / "zip_extraido" / arquivo.replace(".zip", "")
                if pasta.exists():
                    for ext_valida in ("*.pdf", "*.xlsx", "*.xlsm", "*.csv", "*.txt"):
                        for f_int in pasta.rglob(ext_valida):
                            k = str(f_int)
                            if k not in vistos:
                                vistos.add(k)
                                tarefas.append((k, _meta(reg, cat, url)))
                continue
            elif ext == ".rar":
                pasta = PASTA_PDFS / ano / "rar_extraido" / arquivo.replace(".rar", "")
                if pasta.exists():
                    for ext_valida in ("*.pdf", "*.xlsx", "*.xlsm", "*.csv", "*.txt"):
                        for f_int in pasta.rglob(ext_valida):
                            k = str(f_int)
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

    # TXTs de html_ren e PDFs vinculados dentro dos HTMLs
    for ano in (anos or ["2015", "2016", "2021", "2022"]):
        pasta = PASTA_PDFS / ano / "html_ren"
        if pasta.exists():
            # 1. Pega os textos limpos dos HTMLs
            for txt in pasta.rglob("*.txt"):
                k = str(txt)
                if k not in vistos:
                    vistos.add(k)
                    meta = {**{k2: "" for k2 in DEFAULTS},
                            "ano_fonte": ano, "categoria_pdf": "html_ren",
                            "doc_id": txt.stem}
                    tarefas.append((k, meta))
            
            # 2. Pega os PDFs anexados dentro dos HTMLs
            pasta_vinc = pasta / "pdfs_vinculados"
            if pasta_vinc.exists():
                for pdf_vinc in pasta_vinc.rglob("*.pdf"):
                    k = str(pdf_vinc)
                    if k not in vistos:
                        vistos.add(k)
                        meta = {**{k2: "" for k2 in DEFAULTS},
                                "ano_fonte": ano, "categoria_pdf": "anexo_html",
                                "doc_id": pdf_vinc.stem}
                        tarefas.append((k, meta))

    return tarefas[:limite] if limite else tarefas


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Parser ANEEL — texto + imagens + referências + checkpoint"
    )
    ap.add_argument("--json",           default=str(JSON_LIMPO))
    ap.add_argument("--saida",          default=str(PARQUET_OUT))
    ap.add_argument("--categoria",      nargs="+", default=CATEGORIAS,
                    choices=CATEGORIAS + ["html_ren"])
    ap.add_argument("--ano",            nargs="+", default=[],
                    choices=["2015","2016","2021","2022"])
    ap.add_argument("--limite",         type=int, default=None)
    ap.add_argument("--workers",        type=int, default=WORKERS)
    ap.add_argument("--chunk-size",     type=int, default=CHUNK_SIZE)
    ap.add_argument("--chunk-overlap",  type=int, default=CHUNK_OVERLAP)
    ap.add_argument("--sem-checkpoint", action="store_true",
                    help="Ignora checkpoint e reprocessa tudo do zero")
    ap.add_argument("--pasta-entrada",  type=str, default=None,
                    help="Pasta customizada com os PDFs (ex: pdfs_teste)")
    args = ap.parse_args()

    global PASTA_PDFS
    if args.pasta_entrada:
        PASTA_PDFS = RAIZ / args.pasta_entrada
        log.info(f"Lendo arquivos da pasta customizada: {PASTA_PDFS}")

    log.info("=" * 55)
    log.info("PARSER ANEEL")
    log.info("=" * 55)

    tarefas = coletar_tarefas(
        Path(args.json), args.categoria, args.ano, args.limite
    )

    # Checkpoint — filtra arquivos já processados
    ja_feitos = set()
    if not args.sem_checkpoint and CHECKPOINT.exists():
        ja_feitos = _carregar_checkpoint()
        antes = len(tarefas)
        tarefas = [(c, m) for c, m in tarefas if c not in ja_feitos]
        log.info(f"  Checkpoint: {len(ja_feitos)} já processados, "
                 f"{antes - len(tarefas)} pulados")

    log.info(f"  Arquivos a processar: {len(tarefas)}")

    if not tarefas:
        log.info("Nada a processar. Use --sem-checkpoint para reprocessar tudo.")
        return

    tempo_est = len(tarefas) / args.workers / 3
    h = int(tempo_est // 3600)
    m = int((tempo_est % 3600) // 60)
    log.info(f"  Workers:     {args.workers}")
    log.info(f"  RAM est.:    ~{args.workers * 150 + 500}MB")
    log.info(f"  Tempo est.:  ~{h}h {m}min")
    log.info(f"  Saída:       {args.saida}")
    log.info("=" * 55)

    worker_args = [
        (cam, meta, args.chunk_size, args.chunk_overlap)
        for cam, meta in tarefas
    ]

    # Se há checkpoint, abre o parquet em modo append
    modo_append = not args.sem_checkpoint and ja_feitos and Path(args.saida).exists()
    if modo_append:
        log.info("Modo append — adicionando ao parquet existente")
        writer = ParquetWriter.__new__(ParquetWriter)
        writer._w = pq.ParquetWriter(
            str(args.saida) + ".tmp", SCHEMA, compression="snappy"
        )
        writer._total = 0
        saida_tmp = Path(args.saida + ".tmp")
    else:
        writer = ParquetWriter(Path(args.saida))
        saida_tmp = None

    buffer  = []
    erros   = 0
    imagens = 0
    refs_encontradas = 0
    feitos_agora: set[str] = set()

    barra = tqdm(total=len(tarefas), desc="Parsing", unit="arq",
                 dynamic_ncols=True) if USA_TQDM else None

    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futuros = {ex.submit(worker, a): a[0] for a in worker_args}

        for fut in as_completed(futuros):
            caminho_feito = futuros[fut]
            try:
                chunks = fut.result()
                imagens += sum(1 for c in chunks if c.get("qualidade") == "imagem")
                refs_encontradas += sum(c.get("n_refs", 0) for c in chunks)
                buffer.extend(chunks)
                feitos_agora.add(caminho_feito)

                if len(buffer) >= BATCH_WRITE:
                    writer.write(buffer)
                    buffer = []
                    # Salva checkpoint periodicamente
                    _salvar_checkpoint(ja_feitos | feitos_agora)

            except Exception as e:
                erros += 1
                log.debug(f"Worker error em {caminho_feito}: {e}")

            if barra:
                barra.set_postfix({
                    "buf": len(buffer), "err": erros,
                    "img": imagens, "ref": refs_encontradas,
                })
                barra.update(1)

    if buffer:
        writer.write(buffer)

    total = writer.close()
    if barra:
        barra.close()

    # Salva checkpoint final
    _salvar_checkpoint(ja_feitos | feitos_agora)

    log.info("\n" + "=" * 55)
    log.info("CONCLUÍDO")
    log.info("=" * 55)
    log.info(f"  Chunks gerados      : {total:,}")
    log.info(f"  PDFs escaneados     : {imagens}")
    log.info(f"  Referências extraídas: {refs_encontradas:,}")
    log.info(f"  Erros               : {erros}")
    log.info(f"  Parquet salvo       : {args.saida}")
    log.info("=" * 55)
    log.info("\nProximos passos:")
    log.info("  1. python src/p1_ingestion/unir_parquets.py")
    log.info("  2. python src/p2_search/p2_indexar.py --parquet data/processed/chunks_completo_unificado.parquet --resetar")


if __name__ == "__main__":
    main()