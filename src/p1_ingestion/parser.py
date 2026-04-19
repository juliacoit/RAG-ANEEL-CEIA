"""
parser.py
=========
Pessoa 1 — Data Engineer · Sprint Dia 2

Extrai e limpa o texto dos PDFs baixados pela ANEEL,
produzindo uma lista de documentos estruturados prontos
para o chunker.py.

Biblioteca principal: PyMuPDF (fitz)
  pip install pymupdf

Uso — rodar da RAIZ do projeto:
  python src/p1_ingestion/parser.py
  python src/p1_ingestion/parser.py --ano 2016
  python src/p1_ingestion/parser.py --ano 2016 --limite 50   # teste rápido
  python src/p1_ingestion/parser.py --workers 4              # mais CPUs

Saída:
  Em memória  →  lista de dicts (consumida pelo chunker.py)
  Em disco    →  data/processed/parsed_<ano>.parquet  (opcional, --salvar)

Estrutura de cada documento retornado / salvo:
  {
    "id":               "DSP_3284_2016",
    "tipo_codigo":      "DSP",
    "tipo_nome":        "Despacho",
    "numero":           "3284",
    "ano":              "2016",
    "autor":            "ANEEL",
    "assunto":          "Acatamento",
    "ementa":           "...",          # pode ser null
    "data_assinatura":  "2016-12-15",
    "data_publicacao":  "2016-12-30",
    "url_pdf":          "https://...",
    "arquivo_pdf":      "dsp20163284.pdf",
    "categoria_pdf":    "texto_integral",
    "caminho_local":    "pdfs/2016/texto_integral/dsp20163284.pdf",
    "n_paginas":        4,
    "qualidade_ocr":    "ok",           # "ok" | "baixa" | "vazio"
    "texto_completo":   "..."           # texto limpo, todas as páginas
  }
"""

import re
import json
import logging
import argparse
import unicodedata
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Optional

import fitz          # PyMuPDF
import pandas as pd

# ---------------------------------------------------------------------------
# Configuração de logging
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

RAIZ_PROJETO   = Path(__file__).resolve().parent.parent.parent
ARQUIVO_JSON   = RAIZ_PROJETO / "data" / "aneel_vigentes_completo.json"
PASTA_PDFS     = RAIZ_PROJETO / "pdfs"
PASTA_SAIDA    = RAIZ_PROJETO / "data" / "processed"

# ---------------------------------------------------------------------------
# Parâmetros de qualidade de extração
# ---------------------------------------------------------------------------

# Mínimo de caracteres por página para ser considerada "com conteúdo"
MIN_CHARS_POR_PAGINA = 80

# Proporção mínima de páginas com conteúdo para o doc ser "ok"
MIN_PROPORCAO_PAGINAS_OK = 0.5

# Máximo de caracteres a inspecionar para detectar cabeçalho/rodapé repetido
JANELA_CABECALHO = 300

# ---------------------------------------------------------------------------
# Limpeza de texto
# ---------------------------------------------------------------------------

# Padrões compilados uma única vez para performance
_RE_HIFENIZACAO   = re.compile(r"-\s*\n\s*")            # hifenização no fim de linha
_RE_MULTIPLOS_NL  = re.compile(r"\n{3,}")                # 3+ quebras de linha → 2
_RE_ESPACOS_DUPLOS = re.compile(r"[ \t]{2,}")            # espaços/tabs duplos → 1
_RE_PAGINA_NUM    = re.compile(                          # números de página isolados
    r"(?m)^\s*(?:Página|página|Pg\.?|pg\.?|p\.)?\s*\d{1,4}\s*$"
)
_RE_LINHA_SO_PONTUA = re.compile(                        # linhas só com pontuação/traços
    r"(?m)^\s*[-–—=_*·•]{3,}\s*$"
)
_RE_CARACTERES_RUINS = re.compile(                       # caracteres de controle (exceto \n\t)
    r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]"
)


def _normalizar_unicode(texto: str) -> str:
    """
    Normaliza para NFC (forma canônica composta).
    Resolve problemas de caracteres acentuados duplicados
    comuns em PDFs antigos da ANEEL.
    """
    return unicodedata.normalize("NFC", texto)


def _remover_cabecalho_rodape(paginas: list[str]) -> list[str]:
    """
    Detecta e remove linhas que aparecem de forma idêntica em
    TODAS as páginas (tipicamente cabeçalho/rodapé institucionais).

    Estratégia:
      1. Para cada página, pega as N primeiras e N últimas linhas.
      2. Conta quantas vezes cada linha aparece.
      3. Se uma linha aparece em ≥ 80% das páginas, é cabeçalho/rodapé.
      4. Remove essas linhas de todas as páginas.
    """
    if len(paginas) < 3:
        return paginas  # poucos dados → não arrisca remover

    total = len(paginas)
    n_linhas_inspecao = 5  # primeiras + últimas linhas a inspecionar

    # Conta ocorrências de cada linha nos trechos de cabeçalho/rodapé
    contagem: dict[str, int] = {}
    for pag in paginas:
        linhas = pag.split("\n")
        candidatas = set(
            linhas[:n_linhas_inspecao] + linhas[-n_linhas_inspecao:]
        )
        for linha in candidatas:
            linha_strip = linha.strip()
            if len(linha_strip) > 5:  # ignora linhas muito curtas
                contagem[linha_strip] = contagem.get(linha_strip, 0) + 1

    # Linhas que aparecem em ≥ 80% das páginas → ruído
    ruido = {l for l, c in contagem.items() if c / total >= 0.80}

    if not ruido:
        return paginas

    paginas_limpas = []
    for pag in paginas:
        linhas_limpas = [
            l for l in pag.split("\n")
            if l.strip() not in ruido
        ]
        paginas_limpas.append("\n".join(linhas_limpas))

    return paginas_limpas


def limpar_texto(texto: str) -> str:
    """
    Pipeline completo de limpeza de texto extraído de PDF.

    Ordem das operações importa:
      1. Normalização unicode (antes de qualquer regex)
      2. Remoção de caracteres de controle
      3. Rehifenização (une palavras quebradas no fim de linha)
      4. Remoção de números de página isolados
      5. Remoção de linhas só com pontuação/traços
      6. Colapso de espaços e quebras de linha excessivas
      7. Strip final
    """
    texto = _normalizar_unicode(texto)
    texto = _RE_CARACTERES_RUINS.sub("", texto)
    texto = _RE_HIFENIZACAO.sub("", texto)        # "condi-\nções" → "condições"
    texto = _RE_PAGINA_NUM.sub("", texto)
    texto = _RE_LINHA_SO_PONTUA.sub("", texto)
    texto = _RE_ESPACOS_DUPLOS.sub(" ", texto)
    texto = _RE_MULTIPLOS_NL.sub("\n\n", texto)
    texto = texto.strip()
    return texto


# ---------------------------------------------------------------------------
# Extração de texto por PDF
# ---------------------------------------------------------------------------

def _avaliar_qualidade(paginas_texto: list[str], n_paginas_total: int) -> str:
    """
    Classifica a qualidade de extração do PDF.

    Retorna:
      "ok"     → extração suficiente para uso
      "baixa"  → PDF escaneado ou com poucas camadas de texto
      "vazio"  → nenhum texto extraído
    """
    if n_paginas_total == 0:
        return "vazio"

    paginas_com_conteudo = sum(
        1 for p in paginas_texto if len(p.strip()) >= MIN_CHARS_POR_PAGINA
    )
    proporcao = paginas_com_conteudo / n_paginas_total

    if proporcao == 0:
        return "vazio"
    elif proporcao < MIN_PROPORCAO_PAGINAS_OK:
        return "baixa"
    else:
        return "ok"


def extrair_texto_pdf(caminho_pdf: Path) -> tuple[str, int, str]:
    """
    Abre o PDF e extrai o texto de todas as páginas.

    Retorna:
      texto_completo  (str)  — texto limpo, todas as páginas unidas
      n_paginas       (int)  — total de páginas do PDF
      qualidade       (str)  — "ok" | "baixa" | "vazio"

    Estratégia de extração:
      Usa fitz.Page.get_text("text") com flag PRESERVE_READING_ORDER
      para garantir ordem correta em layouts de duas colunas.
      Se o texto da página for muito curto (página escaneada),
      registra mas não tenta OCR (seria outro serviço).
    """
    paginas_texto: list[str] = []

    try:
        doc = fitz.open(str(caminho_pdf))
        n_paginas = len(doc)

        for pagina in doc:
            # Extrai texto respeitando a ordem de leitura do layout
            texto_pag = pagina.get_text(
                "text",
                flags=fitz.TEXT_PRESERVE_READING_ORDER | fitz.TEXT_DEHYPHENATE,
            )
            paginas_texto.append(texto_pag)

        doc.close()

    except Exception as e:
        log.warning(f"Erro ao abrir PDF {caminho_pdf.name}: {e}")
        return "", 0, "vazio"

    # Remove cabeçalhos/rodapés repetidos entre páginas
    paginas_texto = _remover_cabecalho_rodape(paginas_texto)

    # Avalia qualidade antes de limpar (usa texto bruto para contar chars)
    qualidade = _avaliar_qualidade(paginas_texto, n_paginas)

    # Aplica limpeza em cada página e une tudo
    paginas_limpas = [limpar_texto(p) for p in paginas_texto]
    texto_completo = "\n\n".join(p for p in paginas_limpas if p.strip())

    return texto_completo, n_paginas, qualidade


# ---------------------------------------------------------------------------
# Processamento de um único registro do JSON
# ---------------------------------------------------------------------------

def processar_registro(registro: dict, pasta_pdfs: Path, categoria: str = "texto_integral") -> Optional[dict]:
    """
    Processa um registro do JSON (um ato normativo) e retorna
    o documento estruturado com texto extraído.

    Filtra apenas a categoria desejada (padrão: texto_integral).
    Retorna None se o PDF não existir ou a extração falhar.
    """
    # Encontra o PDF da categoria desejada
    pdf_alvo = None
    for pdf in registro.get("pdfs") or []:
        if pdf.get("categoria") == categoria:
            pdf_alvo = pdf
            break

    if pdf_alvo is None:
        return None  # registro sem PDF na categoria solicitada

    ano_fonte = registro.get("ano_fonte", "")
    arquivo   = pdf_alvo.get("arquivo", "")
    url       = pdf_alvo.get("url", "").replace("http://", "https://")

    caminho_pdf = pasta_pdfs / ano_fonte / categoria / arquivo

    if not caminho_pdf.exists():
        log.debug(f"PDF não encontrado localmente: {caminho_pdf}")
        return None

    if caminho_pdf.stat().st_size < 500:  # arquivo suspeito (< 500 bytes)
        log.warning(f"PDF muito pequeno, provavelmente inválido: {arquivo}")
        return None

    # Extrai texto
    texto, n_paginas, qualidade = extrair_texto_pdf(caminho_pdf)

    return {
        # Metadados do ato normativo
        "id":              registro.get("id", ""),
        "tipo_codigo":     registro.get("tipo_codigo", ""),
        "tipo_nome":       registro.get("tipo_nome", ""),
        "numero":          registro.get("numero", ""),
        "ano":             registro.get("ano", ""),
        "autor":           registro.get("autor", ""),
        "assunto":         registro.get("assunto", ""),
        "ementa":          registro.get("ementa"),
        "data_assinatura": registro.get("data_assinatura", ""),
        "data_publicacao": registro.get("data_publicacao", ""),
        # Metadados do PDF
        "url_pdf":         url,
        "arquivo_pdf":     arquivo,
        "categoria_pdf":   categoria,
        "caminho_local":   str(caminho_pdf.relative_to(pasta_pdfs.parent)),
        # Resultado da extração
        "n_paginas":       n_paginas,
        "qualidade_ocr":   qualidade,
        "texto_completo":  texto,
    }


# ---------------------------------------------------------------------------
# Processamento em lote (com paralelismo por processo)
# ---------------------------------------------------------------------------

def _processar_wrapper(args):
    """Wrapper para uso com ProcessPoolExecutor."""
    registro, pasta_pdfs_str, categoria = args
    return processar_registro(registro, Path(pasta_pdfs_str), categoria)


def parsear_todos(
    registros: list,
    pasta_pdfs: Path,
    categoria: str = "texto_integral",
    workers: int = 2,
    limite: Optional[int] = None,
) -> list[dict]:
    """
    Processa todos os registros em paralelo usando ProcessPoolExecutor.

    Args:
      registros   — lista de dicts do JSON ANEEL
      pasta_pdfs  — caminho para a pasta raiz dos PDFs baixados
      categoria   — categoria de PDF a extrair (padrão: texto_integral)
      workers     — número de processos paralelos
      limite      — se informado, processa só os primeiros N registros (teste)

    Returns:
      Lista de documentos estruturados com texto extraído.
    """
    if limite:
        registros = registros[:limite]
        log.info(f"Modo teste: processando apenas {limite} registros.")

    args_list = [
        (reg, str(pasta_pdfs), categoria)
        for reg in registros
    ]

    documentos = []
    erros = 0
    vazios = 0
    baixa_qualidade = 0

    log.info(f"Iniciando extração de {len(args_list)} registros | workers={workers} | categoria={categoria}")

    with ProcessPoolExecutor(max_workers=workers) as executor:
        futuros = {executor.submit(_processar_wrapper, args): args for args in args_list}

        for i, futuro in enumerate(as_completed(futuros), 1):
            try:
                doc = futuro.result()
            except Exception as e:
                log.error(f"Erro inesperado em worker: {e}")
                erros += 1
                continue

            if doc is None:
                erros += 1
                continue

            q = doc["qualidade_ocr"]
            if q == "vazio":
                vazios += 1
            elif q == "baixa":
                baixa_qualidade += 1
                documentos.append(doc)  # inclui mas sinaliza
            else:
                documentos.append(doc)

            if i % 500 == 0 or i == len(args_list):
                log.info(
                    f"  {i}/{len(args_list)} | "
                    f"ok={len(documentos)-baixa_qualidade} | "
                    f"baixa_qualidade={baixa_qualidade} | "
                    f"vazios={vazios} | "
                    f"erros={erros}"
                )

    log.info("=" * 55)
    log.info(f"EXTRAÇÃO CONCLUÍDA")
    log.info(f"  Documentos com texto  : {len(documentos)}")
    log.info(f"  Baixa qualidade (OCR) : {baixa_qualidade}")
    log.info(f"  Vazios / sem PDF      : {vazios}")
    log.info(f"  Erros                 : {erros}")
    log.info("=" * 55)

    return documentos


# ---------------------------------------------------------------------------
# Persistência em Parquet (opcional)
# ---------------------------------------------------------------------------

def salvar_parquet(documentos: list[dict], caminho_saida: Path) -> None:
    """
    Salva a lista de documentos em formato Parquet.
    Usado quando --salvar é passado ou quando chamado pelo chunker.

    Colunas salvas: todas as chaves do dict, exceto texto_completo
    (que vai para o parquet do chunker, não aqui).
    """
    caminho_saida.parent.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(documentos)

    # Garante tipos corretos
    for col in ["n_paginas"]:
        if col in df.columns:
            df[col] = df[col].astype("int32")

    df.to_parquet(caminho_saida, index=False, engine="pyarrow", compression="snappy")
    log.info(f"Parquet salvo: {caminho_saida}  ({len(df)} documentos, {caminho_saida.stat().st_size / 1024:.0f} KB)")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Extrai e limpa texto dos PDFs ANEEL → documentos estruturados"
    )
    parser.add_argument("--json",      default=str(ARQUIVO_JSON),    help="Caminho do JSON de metadados")
    parser.add_argument("--pdfs",      default=str(PASTA_PDFS),      help="Pasta raiz dos PDFs baixados")
    parser.add_argument("--saida",     default=str(PASTA_SAIDA),     help="Pasta de saída dos Parquets")
    parser.add_argument("--categoria", default="texto_integral",      help="Categoria de PDF a extrair")
    parser.add_argument("--ano",       nargs="+", default=[],         help="Filtrar por ano (ex: 2016)")
    parser.add_argument("--limite",    type=int,  default=None,       help="Processar só N registros (teste)")
    parser.add_argument("--workers",   type=int,  default=2,          help="Processos paralelos")
    parser.add_argument("--salvar",    action="store_true",           help="Salvar resultado em Parquet")
    args = parser.parse_args()

    # Carrega JSON
    json_path = Path(args.json)
    if not json_path.exists():
        log.error(f"JSON não encontrado: {json_path}")
        return

    with open(json_path, encoding="utf-8") as f:
        registros = json.load(f)

    log.info(f"JSON carregado: {len(registros)} registros")

    # Filtra por ano se solicitado
    if args.ano:
        registros = [r for r in registros if r.get("ano_fonte") in args.ano]
        log.info(f"Filtro por ano {args.ano}: {len(registros)} registros")

    # Extrai textos
    documentos = parsear_todos(
        registros=registros,
        pasta_pdfs=Path(args.pdfs),
        categoria=args.categoria,
        workers=args.workers,
        limite=args.limite,
    )

    if not documentos:
        log.warning("Nenhum documento extraído. Verifique se os PDFs foram baixados.")
        return

    # Salva em Parquet se solicitado
    if args.salvar:
        anos_str = "_".join(sorted(args.ano)) if args.ano else "todos"
        nome_arquivo = f"parsed_{args.categoria}_{anos_str}.parquet"
        caminho_saida = Path(args.saida) / nome_arquivo
        salvar_parquet(documentos, caminho_saida)

    # Preview dos primeiros resultados
    log.info("\nPREVIEW — primeiros 3 documentos:")
    for doc in documentos[:3]:
        trecho = doc["texto_completo"][:200].replace("\n", " ")
        log.info(
            f"  [{doc['id']}] páginas={doc['n_paginas']} "
            f"qualidade={doc['qualidade_ocr']} "
            f"chars={len(doc['texto_completo'])} "
            f"texto='{trecho}...'"
        )


if __name__ == "__main__":
    main()