"""
baixar_pdfs_aneel.py
====================
Pessoa 1 — Data Engineer

Script unificado de download de todos os arquivos da ANEEL.
Trata automaticamente todos os tipos de arquivo encontrados:
  - PDF         → baixa com curl_cffi (contorna Cloudflare)
  - HTML        → baixa, extrai texto, segue links para PDFs
  - ZIP         → baixa e extrai conteúdo
  - XLSX/XLSM  → baixa planilhas e extrai texto das células
  - RAR         → baixa e extrai conteúdo
  - URL dupla   → corrige protocolo duplicado e baixa

Pré-requisitos:
  pip install curl_cffi tqdm beautifulsoup4 openpyxl
  pip install rarfile  (opcional, para extrair RARs)

Uso:
  # Download completo (padrão — texto_integral):
  python src/p1_ingestion/baixar_pdfs_aneel.py

  # Todas as categorias:
  python src/p1_ingestion/baixar_pdfs_aneel.py --categorias texto_integral voto nota_tecnica decisao anexo outro

  # Testar com 50 arquivos:
  python src/p1_ingestion/baixar_pdfs_aneel.py --limite 50

  # Só um ano:
  python src/p1_ingestion/baixar_pdfs_aneel.py --ano 2016

  # Retentar falhas salvas (lê pdfs/falhas_download.json):
  python src/p1_ingestion/baixar_pdfs_aneel.py --retry-falhas

  # Mais workers = mais rápido (máx recomendado: 10):
  python src/p1_ingestion/baixar_pdfs_aneel.py --workers 8

Estrutura de saída:
  pdfs/
    ANO/texto_integral/     ← PDFs principais
    ANO/voto/               ← votos dos diretores
    ANO/nota_tecnica/       ← notas técnicas
    ANO/html_ren/           ← HTMLs + texto + PDFs vinculados
    ANO/zip_extraido/NOME/  ← conteúdo extraído de ZIPs
    ANO/xlsx/               ← planilhas + texto extraído
    ANO/rar_extraido/NOME/  ← conteúdo extraído de RARs
"""

import json
import time
import re
import zipfile
import argparse
import logging
from pathlib import Path
from io import BytesIO
from collections import defaultdict, Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import local

from curl_cffi import requests as curl_requests
from tqdm import tqdm

try:
    from bs4 import BeautifulSoup
    TEM_BS4 = True
except ImportError:
    TEM_BS4 = False

try:
    import openpyxl
    TEM_OPENPYXL = True
except ImportError:
    TEM_OPENPYXL = False

# ---------------------------------------------------------------------------
# Configurações
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

RAIZ         = Path(__file__).resolve().parent.parent.parent
ARQUIVO_JSON = RAIZ / "data" / "aneel_vigentes_completo.json"
PASTA_PDFS   = RAIZ / "pdfs"

WORKERS_PADRAO = 5
TIMEOUT        = 30
TIMEOUT_ZIP    = 120   # ZIPs de bases de dados podem ser grandes
TIMEOUT_RETRY  = 90
MAX_TENTATIVAS = 3
PAUSA_RETRY    = 2
PAUSA_ENTRE    = 0.3

TODAS_CATEGORIAS = ["texto_integral", "voto", "nota_tecnica", "decisao", "anexo", "outro"]

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Accept": "*/*",
    "Accept-Language": "pt-BR,pt;q=0.9",
    "Referer": "https://www2.aneel.gov.br/",
}

_thread_local = local()


# ---------------------------------------------------------------------------
# Session
# ---------------------------------------------------------------------------

def get_session() -> curl_requests.Session:
    """Sessão curl_cffi por thread — cada worker tem a sua."""
    if not hasattr(_thread_local, "session"):
        s = curl_requests.Session()
        s.headers.update(HEADERS)
        s.impersonate = "chrome124"
        _thread_local.session = s
    return _thread_local.session


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def normalizar_url(url: str) -> str:
    """Converte HTTP → HTTPS e corrige URLs com protocolo duplicado."""
    if not url:
        return ""
    url = url.strip()
    # Corrige protocolo duplicado: 'https://  https://...' ou 'https://https://...'
    matches = list(re.finditer(r'https?://', url))
    if len(matches) > 1:
        url = url[matches[-1].start():].strip()
    # Normaliza HTTP → HTTPS
    return url.replace("http://", "https://")


def extrair_ano(arquivo: str, url: str = "") -> str:
    """Extrai ano do nome do arquivo ou URL."""
    m = re.search(r"(2015|2016|2017|2018|2019|2020|2021|2022|2023)", arquivo + url)
    return m.group(1) if m else "2016"


def categorizar_tipo(tipo_raw: str) -> str:
    """Categoriza o tipo do PDF a partir do campo tipo bruto do JSON."""
    t = (tipo_raw or "").lower().strip().rstrip(":")
    if "texto integral" in t:                       return "texto_integral"
    if "voto" in t:                                 return "voto"
    if "nota" in t:                                 return "nota_tecnica"
    if "decisão" in t or "decisao" in t:            return "decisao"
    if "anexo" in t:                                return "anexo"
    return "outro"


def detectar_tipo_arquivo(url: str, arquivo: str) -> str:
    """Detecta o tipo real do arquivo pela extensão."""
    nome = arquivo.lower().strip()
    if nome.endswith(".zip"):                       return "zip"
    if nome.endswith((".html", ".htm")):            return "html"
    if nome.endswith((".xlsx", ".xlsm")):           return "xlsx"
    if nome.endswith(".rar"):                       return "rar"
    if nome.endswith(".pdf") or "pdf" in nome:      return "pdf"
    return "pdf"  # padrão


def ja_existe(destino: Path) -> bool:
    return destino.exists() and destino.stat().st_size > 512


# ---------------------------------------------------------------------------
# Download de PDF (tipo padrão)
# ---------------------------------------------------------------------------

def baixar_pdf(url: str, destino: Path) -> dict:
    """Baixa um PDF com retry automático."""
    destino.parent.mkdir(parents=True, exist_ok=True)
    session = get_session()

    for tentativa in range(1, MAX_TENTATIVAS + 1):
        try:
            resp = session.get(url, timeout=TIMEOUT)

            if resp.status_code == 200:
                conteudo = resp.content
                if conteudo[:4] != b"%PDF":
                    if tentativa == MAX_TENTATIVAS:
                        return {"status": "erro_html", "url": url}
                    time.sleep(PAUSA_RETRY)
                    continue
                destino.write_bytes(conteudo)
                return {"status": "ok", "kb": len(conteudo) // 1024}

            elif resp.status_code == 404:
                return {"status": "erro_404", "url": url}

            elif resp.status_code == 429:
                time.sleep(PAUSA_RETRY * 10 * tentativa)
                continue

            else:
                if tentativa == MAX_TENTATIVAS:
                    return {"status": f"erro_{resp.status_code}", "url": url}
                time.sleep(PAUSA_RETRY)

        except Exception as e:
            if tentativa == MAX_TENTATIVAS:
                return {"status": "erro_timeout", "url": url, "erro": str(e)[:80]}
            time.sleep(PAUSA_RETRY * tentativa)

    return {"status": "erro_desconhecido", "url": url}


# ---------------------------------------------------------------------------
# Download de HTML (Resoluções Normativas)
# ---------------------------------------------------------------------------

def baixar_html(url: str, arquivo: str, ano: str) -> dict:
    """Baixa HTML, extrai texto limpo e segue links para PDFs."""
    pasta = PASTA_PDFS / ano / "html_ren"
    pasta.mkdir(parents=True, exist_ok=True)
    destino = pasta / arquivo

    if ja_existe(destino):
        return {"status": "pulado"}

    session = get_session()

    try:
        resp = session.get(url, timeout=TIMEOUT)
        if resp.status_code != 200:
            return {"status": f"erro_{resp.status_code}", "url": url}

        html_content = resp.text
        destino.write_text(html_content, encoding="utf-8")

        # Extrai texto limpo
        if TEM_BS4:
            soup = BeautifulSoup(html_content, "html.parser")
            for tag in soup(["script", "style", "nav", "header", "footer"]):
                tag.decompose()
            texto = soup.get_text(separator="\n", strip=True)
            linhas = [l for l in texto.split("\n") if len(l.strip()) > 20]
            if linhas:
                txt_path = pasta / arquivo.replace(".html", ".txt").replace(".htm", ".txt")
                txt_path.write_text("\n".join(linhas), encoding="utf-8")

            # Segue links para PDFs dentro do HTML
            pdfs_baixados = 0
            from urllib.parse import urlparse
            parsed = urlparse(url)
            base = f"{parsed.scheme}://{parsed.netloc}"

            for a in soup.find_all("a", href=True):
                href = a["href"]
                if href.startswith("/"):
                    href = base + href
                elif not href.startswith("http"):
                    href = url.rsplit("/", 1)[0] + "/" + href

                if "aneel.gov.br" not in href:
                    continue

                nome_link = href.split("/")[-1]
                if not nome_link:
                    continue

                if href.endswith(".pdf"):
                    dest_link = pasta / "pdfs_vinculados" / nome_link
                elif href.endswith((".html", ".htm")):
                    dest_link = pasta / "html_vinculado" / nome_link
                else:
                    continue

                if ja_existe(dest_link):
                    continue

                try:
                    time.sleep(PAUSA_ENTRE)
                    r2 = session.get(href, timeout=TIMEOUT)
                    if r2.status_code == 200:
                        dest_link.parent.mkdir(parents=True, exist_ok=True)
                        if href.endswith(".pdf") and r2.content[:4] == b"%PDF":
                            dest_link.write_bytes(r2.content)
                            pdfs_baixados += 1
                        elif href.endswith((".html", ".htm")):
                            dest_link.write_text(r2.text, encoding="utf-8")
                            if TEM_BS4:
                                s2 = BeautifulSoup(r2.text, "html.parser")
                                for t in s2(["script","style","nav"]):
                                    t.decompose()
                                txt2 = s2.get_text(separator="\n", strip=True)
                                linhas2 = [l for l in txt2.split("\n") if len(l.strip()) > 20]
                                if linhas2:
                                    dest_link.with_suffix(".txt").write_text("\n".join(linhas2), encoding="utf-8")
                            pdfs_baixados += 1
                except Exception:
                    pass

            return {"status": "ok", "tipo": "html", "pdfs_vinculados": pdfs_baixados}

        return {"status": "ok", "tipo": "html", "pdfs_vinculados": 0}

    except Exception as e:
        return {"status": "erro_timeout", "url": url, "erro": str(e)[:60]}


# ---------------------------------------------------------------------------
# Download de ZIP
# ---------------------------------------------------------------------------

def baixar_zip(url: str, arquivo: str, ano: str) -> dict:
    """Baixa ZIP e extrai conteúdo na pasta correta."""
    pasta = PASTA_PDFS / ano / "zip_extraido" / arquivo.replace(".zip", "")
    pasta.mkdir(parents=True, exist_ok=True)

    session = get_session()

    # Backoff exponencial: tenta com timeouts crescentes
    timeouts = [30, 60, 120, 240]
    resp = None
    ultimo_erro = ""

    for t in timeouts:
        try:
            resp = session.get(url, timeout=t)
            if resp.status_code == 200:
                break
            ultimo_erro = f"HTTP {resp.status_code}"
        except Exception as e:
            ultimo_erro = str(e)[:80]
            log.info(f"    Timeout em {t}s para {arquivo} — tentando com {min(t*2, 240)}s...")
            time.sleep(2)

    if resp is None or resp.status_code != 200:
        return {"status": "erro_timeout", "url": url, "erro": ultimo_erro}

    extraidos = 0
    pulados   = 0

    def sanitizar_caminho(nome: str) -> str:
        """
        Sanitiza nome de arquivo/pasta extraído do ZIP para funcionar no Windows.
        Remove caracteres ilegais, preserva a estrutura de subpastas.
        """
        import unicodedata
        partes = nome.replace("\\", "/").split("/")
        partes_limpas = []
        chars_ilegais = r'<>:"|?*'
        for parte in partes:
            # Remove caracteres ilegais no Windows
            for c in chars_ilegais:
                parte = parte.replace(c, "_")
            # Remove espaços no início e fim, e pontos no final
            parte = parte.strip().rstrip(".")
            # Limita comprimento de cada parte
            if len(parte) > 100:
                parte = parte[:100]
            if parte:
                partes_limpas.append(parte)
        return "/".join(partes_limpas)

    try:
        with zipfile.ZipFile(BytesIO(resp.content)) as z:
            for nome_interno in z.namelist():
                # Ignora diretórios
                if nome_interno.endswith("/") or nome_interno.endswith("\\"):
                    continue

                # Sanitiza o caminho para Windows
                nome_limpo = sanitizar_caminho(nome_interno)
                if not nome_limpo:
                    continue

                destino = pasta / nome_limpo

                # Cria pastas pai de forma segura
                try:
                    destino.parent.mkdir(parents=True, exist_ok=True)
                except OSError:
                    # Se ainda falhar, tenta caminho mais curto (só o arquivo)
                    nome_arquivo = Path(nome_limpo).name
                    destino = pasta / nome_arquivo
                    destino.parent.mkdir(parents=True, exist_ok=True)

                # Pula se já existe com conteúdo
                if destino.exists() and destino.stat().st_size > 0:
                    pulados += 1
                    continue

                try:
                    destino.write_bytes(z.read(nome_interno))
                    extraidos += 1
                except OSError:
                    pulados += 1

    except zipfile.BadZipFile:
        zip_bruto = pasta / arquivo
        if not zip_bruto.exists():
            zip_bruto.write_bytes(resp.content)
        return {"status": "erro_zip_corrompido", "url": url}

    if extraidos == 0 and pulados > 0:
        return {"status": "pulado", "arquivo": arquivo}

    return {"status": "ok", "tipo": "zip", "extraidos": extraidos, "pulados": pulados}


# ---------------------------------------------------------------------------
# Download de XLSX
# ---------------------------------------------------------------------------

def baixar_xlsx(url: str, arquivo: str, ano: str) -> dict:
    """Baixa planilha Excel e extrai texto das células."""
    pasta = PASTA_PDFS / ano / "xlsx"
    pasta.mkdir(parents=True, exist_ok=True)
    destino = pasta / arquivo

    if ja_existe(destino):
        return {"status": "pulado"}

    session = get_session()
    try:
        resp = session.get(url, timeout=TIMEOUT_ZIP)
        if resp.status_code != 200:
            return {"status": f"erro_{resp.status_code}", "url": url}

        destino.write_bytes(resp.content)

        if TEM_OPENPYXL:
            try:
                wb = openpyxl.load_workbook(BytesIO(resp.content), data_only=True)
                linhas = []
                for sheet in wb.worksheets:
                    linhas.append(f"=== Aba: {sheet.title} ===")
                    for row in sheet.iter_rows(values_only=True):
                        celulas = [str(c) for c in row if c is not None and str(c).strip()]
                        if celulas:
                            linhas.append(" | ".join(celulas))
                if linhas:
                    txt = pasta / re.sub(r"\.(xlsx|xlsm)$", ".txt", arquivo)
                    txt.write_text("\n".join(linhas), encoding="utf-8")
            except Exception:
                pass

        return {"status": "ok", "tipo": "xlsx"}

    except Exception as e:
        return {"status": "erro_timeout", "url": url, "erro": str(e)[:60]}


# ---------------------------------------------------------------------------
# Download de RAR
# ---------------------------------------------------------------------------

def baixar_rar(url: str, arquivo: str, ano: str) -> dict:
    """Baixa RAR e extrai conteúdo."""
    pasta = PASTA_PDFS / ano / "rar_extraido" / arquivo.replace(".rar", "")
    pasta.mkdir(parents=True, exist_ok=True)
    rar_path = pasta / arquivo

    session = get_session()
    try:
        resp = session.get(url, timeout=TIMEOUT)
        if resp.status_code != 200:
            return {"status": f"erro_{resp.status_code}", "url": url}

        rar_path.write_bytes(resp.content)

        try:
            import rarfile
            rarfile.UNRAR_TOOL = r"C:\Program Files\7-Zip\7z.exe"
            with rarfile.RarFile(rar_path) as r:
                r.extractall(pasta)
            rar_path.unlink()
            return {"status": "ok", "tipo": "rar"}
        except ImportError:
            log.warning("rarfile nao instalado: pip install rarfile")
            return {"status": "ok_nao_extraido", "tipo": "rar"}
        except Exception as e:
            log.warning(f"RAR nao extraido: {e}")
            log.warning("Instale o 7-Zip: https://www.7-zip.org/")
            return {"status": "ok_nao_extraido", "tipo": "rar", "nota": "instale 7-Zip para extrair"}

    except Exception as e:
        return {"status": "erro_timeout", "url": url, "erro": str(e)[:60]}


# ---------------------------------------------------------------------------
# Dispatcher — decide qual função chamar
# ---------------------------------------------------------------------------

def baixar_um(item: dict) -> dict:
    """
    Função central — detecta o tipo do arquivo e chama
    o tratador correto automaticamente.
    """
    url     = normalizar_url(item.get("url", ""))
    arquivo = (item.get("arquivo", "") or "").strip()
    ano     = item.get("ano", extrair_ano(arquivo, url))
    cat     = item.get("cat", "texto_integral")

    if not url or not arquivo:
        return {"status": "erro_url_vazia", "arquivo": arquivo}

    tipo = detectar_tipo_arquivo(url, arquivo)

    # Pula se já existe
    if tipo == "pdf":
        destino = PASTA_PDFS / ano / cat / arquivo
        if ja_existe(destino):
            return {"status": "pulado", "arquivo": arquivo}
        r = baixar_pdf(url, destino)
    elif tipo == "html":
        r = baixar_html(url, arquivo, ano)
    elif tipo == "zip":
        r = baixar_zip(url, arquivo, ano)
    elif tipo == "xlsx":
        r = baixar_xlsx(url, arquivo, ano)
    elif tipo == "rar":
        r = baixar_rar(url, arquivo, ano)
    else:
        destino = PASTA_PDFS / ano / cat / arquivo
        if ja_existe(destino):
            return {"status": "pulado", "arquivo": arquivo}
        r = baixar_pdf(url, destino)

    r["arquivo"] = arquivo
    r["tipo_arquivo"] = tipo
    return r


# ---------------------------------------------------------------------------
# Coleta de downloads
# ---------------------------------------------------------------------------

def coletar_downloads(json_path: Path, categorias: list, anos: list, limite: int = None) -> list:
    """Lê o JSON e retorna lista de downloads pendentes."""
    with open(json_path, encoding="utf-8") as f:
        registros = json.load(f)

    downloads = []
    for reg in registros:
        ano_fonte = reg.get("ano_fonte", "")
        if anos and ano_fonte not in anos:
            continue
        for pdf in reg.get("pdfs") or []:
            cat = pdf.get("categoria") or categorizar_tipo(pdf.get("tipo", ""))
            if cat not in categorias:
                continue
            url     = normalizar_url(pdf.get("url", ""))
            arquivo = (pdf.get("arquivo", "") or "").strip()
            if not url or not arquivo:
                continue

            tipo = detectar_tipo_arquivo(url, arquivo)
            if tipo == "pdf":
                destino = PASTA_PDFS / ano_fonte / cat / arquivo
                if ja_existe(destino):
                    continue
            downloads.append({
                "url":      url,
                "arquivo":  arquivo,
                "ano":      ano_fonte,
                "cat":      cat,
            })

    return downloads[:limite] if limite else downloads


def coletar_retry(falhas_path: Path) -> list:
    """Lê arquivo de falhas e prepara para retry."""
    with open(falhas_path, encoding="utf-8") as f:
        falhas = json.load(f)

    downloads = []
    ignorados = 0
    for f in falhas:
        if f.get("status") == "erro_404":
            ignorados += 1
            continue
        url     = normalizar_url(f.get("url", ""))
        arquivo = (f.get("arquivo", "") or "").strip()
        if not url or not arquivo:
            continue
        ano = extrair_ano(arquivo, url)
        downloads.append({
            "url":      url,
            "arquivo":  arquivo,
            "ano":      ano,
            "cat":      "texto_integral",
        })

    log.info(f"  {ignorados} erros 404 ignorados (arquivo removido do servidor)")
    return downloads


# ---------------------------------------------------------------------------
# Execução paralela
# ---------------------------------------------------------------------------

def baixar_todos(downloads: list, workers: int, desc: str = "Baixando") -> tuple:
    """Executa downloads em paralelo com múltiplos workers."""
    resultados = defaultdict(int)
    falhas     = []

    barra = tqdm(total=len(downloads), desc=desc, unit="arq", dynamic_ncols=True)

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futuros = {executor.submit(baixar_um, item): item for item in downloads}
        for futuro in as_completed(futuros):
            r = futuro.result()
            resultados[r["status"]] += 1
            barra.update(1)

            n_ok  = resultados.get("ok", 0) + resultados.get("pulado", 0)
            n_err = sum(v for k, v in resultados.items() if k.startswith("erro"))
            barra.set_postfix({"ok": n_ok, "erros": n_err, "arq": r["arquivo"][:18]})

            if r["status"].startswith("erro") and r["status"] != "erro_404":
                falhas.append({**futuros[futuro], "status": r["status"], **{k: v for k, v in r.items() if k != "arquivo"}})

    barra.close()
    return dict(resultados), falhas


def salvar_falhas(falhas: list) -> None:
    PASTA_PDFS.mkdir(parents=True, exist_ok=True)
    path = PASTA_PDFS / "falhas_download.json"
    if not falhas:
        # Sem falhas — limpa o arquivo antigo se existir
        if path.exists():
            path.unlink()
            log.info("Arquivo de falhas removido — todos resolvidos!")
        return
    with open(path, "w", encoding="utf-8") as f:
        json.dump(falhas, f, ensure_ascii=False, indent=2)
    log.info(f"Falhas salvas: {path}")
    log.info("Use --retry-falhas para tentar novamente.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Download unificado de arquivos da ANEEL (PDF, HTML, ZIP, XLSX, RAR)"
    )
    parser.add_argument("--json", default=str(ARQUIVO_JSON))
    parser.add_argument(
        "--categorias", nargs="+", default=["texto_integral"],
        choices=TODAS_CATEGORIAS,
    )
    parser.add_argument(
        "--ano", nargs="+", default=[],
        choices=["2016", "2021", "2022"],
    )
    parser.add_argument("--limite", type=int, default=None)
    parser.add_argument(
        "--workers", type=int, default=WORKERS_PADRAO,
        help=f"Downloads simultâneos (padrão: {WORKERS_PADRAO}, max: 10)",
    )
    parser.add_argument(
        "--retry-falhas", action="store_true",
        help="Relê pdfs/falhas_download.json e retenta (exceto 404)",
    )
    args = parser.parse_args()

    # Avisa sobre dependências opcionais
    if not TEM_BS4:
        log.warning("beautifulsoup4 não instalado — HTMLs serão baixados sem extração de texto/links")
        log.warning("Instale: pip install beautifulsoup4")
    if not TEM_OPENPYXL:
        log.warning("openpyxl não instalado — XLSX serão baixados sem extração de texto")
        log.warning("Instale: pip install openpyxl")

    # Modo retry
    if args.retry_falhas:
        falhas_path = PASTA_PDFS / "falhas_download.json"
        if not falhas_path.exists():
            log.error(f"Arquivo não encontrado: {falhas_path}")
            return
        log.info(f"Modo retry — lendo {falhas_path.name}...")
        downloads = coletar_retry(falhas_path)
        if not downloads:
            log.info("Nada para retentar.")
            return
        log.info(f"Retentando {len(downloads)} arquivos...\n")
        inicio = time.time()
        resultados, falhas = baixar_todos(downloads, args.workers, "Retry")
        elapsed = time.time() - inicio
        log.info(f"\nRetry concluido em {elapsed/60:.1f} min")
        log.info(f"  Resolvidos : {resultados.get('ok', 0)}")
        log.info(f"  Ainda erros: {len(falhas)}")
        salvar_falhas(falhas)
        return

    # Download normal
    json_path = Path(args.json)
    if not json_path.exists():
        log.error(f"JSON não encontrado: {json_path}")
        return

    log.info(f"Lendo {json_path.name}...")
    downloads = coletar_downloads(json_path, args.categorias, args.ano, args.limite)

    if not downloads:
        log.info("Nada a baixar — todos os arquivos já existem!")
        return

    por_tipo = Counter(detectar_tipo_arquivo(d["url"], d["arquivo"]) for d in downloads)
    por_ano  = Counter(d["ano"] for d in downloads)
    tempo_est = len(downloads) / args.workers / 3.5
    horas, minutos = int(tempo_est // 3600), int((tempo_est % 3600) // 60)
    tempo_str = f"{horas}h {minutos}min" if horas else f"{minutos} min"

    log.info("=" * 55)
    log.info("PLANO DE DOWNLOAD")
    log.info("=" * 55)
    log.info(f"  A baixar       : {len(downloads)}")
    log.info(f"  Workers        : {args.workers}")
    log.info(f"  Tempo estimado : ~{tempo_str}")
    log.info(f"\n  Por tipo de arquivo:")
    for t, n in por_tipo.most_common():
        log.info(f"    {t:15s}: {n}")
    log.info(f"\n  Por ano:")
    for a in sorted(por_ano):
        log.info(f"    {a}: {por_ano[a]}")
    log.info("=" * 55)

    inicio = time.time()
    resultados, falhas = baixar_todos(downloads, args.workers)
    elapsed = time.time() - inicio

    log.info("\n" + "=" * 55)
    log.info("RESULTADO FINAL")
    log.info("=" * 55)
    log.info(f"  Baixados   : {resultados.get('ok', 0)}")
    log.info(f"  Pulados    : {resultados.get('pulado', 0)}")
    log.info(f"  Erro 404   : {resultados.get('erro_404', 0)}")
    log.info(f"  Outros err : {len(falhas)}")
    log.info(f"  Tempo      : {elapsed/60:.1f} min")
    log.info("=" * 55)

    salvar_falhas(falhas)

    if not falhas:
        log.info("\nConcluido! Proximo passo: rodar parser.py para extrair texto dos PDFs.")
    else:
        log.info(f"\n{len(falhas)} falharam. Use --retry-falhas para tentar novamente.")

    # Verifica se parser gerou lista de downloads referenciados
    _baixar_referenciados()


def _baixar_referenciados() -> None:
    """
    Verifica se o parser identificou arquivos referenciados
    fora do corpus e os baixa automaticamente.
    Lê pdfs/downloads_referenciados.json gerado pelo parser.py.
    """
    log_path = PASTA_PDFS / "downloads_referenciados.json"
    if not log_path.exists():
        return

    with open(log_path, encoding="utf-8") as f:
        downloads = json.load(f)

    if not downloads:
        return

    log.info(f"\n{'='*55}")
    log.info(f"BAIXANDO {len(downloads)} ARQUIVOS REFERENCIADOS")
    log.info(f"(documentos citados nos PDFs, ainda não no corpus)")
    log.info(f"{'='*55}")

    resultados, falhas = baixar_todos(downloads, workers=3)

    log.info(f"  Baixados: {resultados.get('ok', 0)}")
    log.info(f"  Erros   : {len(falhas)}")

    # Remove o arquivo de lista após processar
    log_path.unlink(missing_ok=True)

    if falhas:
        salvar_falhas(falhas)


if __name__ == "__main__":
    main()