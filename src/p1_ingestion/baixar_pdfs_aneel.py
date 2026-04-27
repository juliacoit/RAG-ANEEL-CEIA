"""
baixar_pdfs_playwright.py  (v4 — stealth + download direto pelo navegador)
===========================================================================
Estratégia:
  - playwright-stealth esconde todas as marcas de automação do Chromium
  - O navegador baixa cada PDF diretamente (sem requests separado)
  - Intercepta o binário da resposta antes do Chrome renderizar

INSTALAÇÃO:
  pip install playwright playwright-stealth
  playwright install chromium

Uso:
  python src/p1_ingestion/baixar_pdfs_playwright.py --ano 2016 --limite 10
  python src/p1_ingestion/baixar_pdfs_playwright.py --ano 2016
"""

import json
import time
import logging
import argparse
from pathlib import Path
from collections import defaultdict

from playwright.sync_api import sync_playwright, TimeoutError as PWTimeout
from playwright_stealth import Stealth

def stealth_sync(page):
    """Compatível com playwright-stealth 2.x"""
    Stealth().apply_stealth_sync(page)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

RAIZ_PROJETO = Path(__file__).resolve().parent.parent.parent
ARQUIVO_JSON = RAIZ_PROJETO / "data" / "aneel_vigentes_completo.json"
PASTA_PDFS   = RAIZ_PROJETO / "pdfs"

TIMEOUT_MS            = 60_000
PAUSA_ENTRE_DOWNLOADS = 2.0
PAUSA_APOS_ERRO       = 15.0
MAX_ERROS_SEGUIDOS    = 5
CATEGORIAS_PADRAO     = ["texto_integral"]


def coletar_downloads(json_path: Path, categorias: list, anos: list) -> list:
    with open(json_path, encoding="utf-8") as f:
        registros = json.load(f)
    downloads = []
    for reg in registros:
        ano_fonte = reg.get("ano_fonte", "")
        if anos and ano_fonte not in anos:
            continue
        for pdf in reg.get("pdfs") or []:
            cat     = pdf.get("categoria", "")
            url     = pdf.get("url", "").replace("http://", "https://")
            arquivo = pdf.get("arquivo", "")
            if cat not in categorias or not url or not arquivo:
                continue
            destino = PASTA_PDFS / ano_fonte / cat / arquivo
            downloads.append({
                "url": url, "destino": destino,
                "arquivo": arquivo, "categoria": cat,
                "ano_fonte": ano_fonte, "id": reg.get("id", ""),
            })
    return downloads


def baixar_pdfs(downloads: list) -> tuple[dict, list]:
    resultados     = defaultdict(int)
    falhas         = []
    erros_seguidos = 0

    pendentes  = [d for d in downloads if not d["destino"].exists() or d["destino"].stat().st_size < 500]
    ja_existem = len(downloads) - len(pendentes)

    log.info(f"Total na lista    : {len(downloads)}")
    log.info(f"Já existem (pular): {ja_existem}")
    log.info(f"A baixar          : {len(pendentes)}")

    if not pendentes:
        log.info("Nada a fazer.")
        return dict(resultados), falhas

    with sync_playwright() as p:
        browser = p.chromium.launch(
            headless=True,
            args=[
                "--no-sandbox",
                "--disable-setuid-sandbox",
                "--disable-dev-shm-usage",
            ],
        )
        context = browser.new_context(
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0.0.0 Safari/537.36"
            ),
            viewport={"width": 1280, "height": 800},
            locale="pt-BR",
            extra_http_headers={
                "Accept":          "application/pdf,*/*",
                "Accept-Language": "pt-BR,pt;q=0.9",
                "Referer":         "https://www2.aneel.gov.br/",
            },
        )

        page = context.new_page()

        # Aplica stealth — remove todas as marcas de automação
        stealth_sync(page)

        # Aquece o navegador na página inicial
        log.info("Aquecendo navegador na página inicial da ANEEL...")
        try:
            page.goto("https://www2.aneel.gov.br/", timeout=TIMEOUT_MS)
            time.sleep(4)
            log.info("Pronto. Iniciando downloads...")
        except Exception as e:
            log.warning(f"Página inicial falhou ({e}), continuando mesmo assim...")

        for i, item in enumerate(pendentes, 1):
            url     = item["url"]
            destino = Path(item["destino"])
            arquivo = item["arquivo"]

            destino.parent.mkdir(parents=True, exist_ok=True)
            log.info(f"[{i}/{len(pendentes)}] {arquivo}")

            # Captura os bytes da resposta via rota interceptada
            captura = {}

            def handle_route(route, request, _url=url, _cap=captura):
                """Intercepta a requisição do PDF e captura os bytes."""
                if _url in request.url:
                    response = route.fetch()
                    if response.status == 200:
                        _cap["bytes"]        = response.body()
                        _cap["content_type"] = response.headers.get("content-type", "")
                        _cap["status"]       = response.status
                    else:
                        _cap["status"] = response.status
                    route.fulfill(response=response)
                else:
                    route.continue_()

            try:
                # Registra interceptação para a URL específica
                page.route("**/*", handle_route)

                page.goto(url, timeout=TIMEOUT_MS, wait_until="domcontentloaded")
                time.sleep(1)

                page.unroute("**/*", handle_route)

                status_http = captura.get("status")
                dados       = captura.get("bytes", b"")

                if status_http == 403:
                    raise Exception("HTTP 403 — Cloudflare bloqueou mesmo com stealth")

                if not dados or len(dados) < 500:
                    raise Exception(
                        f"Dados insuficientes: {len(dados)} bytes "
                        f"(status={status_http}, type={captura.get('content_type', '?')})"
                    )

                content_type = captura.get("content_type", "")
                if "html" in content_type:
                    raise Exception(f"Resposta HTML em vez de PDF: {content_type}")

                destino.write_bytes(dados)
                log.info(f"  ✓ OK ({len(dados)/1024:.0f} KB)")
                resultados["ok"] += 1
                erros_seguidos = 0

            except PWTimeout:
                page.unroute("**/*", handle_route)
                log.warning("  ✗ Timeout")
                falhas.append({**item, "destino": str(destino), "status": "erro_timeout"})
                resultados["erro_timeout"] += 1
                erros_seguidos += 1
                time.sleep(PAUSA_APOS_ERRO)

            except Exception as e:
                page.unroute("**/*", handle_route)
                erro_str = str(e)
                log.warning(f"  ✗ {erro_str[:150]}")
                status_erro = "erro_403" if "403" in erro_str else "erro_outro"
                falhas.append({**item, "destino": str(destino), "status": status_erro, "erro": erro_str[:200]})
                resultados[status_erro] += 1
                erros_seguidos += 1
                time.sleep(PAUSA_APOS_ERRO if "403" in erro_str else PAUSA_ENTRE_DOWNLOADS)

            if erros_seguidos >= MAX_ERROS_SEGUIDOS:
                log.error(f"{MAX_ERROS_SEGUIDOS} erros consecutivos. Parando.")
                break

            time.sleep(PAUSA_ENTRE_DOWNLOADS)

            if i % 50 == 0:
                log.info(f"--- {i}/{len(pendentes)} | ok={resultados['ok']} | erros={len(falhas)} ---")

        page.close()
        context.close()
        browser.close()

    return dict(resultados), falhas


def salvar_falhas(falhas: list) -> None:
    if not falhas:
        return
    log_path = PASTA_PDFS / "falhas_playwright.json"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    serializaveis = [
        {k: str(v) if isinstance(v, Path) else v for k, v in f.items()}
        for f in falhas
    ]
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(serializaveis, f, ensure_ascii=False, indent=2)
    log.info(f"Log de falhas: {log_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--json",       default=str(ARQUIVO_JSON))
    parser.add_argument("--categorias", nargs="+", default=CATEGORIAS_PADRAO)
    parser.add_argument("--ano",        nargs="+", default=[], choices=["2016", "2021", "2022"])
    parser.add_argument("--limite",     type=int,  default=None)
    args = parser.parse_args()

    json_path = Path(args.json)
    if not json_path.exists():
        log.error(f"JSON não encontrado: {json_path}")
        return

    downloads = coletar_downloads(json_path, args.categorias, args.ano)
    if not downloads:
        log.warning("Nenhum PDF encontrado.")
        return

    if args.limite:
        downloads = downloads[:args.limite]
        log.info(f"Modo teste: {args.limite} arquivos.")

    log.info("=" * 55)
    resultados, falhas = baixar_pdfs(downloads)
    log.info("=" * 55)
    log.info(f"Baixados : {resultados.get('ok', 0)}")
    log.info(f"Erros 403: {resultados.get('erro_403', 0)}")
    log.info(f"Timeouts : {resultados.get('erro_timeout', 0)}")
    log.info(f"Outros   : {resultados.get('erro_outro', 0)}")
    log.info("=" * 55)

    salvar_falhas(falhas)

    if not falhas:
        log.info("Todos os PDFs baixados com sucesso!")
    else:
        log.info(f"{len(falhas)} falharam. Rode novamente.")


if __name__ == "__main__":
    main()