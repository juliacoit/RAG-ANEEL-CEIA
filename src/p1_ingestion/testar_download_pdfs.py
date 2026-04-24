"""
testar_download_pdfs.py
=======================
Pessoa 1 — Data Engineer

Testa múltiplas abordagens para baixar PDFs da ANEEL,
que tem proteção Cloudflare. Roda as abordagens em ordem
e para na primeira que funcionar.

Uso:
  pip install curl_cffi cloudscraper requests aiohttp
  python src/p1_ingestion/testar_download_pdfs.py

  # Testar só uma abordagem específica:
  python src/p1_ingestion/testar_download_pdfs.py --abordagem https
  python src/p1_ingestion/testar_download_pdfs.py --abordagem curl_cffi
  python src/p1_ingestion/testar_download_pdfs.py --abordagem cloudscraper
"""

import argparse
import time
import ssl
import urllib.request
from pathlib import Path

# URLs de teste — 1 de cada ano (HTTP e HTTPS)
URLS_TESTE = {
    "2016_http":  "http://www2.aneel.gov.br/cedoc/dsp20163284.pdf",
    "2016_https": "https://www2.aneel.gov.br/cedoc/dsp20163284.pdf",
    "2021_https": "https://www2.aneel.gov.br/cedoc/dsp20214137ti.pdf",
    "2022_https": "https://www2.aneel.gov.br/cedoc/dsp20223683ti.pdf",
}

PASTA_TESTE = Path("pdfs/teste")
PASTA_TESTE.mkdir(parents=True, exist_ok=True)

HEADERS_CHROME = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Accept": "application/pdf,*/*",
    "Accept-Language": "pt-BR,pt;q=0.9,en;q=0.8",
    "Accept-Encoding": "gzip, deflate, br",
    "Referer": "https://www2.aneel.gov.br/",
    "Connection": "keep-alive",
}


def normalizar_url(url: str) -> str:
    """Converte HTTP para HTTPS."""
    return url.replace("http://", "https://")


def salvar_se_pdf(conteudo: bytes, nome: str) -> bool:
    """Verifica se é um PDF real (não página de bloqueio) e salva."""
    # PDFs começam com %PDF
    if conteudo[:4] == b'%PDF':
        path = PASTA_TESTE / nome
        path.write_bytes(conteudo)
        print(f"    Salvo em: {path} ({len(conteudo)/1024:.1f} KB)")
        return True
    else:
        # Provavelmente HTML de bloqueio do Cloudflare
        preview = conteudo[:200].decode('utf-8', errors='replace')
        print(f"    Recebeu HTML em vez de PDF: {preview[:100]}...")
        return False


# ---------------------------------------------------------------------------
# Abordagem 1 — HTTPS simples (urllib)
# ---------------------------------------------------------------------------

def testar_https_simples(url: str) -> bool:
    """Testa com HTTPS + headers de Chrome, sem biblioteca extra."""
    print("\n[1] HTTPS simples (urllib + headers Chrome)")
    url = normalizar_url(url)
    print(f"    URL: {url}")

    try:
        ctx = ssl.create_default_context()
        req = urllib.request.Request(url, headers=HEADERS_CHROME)
        with urllib.request.urlopen(req, timeout=15, context=ctx) as r:
            print(f"    Status: {r.status} — Content-Type: {r.headers.get('Content-Type')}")
            if r.status == 200:
                return salvar_se_pdf(r.read(), "teste_urllib.pdf")
            return False
    except Exception as e:
        print(f"    Erro: {type(e).__name__}: {e}")
        return False


# ---------------------------------------------------------------------------
# Abordagem 2 — curl_cffi (imita fingerprint TLS do Chrome)
# ---------------------------------------------------------------------------

def testar_curl_cffi(url: str) -> bool:
    """
    curl_cffi imita o fingerprint TLS exato do Chrome.
    O Cloudflare verifica a 'assinatura' da conexão TLS —
    requests/aiohttp têm fingerprints de curl, não de Chrome.
    """
    print("\n[2] curl_cffi (fingerprint TLS do Chrome)")
    url = normalizar_url(url)
    print(f"    URL: {url}")

    try:
        from curl_cffi import requests as curl_requests
        resp = curl_requests.get(
            url,
            impersonate="chrome124",  # imita Chrome 124 exato
            headers=HEADERS_CHROME,
            timeout=15,
            verify=True,
        )
        print(f"    Status: {resp.status_code}")
        if resp.status_code == 200:
            return salvar_se_pdf(resp.content, "teste_curl_cffi.pdf")
        return False
    except ImportError:
        print("    curl_cffi não instalado. Rode: pip install curl_cffi")
        return False
    except Exception as e:
        print(f"    Erro: {type(e).__name__}: {e}")
        return False


# ---------------------------------------------------------------------------
# Abordagem 3 — cloudscraper
# ---------------------------------------------------------------------------

def testar_cloudscraper(url: str) -> bool:
    """
    cloudscraper resolve o challenge JavaScript do Cloudflare
    automaticamente — simula o navegador executando o JS.
    """
    print("\n[3] cloudscraper (resolve JS challenge do Cloudflare)")
    url = normalizar_url(url)
    print(f"    URL: {url}")

    try:
        import cloudscraper
        scraper = cloudscraper.create_scraper(
            browser={"browser": "chrome", "platform": "windows", "mobile": False}
        )
        resp = scraper.get(url, timeout=30)
        print(f"    Status: {resp.status_code}")
        if resp.status_code == 200:
            return salvar_se_pdf(resp.content, "teste_cloudscraper.pdf")
        return False
    except ImportError:
        print("    cloudscraper não instalado. Rode: pip install cloudscraper")
        return False
    except Exception as e:
        print(f"    Erro: {type(e).__name__}: {e}")
        return False


# ---------------------------------------------------------------------------
# Abordagem 4 — requests com sessão e cookies manuais
# ---------------------------------------------------------------------------

def testar_requests_sessao(url: str) -> bool:
    """
    Abre o site principal primeiro para pegar cookies do Cloudflare,
    depois tenta baixar o PDF com esses cookies na sessão.
    """
    print("\n[4] requests com sessão (pega cookies do site principal)")
    url = normalizar_url(url)

    try:
        import requests
        session = requests.Session()
        session.headers.update(HEADERS_CHROME)

        # Passo 1: visita o site principal para pegar cookies
        print("    Visitando site principal para pegar cookies...")
        r1 = session.get("https://www2.aneel.gov.br/", timeout=15, verify=False)
        print(f"    Site principal: {r1.status_code} — cookies: {list(session.cookies.keys())}")

        time.sleep(2)  # pausa humana

        # Passo 2: tenta baixar o PDF com os cookies
        print(f"    Baixando PDF: {url}")
        r2 = session.get(url, timeout=15, verify=False)
        print(f"    Status: {r2.status_code}")

        if r2.status_code == 200:
            return salvar_se_pdf(r2.content, "teste_requests_sessao.pdf")
        return False
    except ImportError:
        print("    requests não instalado. Rode: pip install requests")
        return False
    except Exception as e:
        print(f"    Erro: {type(e).__name__}: {e}")
        return False


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Testa abordagens de download de PDFs da ANEEL"
    )
    parser.add_argument(
        "--abordagem",
        choices=["todas", "https", "curl_cffi", "cloudscraper", "sessao"],
        default="todas",
        help="Qual abordagem testar (padrão: todas em sequência)",
    )
    parser.add_argument(
        "--url",
        default=URLS_TESTE["2016_http"],
        help="URL específica para testar",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("TESTE DE DOWNLOAD — PDFs ANEEL")
    print("=" * 60)
    print(f"URL base: {args.url}")
    print(f"Resultados em: {PASTA_TESTE}/")

    abordagens = {
        "https":       testar_https_simples,
        "curl_cffi":   testar_curl_cffi,
        "cloudscraper": testar_cloudscraper,
        "sessao":      testar_requests_sessao,
    }

    if args.abordagem == "todas":
        for nome, func in abordagens.items():
            ok = func(args.url)
            if ok:
                print(f"\n✅ SUCESSO com abordagem: {nome}")
                print(f"   PDF salvo em {PASTA_TESTE}/")
                print(f"\n   Próximo passo: atualizar baixar_pdfs_aneel.py para usar '{nome}'")
                return
            time.sleep(1)

        print("\n❌ Todas as abordagens falharam.")
        print("   Opções restantes:")
        print("   1. VPN residencial (não de datacenter)")
        print("   2. Solicitar acesso formal à ANEEL")
        print("   3. Download manual pelo navegador + cookies exportados")
    else:
        func = abordagens[args.abordagem]
        ok = func(args.url)
        if ok:
            print(f"\n✅ SUCESSO!")
        else:
            print(f"\n❌ Falhou.")


if __name__ == "__main__":
    main()