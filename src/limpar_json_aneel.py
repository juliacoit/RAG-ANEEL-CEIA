"""
limpar_json_aneel.py
====================
Pessoa 1 — Data Engineer · Sprint Dia 1 (manhã)

O que faz:
  - Lê o JSON bruto da biblioteca ANEEL
  - Limpa prefixos verbosos ("Situação:", "Assunto:", "Assinatura:", etc.)
  - Extrai tipo e número do documento do campo título
  - Converte datas para formato ISO (YYYY-MM-DD)
  - Marca vigência de cada documento (vigente: true/false)
  - Categoriza os PDFs (texto_integral, voto, nota_tecnica, decisao, anexo)
  - Remove artefato " Imprimir" do final das ementas
  - Gera dois arquivos de saída:
      aneel_limpo.json      → todos os registros
      aneel_vigentes.json   → apenas documentos vigentes (filtro padrão do RAG)

Uso:
  pip install tqdm
  python limpar_json_aneel.py

  Ou com caminho customizado:
  python limpar_json_aneel.py --input meu_arquivo.json --output pasta_saida/
"""

import json
import re
import argparse
from datetime import datetime
from pathlib import Path

try:
    from tqdm import tqdm
    USA_TQDM = True
except ImportError:
    USA_TQDM = False
    print("Dica: instale tqdm para barra de progresso  →  pip install tqdm")


# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------

SITUACOES_INATIVAS = {"REVOGADA", "TORNADA SEM EFEITO", "ANULADA", "CADUCADA"}

TIPO_DOC = {
    "DSP": "Despacho",
    "PRT": "Portaria",
    "REA": "Resolução Autorizativa",
    "REH": "Resolução Homologatória",
    "ECT": "Extrato de Contrato",
    "AAP": "Aviso de Audiência Pública",
    "REN": "Resolução Normativa",
    "EDT": "Errata",
    "AVS": "Aviso",
    "COM": "Comunicado",
    "ACP": "Aviso de Consulta Pública",
    "REO": "Resolução de Outorga",
    "INA": "Instrução Normativa",
    "LEL": "Lei",
    "OFC": "Ofício",
    "DEC": "Decreto",
    "ECP": "Extrato de Consulta Pública",
    "PRI": "Portaria Interministerial",
    "RES": "Resolução",
}


# ---------------------------------------------------------------------------
# Funções de limpeza
# ---------------------------------------------------------------------------

def remover_prefixo(valor: str | None, prefixo: str) -> str | None:
    """Remove 'Prefixo:' do início de um campo e faz strip."""
    if not valor:
        return None
    return valor.replace(f"{prefixo}:", "").strip()


def parsear_data(valor: str | None) -> str | None:
    """Converte 'Assinatura:15/12/2016' → '2016-12-15'. Retorna None se inválido."""
    if not valor:
        return None
    # Remove qualquer prefixo (Assinatura:, Publicação:, etc.)
    raw = re.sub(r'^[^:]+:', '', valor).strip()
    try:
        return datetime.strptime(raw, "%d/%m/%Y").strftime("%Y-%m-%d")
    except ValueError:
        return raw or None  # devolve o que vier se não conseguir parsear


def limpar_ementa(ementa: str | None) -> str | None:
    """Remove artefatos de scraping da ementa."""
    if not ementa:
        return None
    return (
        ementa
        .replace(" Imprimir", "")
        .replace("Imprimir", "")
        .strip()
    )


def categorizar_pdf(tipo_raw: str | None) -> str:
    """
    Classifica o tipo do PDF em categoria padronizada.

    Retorna um de: texto_integral | voto | nota_tecnica | decisao | anexo | outro
    """
    t = (tipo_raw or "").strip().lower().rstrip(":")
    if t.startswith("texto integral"):
        return "texto_integral"
    if t.startswith("voto"):
        return "voto"
    if t.startswith("nota técnica") or t.startswith("nota tecnica"):
        return "nota_tecnica"
    if t.startswith("decisão") or t.startswith("decisao"):
        return "decisao"
    if t.startswith("anexo"):
        return "anexo"
    return "outro"


def extrair_tipo_e_numero(titulo: str | None) -> tuple[str, str | None, str | None]:
    """
    Extrai (tipo_codigo, numero, ano) do título.

    Exemplos:
      'DSP - DESPACHO 3284/2016' → ('DSP', '3284', '2016')
      'PRT - PORTARIA 383/2015'  → ('PRT', '383', '2015')
    """
    if not titulo:
        return ("", None, None)

    # Tipo = tudo antes do primeiro ' - '
    tipo_codigo = titulo.split(" - ")[0].strip() if " - " in titulo else titulo.split()[0]

    # Número e ano: primeiro padrão NNNN/AAAA encontrado
    m = re.search(r"(\d+)/(\d{4})", titulo)
    if m:
        return (tipo_codigo, m.group(1), m.group(2))
    return (tipo_codigo, None, None)


def gerar_id(tipo_codigo: str, numero: str | None, titulo: str) -> str:
    """Gera um ID único e legível para o registro."""
    if numero:
        return f"{tipo_codigo}_{numero}"
    # Fallback: slug do título sem espaços
    return re.sub(r"[^a-zA-Z0-9_]", "_", titulo)[:60]


# ---------------------------------------------------------------------------
# Processamento principal
# ---------------------------------------------------------------------------

def limpar_registro(reg: dict, data_publicacao: str) -> dict:
    """Transforma um registro bruto em registro limpo."""

    titulo = reg.get("titulo") or ""
    tipo_codigo, numero, ano = extrair_tipo_e_numero(titulo)

    situacao_raw = remover_prefixo(reg.get("situacao"), "Situação")
    vigente = situacao_raw not in SITUACOES_INATIVAS if situacao_raw else True

    pdfs_limpos = []
    for p in reg.get("pdfs") or []:
        pdfs_limpos.append({
            "categoria": categorizar_pdf(p.get("tipo")),
            "arquivo": p.get("arquivo"),
            "url": p.get("url"),
            "baixado": p.get("baixado", False),
        })

    return {
        # Identificação
        "id": gerar_id(tipo_codigo, numero, titulo),
        "titulo": titulo,
        "tipo_codigo": tipo_codigo,
        "tipo_nome": TIPO_DOC.get(tipo_codigo, tipo_codigo),
        "numero": numero,
        "ano": ano,

        # Autoria e classificação
        "autor": reg.get("autor"),
        "assunto": remover_prefixo(reg.get("assunto"), "Assunto"),

        # Vigência jurídica  ← filtro mais importante do RAG
        "situacao": situacao_raw,
        "vigente": vigente,

        # Datas
        "data_assinatura": parsear_data(reg.get("assinatura")),
        "data_publicacao": data_publicacao,

        # Conteúdo textual
        "ementa": limpar_ementa(reg.get("ementa")),
        "tem_ementa": bool(reg.get("ementa")),

        # Arquivos vinculados
        "pdfs": pdfs_limpos,
        "n_pdfs": len(pdfs_limpos),
    }


def processar(caminho_entrada: Path) -> list[dict]:
    """Lê o JSON bruto e retorna lista de registros limpos."""
    print(f"Lendo {caminho_entrada} ...")
    with open(caminho_entrada, encoding="utf-8") as f:
        dados_brutos = json.load(f)

    registros_limpos = []
    dias = list(dados_brutos.items())

    iterador = tqdm(dias, desc="Processando dias", unit="dia") if USA_TQDM else dias

    for data_pub, dia in iterador:
        for reg in dia.get("registros") or []:
            registros_limpos.append(limpar_registro(reg, data_pub))

    return registros_limpos


def salvar(registros: list[dict], pasta: Path) -> None:
    """Salva aneel_limpo.json e aneel_vigentes.json."""
    pasta.mkdir(parents=True, exist_ok=True)

    # Todos os registros
    caminho_todos = pasta / "aneel_limpo.json"
    with open(caminho_todos, "w", encoding="utf-8") as f:
        json.dump(registros, f, ensure_ascii=False, indent=2)
    print(f"Salvo: {caminho_todos}  ({len(registros)} registros)")

    # Apenas vigentes
    vigentes = [r for r in registros if r["vigente"]]
    caminho_vigentes = pasta / "aneel_vigentes.json"
    with open(caminho_vigentes, "w", encoding="utf-8") as f:
        json.dump(vigentes, f, ensure_ascii=False, indent=2)
    print(f"Salvo: {caminho_vigentes}  ({len(vigentes)} registros vigentes)")


def imprimir_resumo(registros: list[dict]) -> None:
    """Imprime um resumo de validação no terminal."""
    total = len(registros)
    vigentes = sum(1 for r in registros if r["vigente"])
    com_ementa = sum(1 for r in registros if r["tem_ementa"])
    sem_numero = sum(1 for r in registros if not r["numero"])

    from collections import Counter
    tipos = Counter(r["tipo_codigo"] for r in registros)

    print("\n" + "=" * 50)
    print("RESUMO DA LIMPEZA")
    print("=" * 50)
    print(f"  Total de registros   : {total}")
    print(f"  Vigentes             : {vigentes} ({100*vigentes/total:.1f}%)")
    print(f"  Com ementa           : {com_ementa} ({100*com_ementa/total:.1f}%)")
    print(f"  Sem número extraído  : {sem_numero} (verificar manualmente)")
    print(f"\n  Top tipos de documento:")
    for tipo, qtd in tipos.most_common(6):
        nome = TIPO_DOC.get(tipo, tipo)
        print(f"    {tipo:6s} ({nome:30s}): {qtd}")
    print("=" * 50 + "\n")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Limpa o JSON de metadados da ANEEL")
    parser.add_argument(
        "--input", "-i",
        default="biblioteca_aneel_gov_br_legislacao_2016_metadados.json",
        help="Caminho para o JSON bruto (padrão: arquivo original)"
    )
    parser.add_argument(
        "--output", "-o",
        default="output/",
        help="Pasta de saída (padrão: output/)"
    )
    args = parser.parse_args()

    caminho_entrada = Path(args.input)
    pasta_saida = Path(args.output)

    if not caminho_entrada.exists():
        print(f"ERRO: arquivo não encontrado → {caminho_entrada}")
        return

    registros = processar(caminho_entrada)
    imprimir_resumo(registros)
    salvar(registros, pasta_saida)
    print("Pronto! Compartilhe a pasta output/ com P2 e P3.")


if __name__ == "__main__":
    main()
