"""
analytics.py
============
Análise agregada sobre metadados dos atos normativos da ANEEL.

Responde perguntas que requerem contagem, agrupamento e ranking —
operações que o RAG não consegue fazer (busca semântica não agrega dados).

Exemplos de perguntas suportadas:
  - "Qual mês de 2016 teve mais despachos revogados?"
  - "Quais foram os despachos revogados em março de 2016?"
  - "Quantos atos foram publicados por tipo em 2021?"
  - "Qual autor emitiu mais atos em 2022?"

Uso:
  from src.api.analytics import responder_analitico
  resposta = responder_analitico("qual mês de 2016 teve mais despachos revogados")
"""

import json
import logging
import os
import re
from pathlib import Path

import pandas as pd

log = logging.getLogger(__name__)

RAIZ      = Path(__file__).resolve().parent.parent.parent
JSON_PATH = RAIZ / "data" / "aneel_limpo_completo.json"

MESES_PT = {
    1: "Janeiro", 2: "Fevereiro", 3: "Março",    4: "Abril",
    5: "Maio",    6: "Junho",     7: "Julho",     8: "Agosto",
    9: "Setembro",10: "Outubro",  11: "Novembro", 12: "Dezembro",
}

MESES_NOME = {v.lower(): k for k, v in MESES_PT.items()}
MESES_NOME.update({"marco": 3, "marc": 3})  # sem acento

SITUACOES_INATIVAS = {"REVOGADA", "TORNADA SEM EFEITO", "ANULADA", "CADUCADA", "SUSPENSA"}

# Cache do DataFrame
_df_cache: pd.DataFrame | None = None


def _carregar_df() -> pd.DataFrame:
    """Carrega e cacheia o JSON como DataFrame."""
    global _df_cache
    if _df_cache is not None:
        return _df_cache

    if not JSON_PATH.exists():
        raise FileNotFoundError(f"JSON não encontrado: {JSON_PATH}")

    log.info(f"Carregando {JSON_PATH.name} para análise...")
    with open(JSON_PATH, encoding="utf-8") as f:
        dados = json.load(f)

    df = pd.DataFrame(dados)

    # Garante colunas necessárias
    for col in ["situacao", "tipo_codigo", "tipo_nome", "ano", "numero",
                "autor", "data_publicacao", "data_assinatura", "assunto"]:
        if col not in df.columns:
            df[col] = ""

    # Converte datas
    df["data_pub"] = pd.to_datetime(df["data_publicacao"], errors="coerce")
    df["mes"]      = df["data_pub"].dt.month
    df["mes_nome"] = df["mes"].map(MESES_PT)
    df["ano_num"]  = pd.to_numeric(df["ano"], errors="coerce")
    df["inativo"]  = df["situacao"].isin(SITUACOES_INATIVAS)

    _df_cache = df
    log.info(f"DataFrame carregado: {len(df)} registros")
    return df


# ---------------------------------------------------------------------------
# Detecção do tipo de pergunta analítica
# ---------------------------------------------------------------------------

def _detectar_intent(pergunta: str) -> dict:
    """
    Extrai parâmetros da pergunta analítica:
      ano, mes, tipo_ato, situacao, modo (ranking|listagem|contagem)
    """
    p = pergunta.lower()

    # Ano
    m = re.search(r"\b(2016|2021|2022)\b", p)
    ano = int(m.group(1)) if m else None

    # Mês por nome
    mes = None
    for nome, num in MESES_NOME.items():
        if nome in p:
            mes = num
            break
    # Mês por número
    if not mes:
        m = re.search(r"\bm[eê]s\s+(\d{1,2})\b", p)
        if m:
            mes = int(m.group(1))

    # Tipo de ato
    tipo = None
    if any(t in p for t in ["despacho", "dsp"]):
        tipo = "DSP"
    elif any(t in p for t in ["resolução normativa", "ren"]):
        tipo = "REN"
    elif any(t in p for t in ["resolução homologatória", "reh"]):
        tipo = "REH"
    elif any(t in p for t in ["resolução autorizativa", "rea"]):
        tipo = "REA"
    elif any(t in p for t in ["portaria", "prt"]):
        tipo = "PRT"

    # Situação
    situacao = None
    if any(t in p for t in ["revogado", "revogados", "revogada"]):
        situacao = "REVOGADA"
    elif any(t in p for t in ["tornada sem efeito"]):
        situacao = "TORNADA SEM EFEITO"
    elif any(t in p for t in ["anulado", "anulados", "anulada"]):
        situacao = "ANULADA"
    elif any(t in p for t in ["suspenso", "suspensos", "suspensa"]):
        situacao = "SUSPENSA"
    elif any(t in p for t in ["inativo", "inativos", "não vigente"]):
        situacao = "INATIVO"  # qualquer situação inativa

    # Modo
    modo = "contagem"
    if any(t in p for t in ["quais foram", "quais são", "liste", "listar", "mostre", "mostra"]):
        modo = "listagem"
    elif any(t in p for t in ["qual mês", "qual mes", "qual autor", "qual tipo", "maior", "mais"]):
        modo = "ranking"
    elif any(t in p for t in ["quantos", "quantas", "total", "quantidade"]):
        modo = "contagem"

    # Agrupamento
    agrupar = "mes"
    if any(t in p for t in ["autor", "quem emitiu", "quem publicou"]):
        agrupar = "autor"
    elif any(t in p for t in ["tipo", "por tipo"]):
        agrupar = "tipo"
    elif any(t in p for t in ["assunto"]):
        agrupar = "assunto"

    return {
        "ano": ano, "mes": mes, "tipo": tipo,
        "situacao": situacao, "modo": modo, "agrupar": agrupar,
    }


# ---------------------------------------------------------------------------
# Análises
# ---------------------------------------------------------------------------

def _aplicar_filtros(df: pd.DataFrame, intent: dict) -> pd.DataFrame:
    """Aplica filtros de ano, mês, tipo e situação."""
    if intent["ano"]:
        df = df[df["ano_num"] == intent["ano"]]
    if intent["mes"]:
        df = df[df["mes"] == intent["mes"]]
    if intent["tipo"]:
        df = df[df["tipo_codigo"] == intent["tipo"]]
    if intent["situacao"] == "INATIVO":
        df = df[df["inativo"]]
    elif intent["situacao"]:
        df = df[df["situacao"] == intent["situacao"]]
    return df


def _ranking(df: pd.DataFrame, agrupar: str, top: int = 5) -> str:
    """Retorna ranking por campo agrupado."""
    col_map = {
        "mes":    ("mes_nome", "Mês"),
        "autor":  ("autor",    "Autor"),
        "tipo":   ("tipo_nome","Tipo"),
        "assunto":("assunto",  "Assunto"),
    }
    col, label = col_map.get(agrupar, ("mes_nome", "Mês"))

    contagem = df[col].value_counts().head(top)
    if contagem.empty:
        return "Nenhum registro encontrado com esses critérios."

    linhas = [f"| {label} | Quantidade |", "|---|---|"]
    for val, qtd in contagem.items():
        linhas.append(f"| {val} | {qtd} |")

    primeiro = contagem.index[0]
    return "\n".join(linhas) + f"\n\n**Destaque:** {primeiro} com {contagem.iloc[0]} registros."


def _listagem(df: pd.DataFrame, limite: int = 20) -> str:
    """Lista os atos encontrados."""
    if df.empty:
        return "Nenhum ato encontrado com esses critérios."

    total = len(df)
    df_show = df.head(limite)

    linhas = [f"**Total encontrado: {total} atos**\n"]
    if total > limite:
        linhas.append(f"*(Mostrando os primeiros {limite})*\n")

    for _, row in df_show.iterrows():
        tipo  = row.get("tipo_nome") or row.get("tipo_codigo") or "?"
        num   = row.get("numero") or "?"
        ano   = row.get("ano") or "?"
        data  = str(row.get("data_publicacao") or "")[:10]
        sit   = row.get("situacao") or ""
        assunto = str(row.get("assunto") or "")[:80]

        linha = f"- **{tipo} nº {num}/{ano}**"
        if data:
            linha += f" | publicado: {data}"
        if sit:
            linha += f" | situação: {sit}"
        if assunto:
            linha += f"\n  _{assunto}_"
        linhas.append(linha)

    return "\n".join(linhas)


def _contagem(df: pd.DataFrame, intent: dict) -> str:
    """Retorna contagem simples."""
    total = len(df)
    partes = []
    if intent["tipo"]:
        partes.append(f"{intent['tipo']}")
    if intent["situacao"]:
        partes.append(f"com situação {intent['situacao']}")
    if intent["ano"]:
        partes.append(f"em {intent['ano']}")
    if intent["mes"]:
        partes.append(f"no mês de {MESES_PT.get(intent['mes'], intent['mes'])}")

    descricao = " ".join(partes) if partes else "atos"
    return f"**Total de {descricao}: {total} registros**"


# ---------------------------------------------------------------------------
# Função principal
# ---------------------------------------------------------------------------

def responder_analitico(pergunta: str) -> dict:
    """
    Responde perguntas analíticas sobre metadados dos atos normativos.

    Retorna dict com:
      resposta    — texto formatado em markdown
      tipo        — "analitico"
      intent      — parâmetros extraídos da pergunta
      total       — total de registros encontrados
    """
    try:
        df = _carregar_df()
    except FileNotFoundError as e:
        return {"resposta": str(e), "tipo": "analitico", "total": 0}

    intent = _detectar_intent(pergunta)
    log.info(f"Intent analítico: {intent}")

    df_filtrado = _aplicar_filtros(df.copy(), intent)

    if intent["modo"] == "ranking":
        resposta = _ranking(df_filtrado, intent["agrupar"])
    elif intent["modo"] == "listagem":
        resposta = _listagem(df_filtrado)
    else:
        resposta = _contagem(df_filtrado, intent)

    # Adiciona contexto se ranking por mês
    if intent["modo"] == "ranking" and intent["agrupar"] == "mes" and not df_filtrado.empty:
        mes_top = df_filtrado["mes"].value_counts().idxmax()
        atos_mes = df_filtrado[df_filtrado["mes"] == mes_top]
        resposta += f"\n\nOs {len(atos_mes)} atos de {MESES_PT[mes_top]} incluem:\n"
        for _, row in atos_mes.head(5).iterrows():
            tipo = row.get("tipo_nome") or row.get("tipo_codigo") or "?"
            num  = row.get("numero") or "?"
            ano  = row.get("ano") or "?"
            resposta += f"- {tipo} nº {num}/{ano}\n"
        if len(atos_mes) > 5:
            resposta += f"- *(e mais {len(atos_mes)-5} outros)*\n"

    return {
        "resposta": resposta,
        "tipo":     "analitico",
        "intent":   intent,
        "total":    len(df_filtrado),
    }


def is_pergunta_analitica(pergunta: str) -> bool:
    """
    Detecta se a pergunta é analítica (agregação/contagem)
    ou semântica (busca por conteúdo).
    """
    p = pergunta.lower()
    indicadores = [
        "qual mês", "qual mes", "quantos", "quantas",
        "quais foram os", "liste os", "liste as",
        "total de", "quantidade de", "ranking",
        "mais publicou", "mais emitiu", "mais apresentou",
        "por mês", "por mes", "por ano", "por tipo",
        "maior número", "maior numero",
    ]
    return any(ind in p for ind in indicadores)


# ---------------------------------------------------------------------------
# Teste direto
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    perguntas = [
        "Qual foi o mês em 2016 que mais apresentou despachos revogados?",
        "Quais foram os despachos revogados em dezembro de 2016?",
        "Quantos atos foram revogados em 2021?",
        "Quantas resoluções normativas foram publicadas em 2022?",
        "Qual autor emitiu mais atos em 2016?",
        "Liste os despachos revogados em março de 2016",
    ]

    for p in perguntas:
        print(f"\n{'='*60}")
        print(f"PERGUNTA: {p}")
        resultado = responder_analitico(p)
        print(f"Intent:   {resultado['intent']}")
        print(f"Total:    {resultado['total']}")
        print(f"Resposta:\n{resultado['resposta']}")
