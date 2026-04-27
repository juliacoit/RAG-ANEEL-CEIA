"""
query_optimizer.py — otimizacao de queries com historico, deteccao de contexto e esclarecimento
"""
import json
import logging
import re
import time
from dataclasses import dataclass, field
from groq import Groq

log = logging.getLogger(__name__)

ANOS_BASE = ["2016", "2021", "2022"]
TIPOS_ATO = {"DSP", "REN", "REH", "REA", "PRT", "RES", "DEC", "INA"}
MAX_ESCLARECIMENTOS = 3  # limite de perguntas de esclarecimento por contexto

VOCABULARIO = (
    "TUSD, TUST, PCH, UHE, CGH, UFV, UTN, EOL, microgeracao distribuida, "
    "minigeracao distribuida, geracao distribuida, PRODIST, PRORET, ANEEL, ONS, "
    "CCEE, concessionaria, tarifa de energia, tarifa de uso, encargo setorial, "
    "CDE, ESS, EER, PROINFA, PLD, auto de infracao, embargo, penalidade, TFSEE, "
    "concessao, autorizacao, leilao de energia, MCSE, manual de contabilidade."
)

PROMPT_SISTEMA = (
    "Voce e um especialista em regulacao do setor eletrico brasileiro e sistemas RAG.\n\n"
    "Base: atos normativos ANEEL (DSP=Despacho, REN=Resolucao Normativa, REH=Resolucao Homologatoria, "
    "REA=Resolucao Autorizativa, PRT=Portaria). Anos: 2016, 2021, 2022. "
    "Arquivos: PDFs, XLSX tarifarios, HTMLs.\n\n"
    "Vocabulario tecnico: " + VOCABULARIO + "\n\n"
    "INSTRUCAO: Dado uma pergunta e historico opcional, retorne APENAS JSON valido:\n"
    '{"tipo_pergunta":"<resumo|comparacao|tabela|busca|especifica>",'
    '"filtros":{"ano":"<ano>","tipo_codigo":"<DSP|REN|REH|REA|PRT>","numero":"<numero>"},'
    '"query_reescrita":"<query expandida com termos tecnicos>",'
    '"hyde_texto":"<3-5 linhas simulando trecho real do documento respondendo a pergunta>",'
    '"sub_queries":["<sub1>","<sub2>"],'
    '"termos_chave":["<t1>","<t2>"],'
    '"requer_multiplos_docs":false,'
    '"ambigua":false,'
    '"contexto_mudou":false,'
    '"pergunta_esclarecimento":"<pergunta especifica sobre o que falta para responder, ou vazio>"}\n\n'
    "REGRAS:\n"
    "- HISTORICO: Use para resolver pronomes/referencias. Se historico menciona DSP 2488/2022 "
    "e pergunta diz esse despacho, extraia numero=2488, ano=2022 nos filtros.\n"
    "- contexto_mudou: true quando a pergunta atual e sobre tema/documento completamente diferente "
    "dos turnos anteriores do historico. Ex: historico fala de microgeracao e nova pergunta e sobre multas.\n"
    "- ambigua: true quando NAO identificar doc ou tema especifico mesmo com historico.\n"
    "- pergunta_esclarecimento: quando ambigua=true, gere uma pergunta ESPECIFICA sobre o que falta. "
    "Ex: 'Sobre qual ato normativo voce quer saber a multa? Ex: REN 846, Despacho 2488/2022' "
    "Em vez de bullets genericos. Deixe vazio quando ambigua=false.\n"
    "- filtros: so inclua campos extraiveis com certeza. Omita incertos.\n"
    "- ano: string se um ano, lista se multiplos.\n"
    "- query_reescrita: para comparacoes adicione revogado/altera/substitui. "
    "Para tabelas adicione valor/percentual/tabela. Substitua pronomes pelos docs referenciados.\n"
    "- hyde_texto: texto tecnico como trecho real da ANEEL respondendo a pergunta.\n"
    "- sub_queries: max 3, so quando cruzar multiplos docs.\n"
    "- Tipos: resumo=explicacao, comparacao=revogacao/alteracao, tabela=valores/tarifas, "
    "especifica=artigo/inciso, busca=tema geral."
)

PROMPT_MESCLAR = (
    "Voce e um especialista em regulacao do setor eletrico brasileiro.\n"
    "Dadas as respostas anteriores sobre o mesmo tema e a pergunta atual, "
    "gere uma query de busca otimizada que aproveite o contexto ja respondido "
    "para encontrar informacoes complementares e nao repetidas.\n"
    "Retorne APENAS a query de busca em texto simples, sem explicacoes."
)


@dataclass
class QueryOtimizada:
    query_original:        str
    queries:               list
    hyde_texto:            str
    filtros:               dict
    tipo_pergunta:         str
    sub_queries:           list
    termos_chave:          list
    requer_multiplos_docs: bool
    ambigua:               bool = False
    contexto_mudou:        bool = False
    pergunta_esclarecimento: str = ""
    latencia_ms:           int = 0


def _formatar_historico(historico):
    if not historico:
        return ""
    linhas = ["Historico da conversa (use para resolver referencias):"]
    for i, t in enumerate(historico[-5:], 1):
        linhas.append("[%d] Usuario: %s" % (i, str(t.get("pergunta", ""))[:200]))
        linhas.append("[%d] Sistema: %s" % (i, str(t.get("resposta", ""))[:300]))
    return "\n".join(linhas)


def _respostas_mesmo_contexto(historico):
    """Retorna respostas dos ultimos turnos do mesmo contexto."""
    if not historico:
        return ""
    respostas = []
    for t in historico[-3:]:
        r = str(t.get("resposta", "")).strip()
        if r and r != "Nao encontrado nos atos normativos consultados.":
            respostas.append(r[:400])
    if not respostas:
        return ""
    return "Respostas anteriores do mesmo contexto:\n" + "\n---\n".join(respostas)


def _validar_filtros(raw):
    filtros = {}
    for campo, valor in (raw or {}).items():
        if not valor or valor in ("", [], None, "null"):
            continue
        if campo == "ano":
            if isinstance(valor, str) and valor in ANOS_BASE:
                filtros[campo] = valor
            elif isinstance(valor, list):
                validos = [a for a in valor if a in ANOS_BASE]
                if validos:
                    filtros[campo] = validos
        elif campo == "tipo_codigo" and valor in TIPOS_ATO:
            filtros[campo] = valor
        elif campo == "numero" and str(valor).isdigit():
            filtros[campo] = str(valor)
    return filtros


def _gerar_query_mesclada(pergunta, historico, client, modelo):
    """
    Gera query mesclando contexto das respostas anteriores com a pergunta atual.
    Evita buscar informacoes ja respondidas.
    """
    contexto_anterior = _respostas_mesmo_contexto(historico)
    if not contexto_anterior:
        return None
    try:
        resp = client.chat.completions.create(
            model=modelo,
            messages=[
                {"role": "system", "content": PROMPT_MESCLAR},
                {"role": "user", "content": (
                    contexto_anterior + "\n\nPergunta atual: " + pergunta
                )},
            ],
            temperature=0.1,
            max_tokens=150,
            timeout=8,
        )
        query_m = resp.choices[0].message.content.strip()
        if query_m and len(query_m) > 10:
            log.info("Query mesclada gerada: %s", query_m[:100])
            return query_m
    except Exception as e:
        log.warning("Erro ao gerar query mesclada: %s", e)
    return None


def otimizar_query(pergunta, client, modelo="llama-3.3-70b-versatile",
                   historico=None, n_esclarecimentos=0):
    """
    Otimiza a query usando LLM com suporte a:
      - Historico de conversa (resolve referencias como esse despacho)
      - Deteccao de mudanca de contexto (descarta historico irrelevante)
      - Esclarecimento dinamico e especifico (max 3 tentativas)
      - Mesclagem de respostas anteriores do mesmo contexto

    Parametros:
      pergunta          — pergunta do usuario
      client            — cliente Groq
      historico         — lista [{"pergunta": ..., "resposta": ...}]
      n_esclarecimentos — quantas vezes ja pediu esclarecimento neste contexto
    """
    inicio = time.monotonic()
    hist_str = _formatar_historico(historico)
    conteudo = (hist_str + "\n\nPergunta atual: " + pergunta) if hist_str else ("Pergunta: " + pergunta)

    dados = {}
    try:
        resp = client.chat.completions.create(
            model=modelo,
            messages=[
                {"role": "system", "content": PROMPT_SISTEMA},
                {"role": "user", "content": conteudo},
            ],
            temperature=0.1,
            max_tokens=600,
            timeout=15,
        )
        raw = resp.choices[0].message.content.strip()
        raw = re.sub(r"^```json\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)
        dados = json.loads(raw)
    except json.JSONDecodeError as e:
        log.warning("JSON invalido do otimizador: %s", e)
    except Exception as e:
        log.warning("Erro no otimizador: %s", e)

    lat = int((time.monotonic() - inicio) * 1000)

    filtros = _validar_filtros(dados.get("filtros", {}))
    contexto_mudou = bool(dados.get("contexto_mudou", False))
    ambigua = bool(dados.get("ambigua", False))
    pergunta_escl = str(dados.get("pergunta_esclarecimento", "")).strip()

    # Se contexto mudou, ignora historico para nao contaminar
    historico_efetivo = [] if contexto_mudou else (historico or [])
    if contexto_mudou:
        log.info("Contexto mudou — historico descartado para esta busca")

    # Limite de esclarecimentos atingido: busca com o que tem
    if ambigua and n_esclarecimentos >= MAX_ESCLARECIMENTOS:
        log.info("Limite de %d esclarecimentos atingido — buscando com query original", MAX_ESCLARECIMENTOS)
        ambigua = False
        pergunta_escl = ""

    # Monta queries — inclui query mesclada se mesmo contexto
    query_r = str(dados.get("query_reescrita", "")).strip() or pergunta
    subs = [q for q in dados.get("sub_queries", []) if q and str(q).strip()]
    todas_queries = [query_r] + subs

    # Adiciona query mesclada com contexto anterior (mesmo contexto, nao ambigua)
    if not ambigua and not contexto_mudou and historico:
        query_mesclada = _gerar_query_mesclada(pergunta, historico_efetivo, client, modelo)
        if query_mesclada and query_mesclada not in todas_queries:
            todas_queries.append(query_mesclada)

    hyde = str(dados.get("hyde_texto", "")).strip() or query_r

    log.info("Otimizado em %dms | tipo=%s | ambigua=%s | contexto_mudou=%s | filtros=%s",
             lat, dados.get("tipo_pergunta", "busca"), ambigua, contexto_mudou, filtros)

    return QueryOtimizada(
        query_original=pergunta,
        queries=todas_queries,
        hyde_texto=hyde,
        filtros=filtros,
        tipo_pergunta=dados.get("tipo_pergunta", "busca"),
        sub_queries=subs,
        termos_chave=dados.get("termos_chave", []),
        requer_multiplos_docs=bool(dados.get("requer_multiplos_docs", False)),
        ambigua=ambigua,
        contexto_mudou=contexto_mudou,
        pergunta_esclarecimento=pergunta_escl,
        latencia_ms=lat,
    )


if __name__ == "__main__":
    import os
    from dotenv import load_dotenv
    load_dotenv()
    logging.basicConfig(level=logging.INFO)
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))

    hist = [
        {"pergunta": "O que diz o Despacho 2488 de 2022?",
         "resposta": "O Despacho 2488/2022 trata de dosimetria de multa com referencia a REN 846."},
    ]

    casos = [
        ("Qual o valor maximo da multa nesse despacho?", hist, 0),
        ("Existe despacho de 2021 revogado em 2022?", None, 0),
        ("O que ele estabelece?", None, 0),           # ambigua sem historico
        ("O que ele estabelece?", None, 3),           # limite atingido — busca mesmo assim
        ("Quantas tarifas existem?", hist, 0),        # contexto mudou
    ]

    for pergunta, h, n_escl in casos:
        print("\n" + "="*55)
        print("Q:", pergunta, "| esclarecimentos:", n_escl)
        r = otimizar_query(pergunta, client, historico=h, n_esclarecimentos=n_escl)
        print("Tipo:", r.tipo_pergunta)
        print("Filtros:", r.filtros)
        print("Ambigua:", r.ambigua, "| Contexto mudou:", r.contexto_mudou)
        print("Esclarecimento:", r.pergunta_esclarecimento or "(nenhum)")
        print("Queries:", r.queries)