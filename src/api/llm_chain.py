"""
llm_chain.py
============
Pessoa 3 — LLM Engineer

Suporta multiplos provedores de LLM com fallback automatico:
  1. Groq (llama-3.3-70b-versatile) — 100k tokens/dia gratis
  2. DeepSeek R1 (deepseek-reasoner) — via API OpenAI-compativel

Configuracao no .env:
  LLM_PROVIDER=groq          # groq | deepseek
  LLM_MODEL=llama-3.3-70b-versatile
  GROQ_API_KEY=...           # console.groq.com — gratis
  DEEPSEEK_API_KEY=...       # platform.deepseek.com

Fallback automatico:
  Se Groq retornar 429 (rate limit), tenta DeepSeek R1 automaticamente.
  Se DeepSeek falhar, retorna mensagem de erro clara.

Uso direto (teste):
  python src/api/llm_chain.py
"""

import logging
import os
import re
import time
from dataclasses import dataclass
from src.utils.logger_metrics import salvar_log

from dotenv import load_dotenv

load_dotenv()

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuracao
# ---------------------------------------------------------------------------

PROVIDER         = os.getenv("LLM_PROVIDER", "groq").lower()
MODELO           = os.getenv("LLM_MODEL", "llama-3.3-70b-versatile")
MODELO_DEEPSEEK  = os.getenv("DEEPSEEK_MODEL", "deepseek-reasoner")
TEMPERATURA   = 0.1
MAX_TOKENS    = 2048

FALLBACK_MSG         = "Nao encontrado nos atos normativos consultados."
SYSTEM_PROMPT_VERSION = "V4"

# ---------------------------------------------------------------------------
# System prompts por tipo de pergunta
# ---------------------------------------------------------------------------

SYSTEM_PROMPT_BASE = (
    "Voce e um especialista em regulacao do setor eletrico brasileiro com acesso a atos normativos da ANEEL.\n\n"
    "Regras obrigatorias:\n"
    "1. Responda EXCLUSIVAMENTE com base nos trechos fornecidos.\n"
    "2. NUNCA invente, infira ou complete com conhecimento externo.\n"
    "3. Sempre cite o ato normativo de origem (ex: conforme Despacho 1380/2021).\n"
    "4. Se o documento solicitado nao estiver disponivel mas for MENCIONADO em outros atos dos trechos,\n"
    "   informe quais atos o mencionam e em que contexto, sem inventar o conteudo do documento ausente.\n"
    "5. So responda Nao encontrado nos atos normativos consultados. se o tema nao aparecer\n"
    "   de nenhuma forma nos trechos, nem como documento principal nem como mencao.\n"
)

SYSTEM_PROMPT_RESUMO = SYSTEM_PROMPT_BASE + (
    "\nModo: RESUMO\n"
    "- Elabore uma resposta completa e bem desenvolvida.\n"
    "- Organize em paragrafos tematicos quando houver mais de um aspecto.\n"
    "- Nao se limite a uma frase, desenvolva o conteudo encontrado nos trechos.\n"
    "- Se encontrar contexto historico, objetivo, escopo ou definicoes, inclua tudo.\n"
)

SYSTEM_PROMPT_BUSCA = SYSTEM_PROMPT_BASE + (
    "\nModo: BUSCA DE ATOS\n"
    "- Liste todos os atos normativos relevantes encontrados.\n"
    "- Para cada ato: numero, ano, e o que ele estabelece.\n"
    "- Seja direto e objetivo.\n"
)

SYSTEM_PROMPT_COMPARACAO = SYSTEM_PROMPT_BASE + (
    "\nModo: COMPARACAO / RELACAO ENTRE ATOS\n"
    "- Identifique explicitamente se algum ato revoga, altera, complementa ou contradiz outro.\n"
    "- Frases como fica revogado, altera o art., em substituicao a sao indicadores importantes.\n"
    "- Se dois atos tratam do mesmo tema com disposicoes diferentes, aponte a diferenca.\n"
    "- Organize cronologicamente quando relevante.\n"
)

SYSTEM_PROMPT_TABELA = SYSTEM_PROMPT_BASE + (
    "\nModo: DADOS TABULARES\n"
    "- Os trechos podem conter dados de planilhas (valores, percentuais, datas).\n"
    "- Interprete os valores no contexto da pergunta.\n"
    "- Apresente os dados de forma organizada.\n"
    "- Cite a fonte de cada valor encontrado.\n"
    "- Se os valores estiverem em unidades diferentes (R$/kW, R$/MWh), informe a unidade.\n"
)

SYSTEM_PROMPT_MENCAO = SYSTEM_PROMPT_BASE + (
    "\nModo: MENCAO, documento solicitado nao esta disponivel na base\n"
    "- O documento solicitado NAO esta nos trechos fornecidos.\n"
    "- Verifique se algum dos trechos MENCIONA o documento solicitado.\n"
    "- Se encontrar mencoes, informe: qual ato menciona, em que contexto (revogacao, alteracao, referencia).\n"
    "- Deixe claro que o texto completo nao esta disponivel.\n"
    "- NAO invente o conteudo do documento ausente.\n"
)

SYSTEM_PROMPT_ESPECIFICA = SYSTEM_PROMPT_RESUMO


# ---------------------------------------------------------------------------
# Detecta tipo de pergunta
# ---------------------------------------------------------------------------

def _detectar_tipo_pergunta(pergunta):
    p = pergunta.lower()

    if any(t in p for t in ["resumo", "resume", "resuma", "sintetize", "introducao",
                              "prefacio", "o que diz", "do que trata", "explique",
                              "explica", "descreva", "contexto", "historico"]):
        return "resumo"

    if any(t in p for t in ["revoga", "revogou", "cancela", "substitui", "substituiu",
                              "altera", "alterou", "contradiz", "diferenca", "comparar",
                              "relacao entre", "afeta", "anterior", "posterior"]):
        return "comparacao"

    if any(t in p for t in ["valor", "valores", "tarifa", "tarifas", "percentual",
                              "tabela", "planilha", "quanto", "custo", "preco",
                              "taxa", "taxas", "indice", "quantos", "tust", "tusd", "rap",
                              "depreciacao", "depreciação", "gerador", "turbina",
                              "tuc", "tipo de unidade", "amortizacao", "amortização",
                              "tuc", "vida util", "vida útil", "mcpse", "mcse",
                              "aliquota", "alíquota", "unitario", "unitário"]):
        return "tabela"

    if any(t in p for t in ["artigo", "art.", "inciso", "paragrafo", "clausula",
                              "item ", "secao"]):
        return "especifica"

    return "busca"


def _selecionar_prompt(tipo):
    return {
        "resumo":     SYSTEM_PROMPT_RESUMO,
        "comparacao": SYSTEM_PROMPT_COMPARACAO,
        "tabela":     SYSTEM_PROMPT_TABELA,
        "busca":      SYSTEM_PROMPT_BUSCA,
        "mencao":     SYSTEM_PROMPT_MENCAO,
        "especifica": SYSTEM_PROMPT_ESPECIFICA,
    }.get(tipo, SYSTEM_PROMPT_BUSCA)


# ---------------------------------------------------------------------------
# Formatacao do contexto
# ---------------------------------------------------------------------------

def _formatar_contexto(chunks):
    partes = []
    for i, chunk in enumerate(chunks, 1):
        tipo_nome = chunk.get("tipo_nome") or ""
        numero    = chunk.get("numero") or ""
        ano       = chunk.get("ano") or ""
        fonte     = chunk.get("fonte_texto") or ""

        if tipo_nome and numero and ano:
            titulo = "%s n %s/%s" % (tipo_nome, numero, ano)
        else:
            titulo = chunk.get("titulo") or ("Documento %d" % i)

        fonte_label = " [%s]" % fonte if fonte else ""

        partes.append(
            "[Trecho %d] %s%s\n%s" % (i, titulo, fonte_label, chunk.get("texto", "").strip())
        )
    return "\n\n---\n\n".join(partes)


def _formatar_prompt(pergunta, contexto, tipo):
    dica = ""
    if tipo == "resumo":
        dica = "\nDica: desenvolva a resposta com todos os detalhes encontrados nos trechos."
    elif tipo == "comparacao":
        dica = "\nDica: verifique se algum trecho menciona revogacao, alteracao ou substituicao."
    elif tipo == "tabela":
        dica = "\nDica: extraia e organize os valores encontrados, informando a unidade de cada um."

    return "DOCUMENTOS CONSULTADOS:\n\n%s\n\nPERGUNTA: %s%s" % (contexto, pergunta, dica)


# ---------------------------------------------------------------------------
# Estrutura de resposta
# ---------------------------------------------------------------------------

@dataclass
class RespostaLLM:
    texto:           str
    modelo:          str
    tokens_prompt:   int
    tokens_resposta: int
    latencia_ms:     int
    system_prompt:   str


# ---------------------------------------------------------------------------
# Clientes LLM
# ---------------------------------------------------------------------------

_groq_client     = None
_deepseek_client = None


def _get_groq():
    global _groq_client
    if _groq_client is None:
        from groq import Groq
        api_key = os.getenv("GROQ_API_KEY", "").strip()
        if not api_key:
            raise EnvironmentError("GROQ_API_KEY nao encontrada no .env")
        _groq_client = Groq(api_key=api_key)
        log.info("Cliente Groq carregado: %s", MODELO)
    return _groq_client


def _get_deepseek():
    """Cliente DeepSeek R1 via API OpenAI-compativel (openai SDK)."""
    global _deepseek_client
    if _deepseek_client is None:
        from openai import OpenAI
        api_key = os.getenv("DEEPSEEK_API_KEY", "").strip()
        if not api_key:
            raise EnvironmentError(
                "DEEPSEEK_API_KEY nao encontrada no .env. "
                "Obtenha em platform.deepseek.com"
            )
        _deepseek_client = OpenAI(
            api_key=api_key,
            base_url="https://api.deepseek.com",
        )
        log.info("Cliente DeepSeek carregado: %s", MODELO_DEEPSEEK)
    return _deepseek_client


# ---------------------------------------------------------------------------
# Chamadas por provedor
# ---------------------------------------------------------------------------

def _chamar_groq(system_prompt, prompt_usuario):
    client = _get_groq()
    inicio = time.monotonic()

    resp = client.chat.completions.create(
        model=MODELO,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": prompt_usuario},
        ],
        temperature=TEMPERATURA,
        max_tokens=MAX_TOKENS,
    )

    lat = int((time.monotonic() - inicio) * 1000)
    tokens_p = resp.usage.prompt_tokens if resp.usage else 0
    tokens_r = resp.usage.completion_tokens if resp.usage else 0
    texto    = resp.choices[0].message.content or ""

    return texto, tokens_p, tokens_r, lat, MODELO


def _chamar_deepseek(system_prompt, prompt_usuario):
    """Chama o DeepSeek R1 via API OpenAI-compativel."""
    client = _get_deepseek()
    inicio = time.monotonic()

    resp = client.chat.completions.create(
        model=MODELO_DEEPSEEK,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": prompt_usuario},
        ],
        temperature=TEMPERATURA,
        max_tokens=MAX_TOKENS,
    )

    lat      = int((time.monotonic() - inicio) * 1000)
    tokens_p = resp.usage.prompt_tokens if resp.usage else 0
    tokens_r = resp.usage.completion_tokens if resp.usage else 0
    texto    = resp.choices[0].message.content or ""

    return texto, tokens_p, tokens_r, lat, MODELO_DEEPSEEK


def _chamar_llm(system_prompt, prompt_usuario, forcar_deepseek=False):
    """
    Tenta o provedor configurado. Se receber 429, faz fallback automatico para DeepSeek R1.
    """
    provider = "deepseek" if forcar_deepseek else PROVIDER

    if provider == "deepseek":
        return _chamar_deepseek(system_prompt, prompt_usuario)

    # Groq (padrao)
    try:
        return _chamar_groq(system_prompt, prompt_usuario)
    except Exception as e:
        erro = str(e)
        if "429" in erro or "rate_limit" in erro:
            log.warning("Groq rate limit (429) — fazendo fallback para DeepSeek R1")
            try:
                return _chamar_deepseek(system_prompt, prompt_usuario)
            except Exception as e2:
                log.error("DeepSeek tambem falhou: %s", e2)
                raise EnvironmentError(
                    "Rate limit do Groq atingido e DeepSeek nao esta configurado. "
                    "Adicione DEEPSEEK_API_KEY no .env ou aguarde o reset do Groq."
                )
        raise


# ---------------------------------------------------------------------------
# Geracao de resposta
# ---------------------------------------------------------------------------

def gerar_resposta(pergunta, chunks, tipo_forcado=None):
    """
    Gera uma resposta fundamentada nos chunks recuperados pela P2.

    Seleciona automaticamente o system prompt por tipo de pergunta.
    Faz fallback para Gemini se Groq atingir rate limit (429).
    """
    if not chunks:
        return RespostaLLM(
            texto=FALLBACK_MSG,
            modelo=MODELO,
            tokens_prompt=0,
            tokens_resposta=0,
            latencia_ms=0,
            system_prompt="",
        )

    # Detecta tipo e seleciona prompt
    e_fallback = any(c.get("busca_fallback") for c in chunks)
    tipo = "mencao" if e_fallback else (tipo_forcado or _detectar_tipo_pergunta(pergunta))
    system_prompt = _selecionar_prompt(tipo)

    log.info("Tipo detectado: %s | chunks: %d | provider: %s", tipo, len(chunks), PROVIDER)

    contexto        = _formatar_contexto(chunks)
    prompt_usuario  = _formatar_prompt(pergunta, contexto, tipo)

    inicio = time.monotonic()

    try:
        texto, tokens_p, tokens_r, lat_llm, modelo_usado = _chamar_llm(
            system_prompt, prompt_usuario
        )
    except EnvironmentError as e:
        return RespostaLLM(
            texto=str(e),
            modelo=MODELO,
            tokens_prompt=0,
            tokens_resposta=0,
            latencia_ms=int((time.monotonic() - inicio) * 1000),
            system_prompt=system_prompt,
        )

    # Segunda tentativa para comparacao sem resultado — fora do try principal
    if tipo == "comparacao" and (
        not texto.strip()
        or FALLBACK_MSG in texto
        or "Não encontrado" in texto
        or "Nao encontrado" in texto
        or len(texto.strip()) < 200
    ):
        log.info("Comparacao sem resultado — segunda tentativa com prompt de busca")
        try:
            system_prompt_fb = _selecionar_prompt("busca")
            prompt_fb = _formatar_prompt(pergunta, contexto, "busca")
            texto2, tokens_p2, tokens_r2, lat2, modelo_usado = _chamar_llm(
                system_prompt_fb, prompt_fb
            )
            if texto2 and texto2.strip() and FALLBACK_MSG not in texto2:
                texto = texto2
                tokens_p += tokens_p2
                tokens_r += tokens_r2
                lat_llm += lat2
                log.info("Segunda tentativa bem sucedida | tokens extras=(%dp + %dr)", tokens_p2, tokens_r2)
        except Exception as e:
            log.warning("Segunda tentativa falhou: %s", e)

    # Trata resposta vazia
    if not texto or not texto.strip():
        texto = FALLBACK_MSG

    log.info("Resposta gerada | lat=%dms | tokens=(%dp + %dr) | modelo=%s",
             lat_llm, tokens_p, tokens_r, modelo_usado)

    avaliacao = avaliar_resposta(texto, chunks)
    fallback  = int(FALLBACK_MSG in texto)

    log_data = {
        "query":                  pergunta,
        "tipo_pergunta":          tipo,
        "system_prompt":          system_prompt,
        "system_prompt_version":  SYSTEM_PROMPT_VERSION,
        "user_prompt":            prompt_usuario,
        "response":               texto,
        "chunks": [
            {
                "doc_id": c.get("doc_id"),
                "tipo":   c.get("tipo_nome"),
                "numero": c.get("numero"),
                "ano":    c.get("ano"),
                "score":  c.get("score_final"),
                "fonte":  c.get("fonte_texto"),
            }
            for c in chunks
        ],
        "num_chunks":      len(chunks),
        "latency_ms":      lat_llm,
        "tokens_prompt":   tokens_p,
        "tokens_response": tokens_r,
        "model":           modelo_usado,
        "provider":        PROVIDER,
        "temperature":     TEMPERATURA,
        "fallback":        fallback,
        **avaliacao,
    }

    salvar_log(log_data)

    return RespostaLLM(
        texto=texto,
        modelo=modelo_usado,
        tokens_prompt=tokens_p,
        tokens_resposta=tokens_r,
        latencia_ms=lat_llm,
        system_prompt=system_prompt,
    )


def avaliar_resposta(resposta, chunks):
    contexto_texto = " ".join([c.get("texto", "") for c in chunks])
    faithfulness   = int(any(t in contexto_texto for t in resposta.split(".")))
    citation_ok    = int(
        "Resolucao" in resposta or "Resolução" in resposta or
        "Art." in resposta or "Despacho" in resposta or "Portaria" in resposta
    )
    return {"faithfulness": faithfulness, "citation_accuracy": citation_ok}


# ---------------------------------------------------------------------------
# Teste direto
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    print("Provider:", PROVIDER)
    print("Modelo Groq:", MODELO)
    print("Modelo DeepSeek:", MODELO_DEEPSEEK)

    chunks_mock = [
        {
            "tipo_nome": "Despacho",
            "numero": "3182",
            "ano": "2021",
            "fonte_texto": "pdf_completo",
            "texto": (
                "Tabela I: TUST-RB e TUSTTEMP para a UTE TERMONORTE I, "
                "a precos de 1 de junho de 2021, para o ciclo 2021-2022. "
                "GERADOR CEG TUST-RB TUSTTemp Nucleo R$/kW (R$/MWh) "
                "UTE Termonorte I 027887-4 12,852 21,246 "
                "aplicaveis em horario unico, sem distincao entre ponta e fora de ponta."
            ),
        }
    ]

    resultado = gerar_resposta(
        "Qual o valor da TUST-RB para UTE Termonorte I no ciclo 2021-2022?",
        chunks_mock,
    )

    print("\nResposta:", resultado.texto)
    print("Modelo usado:", resultado.modelo)
    print("Tokens:", resultado.tokens_prompt, "p +", resultado.tokens_resposta, "r")
    print("Latencia:", resultado.latencia_ms, "ms")