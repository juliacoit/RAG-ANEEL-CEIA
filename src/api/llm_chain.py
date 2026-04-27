"""
llm_chain.py
============
Pessoa 3 — LLM Engineer

Suporta multiplos provedores de LLM com fallback automatico:
  1. Qwen (qwen3-30b-a3b via DashScope) — primario, barato
  2. Groq (llama-3.3-70b-versatile)     — fallback rapido gratuito
  3. Gemini Flash                        — ultimo recurso (opcional)

Configuracao no .env:
  LLM_PROVIDER=qwen                     # qwen | groq | gemini
  LLM_MODEL=qwen3-30b-a3b
  QWEN_API_KEY=sk-...
  QWEN_BASE_URL=https://dashscope-intl.aliyuncs.com/compatible-mode/v1
  GROQ_API_KEY=...                      # fallback gratuito (console.groq.com)
  GROQ_MODEL=llama-3.3-70b-versatile   # opcional, esse e o padrao
  GEMINI_API_KEY=...                    # fallback opcional (aistudio.google.com)
  GEMINI_MODEL=gemini-2.0-flash

Cadeia de fallback:
  Qwen (primario)
    → erro → Groq com llama-3.3-70b-versatile (fallback 1)
    → erro → Gemini (fallback 2, so se GEMINI_API_KEY configurada)
    → erro → mensagem clara sem crash

Uso direto (teste):
  python src/api/llm_chain.py
"""

import logging
import os
import re
import time
from dataclasses import dataclass

# ── CRÍTICO: load_dotenv ANTES de qualquer os.getenv ─────────────────────────
# As variáveis globais MODELO, PROVIDER etc são lidas no import do módulo.
# Se load_dotenv() for chamado depois (ex: no main.py), o .env ainda não foi
# carregado e os.getenv retorna None — causando model ID truncado ou errado.
from dotenv import load_dotenv
load_dotenv()

from src.utils.logger_metrics import salvar_log

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuracao — lidas APÓS load_dotenv()
# ---------------------------------------------------------------------------

PROVIDER      = os.getenv("LLM_PROVIDER", "qwen").lower()
MODELO        = os.getenv("LLM_MODEL", "qwen3-30b-a3b")            # modelo Qwen primario
MODELO_GROQ   = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")  # modelo Groq fallback
MODELO_GEMINI = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
QWEN_BASE_URL = os.getenv("QWEN_BASE_URL", "https://dashscope-intl.aliyuncs.com/compatible-mode/v1")
TEMPERATURA   = 0.1
MAX_TOKENS    = 2048

FALLBACK_MSG          = "Nao encontrado nos atos normativos consultados."
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

SYSTEM_PROMPT_DEFINICAO = SYSTEM_PROMPT_BASE + (
    "\nModo: DEFINICAO DE TERMO TECNICO\n"
    "- Localize a definicao formal do termo nos trechos.\n"
    "- Prefira trechos de artigos definitorios (Art. 2, Art. 3, Definicoes).\n"
    "- Se houver mais de uma definicao, aponte a mais recente.\n"
    "- Contextualize com o ato normativo que define o termo.\n"
)

SYSTEM_PROMPT_PROCEDIMENTO = SYSTEM_PROMPT_BASE + (
    "\nModo: PROCEDIMENTO / FLUXO\n"
    "- Identifique as etapas do processo descrito nos trechos.\n"
    "- Organize em ordem cronologica ou sequencial quando possivel.\n"
    "- Aponte prazos, responsaveis e documentos exigidos mencionados.\n"
    "- Cite o artigo ou inciso de origem de cada etapa.\n"
)

SYSTEM_PROMPT_VIGENCIA = SYSTEM_PROMPT_BASE + (
    "\nModo: VIGENCIA / STATUS\n"
    "- Verifique se os trechos indicam revogacao, alteracao ou suspensao do ato.\n"
    "- Frases como fica revogada, tornada sem efeito, suspensa sao indicadores chave.\n"
    "- Informe a data de vigencia se mencionada.\n"
    "- Se nao houver indicacao de revogacao, informe que o ato aparece como vigente nos trechos.\n"
)

SYSTEM_PROMPT_COMPARACAO_TEMPORAL = SYSTEM_PROMPT_BASE + (
    "\nModo: COMPARACAO TEMPORAL\n"
    "- Compare como o tema evoluiu entre os periodos mencionados.\n"
    "- Aponte mudancas concretas: valores alterados, artigos revogados, novas obrigacoes.\n"
    "- Organize cronologicamente: o que valia antes x o que vale depois.\n"
    "- Cite os atos de cada periodo que fundamentam a comparacao.\n"
)

SYSTEM_PROMPT_COMPARACAO_REF = SYSTEM_PROMPT_BASE + (
    "\nModo: REFERENCIAS CRUZADAS\n"
    "- Identifique todos os atos que mencionam o documento solicitado.\n"
    "- Para cada mencao: qual ato cita, em que contexto (revogacao, alteracao, referencia).\n"
    "- Organize por tipo de relacao (revoga, altera, complementa, cita).\n"
)

SYSTEM_PROMPT_AUTORIA = SYSTEM_PROMPT_BASE + (
    "\nModo: AUTORIA\n"
    "- Liste os atos encontrados pelo autor/relator solicitado.\n"
    "- Para cada ato: tipo, numero, ano, assunto resumido.\n"
    "- Organize por data de publicacao quando disponivel.\n"
)

# ---------------------------------------------------------------------------
# Detecta tipo de pergunta
# ---------------------------------------------------------------------------

def _detectar_tipo_pergunta(pergunta):
    p = pergunta.lower()

    if any(t in p for t in ["ainda vigente", "ainda em vigor", "foi revogada", "foi revogado",
                              "quando foi revogad", "continua valendo", "status", "situacao atual",
                              "esta em vigor", "está em vigor"]):
        return "vigencia"

    if any(t in p for t in ["entre 20", "mudou de", "evoluiu", "o que mudou",
                              "comparando os anos", "ao longo dos anos"]):
        anos = re.findall(r'\b(20\d{2})\b', p)
        if len(set(anos)) >= 2:
            return "comparacao_temporal"

    if any(t in p for t in ["revoga", "revogou", "cancela", "substitui", "substituiu",
                              "altera", "alterou", "contradiz", "diferenca", "comparar",
                              "relacao entre", "afeta", "anterior", "posterior",
                              "quais documentos citam", "quais atos mencionam"]):
        return "comparacao_referencia" if any(t in p for t in ["citam", "mencionam", "referenciam"]) else "comparacao"

    if any(t in p for t in ["valor", "valores", "tarifa", "tarifas", "percentual",
                              "tabela", "planilha", "quanto", "custo", "preco",
                              "taxa", "taxas", "indice", "quantos", "tust", "tusd", "rap",
                              "depreciacao", "depreciação", "gerador", "turbina",
                              "tuc", "tipo de unidade", "amortizacao", "amortização",
                              "vida util", "vida útil", "mcpse", "mcse",
                              "aliquota", "alíquota", "unitario", "unitário"]):
        return "tabela"

    if any(t in p for t in ["o que e ", "o que é ", "o que significa", "definicao de",
                              "definição de", "conceito de", "como e definid"]):
        return "definicao"

    if any(t in p for t in ["como solicitar", "como requerer", "como contestar",
                              "qual o processo", "qual o procedimento", "quais os passos",
                              "como fazer", "como obter", "como regularizar"]):
        return "procedimento"

    if any(t in p for t in ["quem assinou", "quem publicou", "quem emitiu",
                              "atos do relator", "atos do diretor", "assinados por"]):
        return "autoria"

    if any(t in p for t in ["resumo", "resume", "resuma", "sintetize", "introducao",
                              "prefacio", "o que diz", "do que trata", "explique",
                              "explica", "descreva", "contexto", "historico"]):
        return "resumo"

    if any(t in p for t in ["artigo", "art.", "inciso", "paragrafo", "clausula",
                              "item ", "secao"]):
        return "especifica"

    return "busca"


def _selecionar_prompt(tipo):
    return {
        "resumo":                SYSTEM_PROMPT_RESUMO,
        "comparacao":            SYSTEM_PROMPT_COMPARACAO,
        "comparacao_referencia": SYSTEM_PROMPT_COMPARACAO_REF,
        "comparacao_temporal":   SYSTEM_PROMPT_COMPARACAO_TEMPORAL,
        "tabela":                SYSTEM_PROMPT_TABELA,
        "busca":                 SYSTEM_PROMPT_BUSCA,
        "mencao":                SYSTEM_PROMPT_MENCAO,
        "especifica":            SYSTEM_PROMPT_ESPECIFICA,
        "definicao":             SYSTEM_PROMPT_DEFINICAO,
        "procedimento":          SYSTEM_PROMPT_PROCEDIMENTO,
        "vigencia":              SYSTEM_PROMPT_VIGENCIA,
        "autoria":               SYSTEM_PROMPT_AUTORIA,
        "agregacao":             SYSTEM_PROMPT_BUSCA,
        "hibrida_complexa":      SYSTEM_PROMPT_BUSCA,
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
# Clientes LLM (lazy init)
# ---------------------------------------------------------------------------

_qwen_client   = None
_groq_client   = None
_gemini_client = None


def _get_qwen():
    """Cliente OpenAI-compatible apontando para DashScope (Alibaba)."""
    global _qwen_client
    if _qwen_client is None:
        from openai import OpenAI
        api_key = os.getenv("QWEN_API_KEY", "").strip()
        if not api_key:
            raise EnvironmentError(
                "QWEN_API_KEY nao encontrada no .env. "
                "Obtenha em dashscope.console.aliyun.com"
            )
        _qwen_client = OpenAI(api_key=api_key, base_url=QWEN_BASE_URL)
        log.info("Cliente Qwen carregado: %s | base_url: %s", MODELO, QWEN_BASE_URL)
    return _qwen_client


def _get_groq():
    global _groq_client
    if _groq_client is None:
        from groq import Groq
        api_key = os.getenv("GROQ_API_KEY", "").strip()
        if not api_key:
            raise EnvironmentError("GROQ_API_KEY nao encontrada no .env")
        _groq_client = Groq(api_key=api_key)
        log.info("Cliente Groq carregado: %s", MODELO_GROQ)
    return _groq_client


def _get_gemini():
    """
    Suporta as duas bibliotecas do Gemini:
      - google-genai (nova): pip install google-genai
      - google-generativeai (antiga): pip install google-generativeai
    """
    global _gemini_client
    if _gemini_client is None:
        api_key = os.getenv("GEMINI_API_KEY", "").strip()
        if not api_key:
            raise EnvironmentError(
                "GEMINI_API_KEY nao encontrada no .env. "
                "Obtenha gratis em aistudio.google.com"
            )
        try:
            from google import genai as genai_new
            client = genai_new.Client(api_key=api_key)
            _gemini_client = ("new", client)
            log.info("Cliente Gemini (google-genai) carregado: %s", MODELO_GEMINI)
        except ImportError:
            try:
                import google.generativeai as genai_old
                genai_old.configure(api_key=api_key)
                _gemini_client = ("old", genai_old)
                log.info("Cliente Gemini (google-generativeai) carregado: %s", MODELO_GEMINI)
            except ImportError:
                raise ImportError(
                    "Nenhuma biblioteca Gemini encontrada. "
                    "Instale: pip install google-genai"
                )
    return _gemini_client


# ---------------------------------------------------------------------------
# Chamadas por provedor
# ---------------------------------------------------------------------------

def _chamar_qwen(system_prompt, prompt_usuario):
    """
    Chama o Qwen3 via DashScope com thinking desativado.
    enable_thinking=False evita tokens <think>...</think> que causam
    resposta vazia se nao tratados pelo cliente.
    """
    client = _get_qwen()
    inicio = time.monotonic()

    resp = client.chat.completions.create(
        model=MODELO,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": prompt_usuario},
        ],
        temperature=TEMPERATURA,
        max_tokens=MAX_TOKENS,
        extra_body={"enable_thinking": False},
    )

    lat      = int((time.monotonic() - inicio) * 1000)
    tokens_p = resp.usage.prompt_tokens if resp.usage else 0
    tokens_r = resp.usage.completion_tokens if resp.usage else 0
    texto    = resp.choices[0].message.content or ""

    # Limpa tokens de thinking residuais
    texto = re.sub(r"<think>.*?</think>", "", texto, flags=re.DOTALL).strip()

    return texto, tokens_p, tokens_r, lat, MODELO


def _chamar_groq(system_prompt, prompt_usuario):
    """
    Chama o Groq usando MODELO_GROQ (llama-3.3-70b-versatile).
    Nunca usa MODELO (que e o modelo Qwen) — sao variaveis separadas.
    """
    client = _get_groq()
    inicio = time.monotonic()

    resp = client.chat.completions.create(
        model=MODELO_GROQ,   # ← sempre llama, independente do MODELO Qwen
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

    return texto, tokens_p, tokens_r, lat, MODELO_GROQ


def _chamar_gemini(system_prompt, prompt_usuario):
    """Chama o Gemini usando a biblioteca disponivel (nova ou antiga)."""
    lib_tipo, client = _get_gemini()
    inicio = time.monotonic()

    if lib_tipo == "new":
        from google.genai import types as genai_types
        resp = client.models.generate_content(
            model=MODELO_GEMINI,
            contents=prompt_usuario,
            config=genai_types.GenerateContentConfig(
                system_instruction=system_prompt,
                temperature=TEMPERATURA,
                max_output_tokens=MAX_TOKENS,
            ),
        )
        texto    = resp.text or ""
        tokens_p = getattr(getattr(resp, "usage_metadata", None), "prompt_token_count", 0) or 0
        tokens_r = getattr(getattr(resp, "usage_metadata", None), "candidates_token_count", 0) or 0
    else:
        genai_mod = client
        model = genai_mod.GenerativeModel(
            model_name=MODELO_GEMINI,
            system_instruction=system_prompt,
            generation_config=genai_mod.GenerationConfig(
                temperature=TEMPERATURA,
                max_output_tokens=MAX_TOKENS,
            ),
        )
        resp     = model.generate_content(prompt_usuario)
        texto    = resp.text or ""
        tokens_p = getattr(resp.usage_metadata, "prompt_token_count", 0) or 0
        tokens_r = getattr(resp.usage_metadata, "candidates_token_count", 0) or 0

    lat = int((time.monotonic() - inicio) * 1000)
    return texto, tokens_p, tokens_r, lat, MODELO_GEMINI


def _chamar_llm(system_prompt, prompt_usuario, forcar_gemini=False):
    """
    Cadeia de fallback:
      Qwen (primario) → Groq/llama (fallback 1) → Gemini (fallback 2, opcional)

    Gemini so e tentado se GEMINI_API_KEY estiver no .env.
    Se nenhum provider funcionar, retorna EnvironmentError com mensagem clara.
    """
    if forcar_gemini:
        return _chamar_gemini(system_prompt, prompt_usuario)

    # ── 1. Qwen como primario ─────────────────────────────────────────────
    if PROVIDER == "qwen":
        try:
            return _chamar_qwen(system_prompt, prompt_usuario)
        except Exception as e:
            log.warning("Qwen falhou (%s) — fallback para Groq (%s)", str(e)[:120], MODELO_GROQ)

        # ── 2. Groq como fallback ─────────────────────────────────────────
        groq_key = os.getenv("GROQ_API_KEY", "").strip()
        if groq_key:
            try:
                log.info("Tentando Groq (%s) como fallback...", MODELO_GROQ)
                return _chamar_groq(system_prompt, prompt_usuario)
            except Exception as e2:
                log.warning("Groq falhou (%s) — fallback para Gemini", str(e2)[:120])
        else:
            log.warning("GROQ_API_KEY nao configurada — pulando fallback Groq")

        # ── 3. Gemini como ultimo recurso (opcional) ──────────────────────
        gemini_key = os.getenv("GEMINI_API_KEY", "").strip()
        if gemini_key:
            try:
                log.info("Tentando Gemini (%s) como ultimo fallback...", MODELO_GEMINI)
                return _chamar_gemini(system_prompt, prompt_usuario)
            except Exception as e3:
                raise EnvironmentError(
                    "Todos os provedores falharam. "
                    "Ultimo erro Gemini: %s" % str(e3)[:120]
                )

        raise EnvironmentError(
            "Qwen falhou e Groq nao esta disponivel. "
            "Verifique QWEN_API_KEY e GROQ_API_KEY no .env."
        )

    # ── Groq como primario (LLM_PROVIDER=groq) ───────────────────────────
    if PROVIDER == "groq":
        try:
            return _chamar_groq(system_prompt, prompt_usuario)
        except Exception as e:
            erro = str(e)
            if "429" in erro or "rate_limit" in erro:
                log.warning("Groq rate limit (429) — fallback para Gemini")
                gemini_key = os.getenv("GEMINI_API_KEY", "").strip()
                if gemini_key:
                    try:
                        return _chamar_gemini(system_prompt, prompt_usuario)
                    except Exception as e2:
                        raise EnvironmentError(
                            "Rate limit do Groq e Gemini tambem falhou: %s" % str(e2)[:120]
                        )
                raise EnvironmentError(
                    "Rate limit do Groq atingido. "
                    "Adicione GEMINI_API_KEY no .env ou aguarde reset do Groq."
                )
            raise

    # ── Gemini como primario (LLM_PROVIDER=gemini) ───────────────────────
    return _chamar_gemini(system_prompt, prompt_usuario)


# ---------------------------------------------------------------------------
# Geracao de resposta
# ---------------------------------------------------------------------------

def gerar_resposta(pergunta, chunks, tipo_forcado=None):
    """
    Gera uma resposta fundamentada nos chunks recuperados pela P2.

    Seleciona automaticamente o system prompt por tipo de pergunta.
    Faz fallback automatico entre provedores conforme cadeia configurada.
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

    e_fallback    = any(c.get("busca_fallback") for c in chunks)
    tipo          = "mencao" if e_fallback else (tipo_forcado or _detectar_tipo_pergunta(pergunta))
    system_prompt = _selecionar_prompt(tipo)

    log.info("Tipo detectado: %s | chunks: %d | provider: %s | modelo: %s",
             tipo, len(chunks), PROVIDER, MODELO)

    contexto       = _formatar_contexto(chunks)
    prompt_usuario = _formatar_prompt(pergunta, contexto, tipo)

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

    # Segunda tentativa para comparacao sem resultado
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
            prompt_fb        = _formatar_prompt(pergunta, contexto, "busca")
            texto2, tokens_p2, tokens_r2, lat2, modelo_usado = _chamar_llm(
                system_prompt_fb, prompt_fb
            )
            if texto2 and texto2.strip() and FALLBACK_MSG not in texto2:
                texto     = texto2
                tokens_p += tokens_p2
                tokens_r += tokens_r2
                lat_llm  += lat2
                log.info("Segunda tentativa bem sucedida | tokens extras=(%dp + %dr)",
                         tokens_p2, tokens_r2)
        except Exception as e:
            log.warning("Segunda tentativa falhou: %s", e)

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
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    print("Provider  :", PROVIDER)
    print("Modelo    :", MODELO)
    print("Groq model:", MODELO_GROQ)
    print("Gemini    :", MODELO_GEMINI)
    print("Qwen URL  :", QWEN_BASE_URL)

    chunks_mock = [
        {
            "tipo_nome":   "Despacho",
            "numero":      "3182",
            "ano":         "2021",
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

    print("\nResposta :", resultado.texto)
    print("Modelo   :", resultado.modelo)
    print("Tokens   :", resultado.tokens_prompt, "p +", resultado.tokens_resposta, "r")
    print("Latencia :", resultado.latencia_ms, "ms")