"""
llm_chain.py
============
Pessoa 3 — LLM Engineer

Recebe a pergunta do usuário + chunks recuperados pela P2 e gera uma
resposta fundamentada usando Groq (gratuito — console.groq.com).

Pré-requisitos:
  - pip install groq
  - .env com GROQ_API_KEY (obtida em console.groq.com — sem cartão)

Uso direto (teste):
  python src/api/llm_chain.py
"""

import logging
import os
import time
from dataclasses import dataclass
from src.utils.logger_metrics import salvar_log

from groq import Groq
from dotenv import load_dotenv

load_dotenv()

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuração do modelo
# ---------------------------------------------------------------------------

MODELO = os.getenv("LLM_MODEL", "llama-3.3-70b-versatile")
TEMPERATURA = 0.1
MAX_TOKENS = 1024

SYSTEM_PROMPT = """Você é um especialista em regulação do setor elétrico brasileiro.
Responda EXCLUSIVAMENTE com base nos trechos fornecidos abaixo.

Liste explicitamente todos os atos normativos relevantes encontrados no contexto.

Para cada ato citado:
- informe o número e ano
- descreva brevemente o que ele trata
- cite explicitamente a fonte

NÃO inclua atos que não estejam claramente presentes nos trechos.

Se não houver evidência suficiente, responda:
"Não encontrado nos atos normativos consultados."
Toda afirmação deve citar o ato normativo de origem (ex: "conforme o Art. 3º
da Resolução Normativa ANEEL nº 687/2016").
Se a informação não constar nos documentos, responda:
"Não encontrado nos atos normativos consultados."
Nunca infira, suponha ou complete com conhecimento externo."""

FALLBACK_MSG = "Não encontrado nos atos normativos consultados."

SYSTEM_PROMPT_VERSION = "V2"

_client_cache: Groq | None = None


def _carregar_client() -> Groq:
    global _client_cache
    if _client_cache is None:
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "GROQ_API_KEY não encontrada. "
                "Obtenha em console.groq.com e adicione ao .env"
            )
        _client_cache = Groq(api_key=api_key.strip())
        log.info(f"Cliente Groq carregado: {MODELO}")
    return _client_cache


# ---------------------------------------------------------------------------
# Formatação do contexto
# ---------------------------------------------------------------------------

def _formatar_contexto(chunks: list[dict]) -> str:
    partes = []
    for i, chunk in enumerate(chunks, 1):
        titulo = chunk.get("titulo") or (
            f"{chunk.get('tipo_nome', '')} nº {chunk.get('numero', '')} "
            f"{chunk.get('ano', '')}"
        ).strip()
        partes.append(
            f"[Trecho {i}] {titulo}\n"
            f"{chunk.get('texto', '').strip()}"
        )
    return "\n\n---\n\n".join(partes)


def _formatar_prompt(pergunta: str, contexto: str) -> str:
    return (
        f"DOCUMENTOS CONSULTADOS:\n\n{contexto}\n\n"
        f"PERGUNTA: {pergunta}"
    )


# ---------------------------------------------------------------------------
# Estrutura de resposta
# ---------------------------------------------------------------------------

@dataclass
class RespostaLLM:
    texto: str
    modelo: str
    tokens_prompt: int
    tokens_resposta: int
    latencia_ms: int
    system_prompt: str


# ---------------------------------------------------------------------------
# Geração de resposta
# ---------------------------------------------------------------------------

def gerar_resposta(pergunta: str, chunks: list[dict]) -> RespostaLLM:
    """
    Gera uma resposta fundamentada nos chunks recuperados pela P2.

    Parâmetros:
      pergunta — pergunta do usuário em linguagem natural
      chunks   — lista de dicts retornada por p2_indexar.buscar()

    Retorna RespostaLLM com texto, modelo, contagem de tokens e latência.
    """
    if not chunks:
        return RespostaLLM(
            texto=FALLBACK_MSG,
            modelo=MODELO,
            tokens_prompt=0,
            tokens_resposta=0,
            latencia_ms=0,
        )

    client = _carregar_client()
    contexto = _formatar_contexto(chunks)
    prompt = _formatar_prompt(pergunta, contexto)

    log.info(f"Gerando resposta | modelo={MODELO} | chunks={len(chunks)}")
    inicio = time.monotonic()

    resposta = client.chat.completions.create(
        model=MODELO,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        temperature=TEMPERATURA,
        max_tokens=MAX_TOKENS,
    )

    latencia_ms = int((time.monotonic() - inicio) * 1000)
    tokens_prompt = resposta.usage.prompt_tokens if resposta.usage else 0
    tokens_resp = resposta.usage.completion_tokens if resposta.usage else 0

    log.info(
        f"Resposta gerada | latencia={latencia_ms}ms | "
        f"tokens=({tokens_prompt}p + {tokens_resp}r)"
    )

    texto_resposta = resposta.choices[0].message.content
    avaliacao = avaliar_resposta(texto_resposta, chunks)
    fallback = int(texto_resposta.strip() == FALLBACK_MSG)

    log_data = {
    "query": pergunta,

    "system_prompt": SYSTEM_PROMPT,
    "system_prompt_version": SYSTEM_PROMPT_VERSION,
    "user_prompt": prompt,

    # resposta
    "response": texto_resposta,

    # chunks (recomendo versão reduzida)
    "chunks": [
        {
            "doc_id": c.get("doc_id"),
            "tipo": c.get("tipo_nome"),
            "numero": c.get("numero"),
            "ano": c.get("ano"),
            "score": c.get("score_final"),
        }
        for c in chunks
    ],

    "num_chunks": len(chunks),

    # métricas
    "latency_ms": latencia_ms,
    "tokens_prompt": tokens_prompt,
    "tokens_response": tokens_resp,
    "model": MODELO,
    "temperature": TEMPERATURA,

    # avaliação
    "fallback": fallback,
    **avaliacao
}

    salvar_log(log_data)

    return RespostaLLM(
        texto=texto_resposta,
        modelo=MODELO,
        tokens_prompt=tokens_prompt,
        tokens_resposta=tokens_resp,
        latencia_ms=latencia_ms,
        system_prompt=SYSTEM_PROMPT
    )

def avaliar_resposta(resposta: str, chunks: list[dict]) -> dict:
    contexto_texto = " ".join([c.get("texto", "") for c in chunks])

    # heurística simples
    faithfulness = int(any(trecho in contexto_texto for trecho in resposta.split(".")))

    citation_ok = int("Resolução" in resposta or "Art." in resposta)

    return {
        "faithfulness": faithfulness,
        "citation_accuracy": citation_ok,
    }

# ---------------------------------------------------------------------------
# Teste rápido (rodar direto)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    chunks_mock = [
        {
            "titulo": "Resolução Normativa ANEEL nº 687/2015",
            "texto": (
                "A micro e minigeração distribuída é a produção de energia elétrica "
                "por consumidores por meio de pequenas centrais geradoras conectadas "
                "à rede de distribuição por meio de instalações de unidades consumidoras."
            ),
        }
    ]

    resultado = gerar_resposta(
        "O que é microgeração distribuída segundo a ANEEL?",
        chunks_mock,
    )

    print(f"\nResposta: {resultado.texto}")
    print(f"Modelo: {resultado.modelo}")
    print(f"Tokens: {resultado.tokens_prompt}p + {resultado.tokens_resposta}r")
    print(f"Latência: {resultado.latencia_ms}ms")
