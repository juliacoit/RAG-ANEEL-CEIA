"""
eval_runner.py
==============
Avaliação automática do sistema RAG ANEEL usando o framework RAGAS.

Métricas avaliadas (sem necessidade de ground truth):
  - Faithfulness            : a resposta está fundamentada no contexto recuperado?
  - ResponseRelevancy       : a resposta é pertinente à pergunta?
  - LLMContextPrecisionWithoutReference : os chunks recuperados são relevantes?

O avaliador usa Groq (mesmo provider da geração) como LLM e OpenAI (opcional)
para embeddings de ResponseRelevancy. Sem OPENAI_API_KEY, usa embeddings locais.

Pré-requisitos:
  pip install -r requirements.txt
  Docker rodando com Qdrant indexado (python src/search/indexar.py)
  .env com GROQ_API_KEY (e opcionalmente OPENAI_API_KEY)

Uso — rodar da RAIZ do projeto:
  python src/evaluation/eval_runner.py --limite 10
  python src/evaluation/eval_runner.py --categoria microgeracao
  python src/evaluation/eval_runner.py --tipo fallback
  python src/evaluation/eval_runner.py
  python src/evaluation/eval_runner.py --saida docs/resultado_avaliacao.csv
"""

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

load_dotenv()

RAIZ = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(RAIZ))

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Imports do projecto
# ---------------------------------------------------------------------------

from src.evaluation.banco_perguntas import (
    BANCO,
    Categoria,
    TipoResposta,
    por_categoria,
    por_tipo_resposta,
)
from src.search.indexar import buscar, carregar_indices, conectar_qdrant
from src.api.llm_chain import gerar_resposta


# ---------------------------------------------------------------------------
# Imports RAGAS — carregados sob demanda para não bloquear outros usos do módulo
# ---------------------------------------------------------------------------

def _importar_ragas():
    """
    Importa RAGAS e os wrappers LangChain necessários.
    Dá mensagem de erro clara se os pacotes não estiverem instalados.
    """
    try:
        from ragas import EvaluationDataset, evaluate
        from ragas.dataset_schema import SingleTurnSample
        try:
            from ragas.metrics.collections import (
                Faithfulness,
                ResponseRelevancy,
                LLMContextPrecisionWithoutReference,
            )
        except ImportError:
            from ragas.metrics import (  # type: ignore[no-redef]
                Faithfulness,
                ResponseRelevancy,
                LLMContextPrecisionWithoutReference,
            )
        from ragas.llms import LangchainLLMWrapper
        from ragas.embeddings import LangchainEmbeddingsWrapper
        from langchain_groq import ChatGroq
        from langchain_openai import OpenAIEmbeddings
        try:
            from ragas.run_config import RunConfig as _RunConfig  # type: ignore[import]
        except ImportError:
            _RunConfig = None
    except ImportError as e:
        print(
            f"\n[ERRO] Pacote não encontrado: {e}\n"
            "Instale as dependências de avaliação:\n"
            "  pip install ragas>=0.2 langchain-groq langchain-openai\n"
        )
        sys.exit(1)

    return (
        EvaluationDataset,
        evaluate,
        SingleTurnSample,
        Faithfulness,
        ResponseRelevancy,
        LLMContextPrecisionWithoutReference,
        LangchainLLMWrapper,
        LangchainEmbeddingsWrapper,
        ChatGroq,
        OpenAIEmbeddings,
        _RunConfig,
    )


# ---------------------------------------------------------------------------
# Pipeline RAG: pergunta → (resposta, contextos, metadados)
# ---------------------------------------------------------------------------

def rodar_rag(
    pergunta: str,
    filtros: Optional[dict],
    qclient,
    bm25,
    bm25_ids,
    n_resultados: int = 5,
) -> tuple[str, list[str], list[dict]]:
    """
    Executa o pipeline RAG completo para uma pergunta.

    Retorna:
        resposta      — texto gerado pelo LLM
        contextos     — lista de strings dos chunks recuperados (para RAGAS)
        chunks_meta   — lista completa com metadados (para logging)
    """
    chunks = buscar(
        query=pergunta,
        qclient=qclient,
        bm25=bm25,
        bm25_ids=bm25_ids,
        n_resultados=n_resultados,
        filtros=filtros,
    )
    chunks = [c for c in chunks if c.get("texto", "").strip()]

    if not chunks:
        return "Não encontrado nos atos normativos consultados.", [], []

    resultado = gerar_resposta(pergunta, chunks)
    contextos = [c.get("texto", "").strip() for c in chunks]
    return resultado.texto, contextos, chunks


# ---------------------------------------------------------------------------
# Configurar avaliadores RAGAS
# ---------------------------------------------------------------------------

def _criar_avaliador_llm(ChatGroq, LangchainLLMWrapper):
    """Cria o LLM avaliador usando Groq."""
    groq_key = os.getenv("GROQ_API_KEY")
    if not groq_key:
        raise EnvironmentError(
            "GROQ_API_KEY não encontrada no .env. "
            "O RAGAS precisa de um LLM para avaliar."
        )
    modelo = os.getenv("LLM_FALLBACK_MODEL", "llama-3.3-70b-versatile")
    llm = ChatGroq(model=modelo, temperature=0)
    return LangchainLLMWrapper(llm)


def _criar_avaliador_embeddings(OpenAIEmbeddings, LangchainEmbeddingsWrapper):
    """
    Cria o modelo de embeddings para ResponseRelevancy.
    Usa OpenAI quando OPENAI_API_KEY disponível; caso contrário, embeddings locais.
    """
    openai_key = os.getenv("OPENAI_API_KEY")
    if openai_key:
        emb = OpenAIEmbeddings(model="text-embedding-3-small")
        return LangchainEmbeddingsWrapper(emb)

    log.warning(
        "OPENAI_API_KEY não encontrada — ResponseRelevancy usará embeddings locais. "
        "Para melhor qualidade, adicione OPENAI_API_KEY ao .env."
    )
    try:
        from langchain_community.embeddings import HuggingFaceEmbeddings
        emb = HuggingFaceEmbeddings(
            model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        )
        return LangchainEmbeddingsWrapper(emb)
    except ImportError:
        log.warning("langchain-community não instalado — ResponseRelevancy desabilitada.")
        return None


# ---------------------------------------------------------------------------
# Execução principal da avaliação
# ---------------------------------------------------------------------------

def executar_avaliacao(
    perguntas: list[dict],
    n_resultados: int = 5,
    saida: Optional[Path] = None,
    verbose: bool = False,
) -> dict:
    """
    Executa o pipeline de avaliação completo.

    Sistema de avaliação:
      1. Para cada pergunta do banco, executa o pipeline RAG completo.
      2. Coleta (pergunta, resposta, contextos_recuperados) em amostras RAGAS.
      3. Chama `ragas.evaluate()` com as métricas configuradas.
      4. Mescla os scores RAGAS com os metadados coletados.
      5. Salva CSV com resultado detalhado + retorna métricas agregadas.

    Parâmetros:
        perguntas    — lista de dicts do banco_perguntas
        n_resultados — top-k chunks por pergunta (padrão: 5)
        saida        — caminho do CSV de resultado (None = gera automático em docs/)
        verbose      — imprime cada pergunta/resposta no terminal

    Retorna:
        dict com métricas agregadas, caminho do arquivo salvo e lista de erros
    """
    (
        EvaluationDataset,
        evaluate,
        SingleTurnSample,
        Faithfulness,
        ResponseRelevancy,
        LLMContextPrecisionWithoutReference,
        LangchainLLMWrapper,
        LangchainEmbeddingsWrapper,
        ChatGroq,
        OpenAIEmbeddings,
        RunConfig,
    ) = _importar_ragas()

    log.info("Conectando ao Qdrant...")
    qclient = conectar_qdrant()
    bm25, bm25_ids = carregar_indices(qclient)
    log.info(f"Índices carregados — {len(bm25_ids)} documentos no BM25.")

    avaliador_llm = _criar_avaliador_llm(ChatGroq, LangchainLLMWrapper)
    avaliador_emb = _criar_avaliador_embeddings(OpenAIEmbeddings, LangchainEmbeddingsWrapper)

    metricas = [
        Faithfulness(llm=avaliador_llm),
        LLMContextPrecisionWithoutReference(llm=avaliador_llm),
    ]
    if avaliador_emb is not None:
        metricas.append(ResponseRelevancy(llm=avaliador_llm, embeddings=avaliador_emb))

    log.info(f"Métricas ativas: {[m.__class__.__name__ for m in metricas]}")

    amostras_ragas: list[SingleTurnSample] = []
    registros_detalhados: list[dict] = []
    erros: list[dict] = []

    total = len(perguntas)
    log.info(f"Iniciando avaliação — {total} perguntas...")

    for idx, item in enumerate(perguntas, 1):
        pid = item["id"]
        pergunta = item["pergunta"]
        filtros = item.get("filtros")

        log.info(f"[{idx}/{total}] {pid}: {pergunta[:70]}")
        inicio = time.monotonic()

        try:
            resposta, contextos, chunks_meta = rodar_rag(
                pergunta=pergunta,
                filtros=filtros,
                qclient=qclient,
                bm25=bm25,
                bm25_ids=bm25_ids,
                n_resultados=n_resultados,
            )
        except Exception as e:
            log.error(f"  [ERRO] {pid}: {e}")
            erros.append({"id": pid, "pergunta": pergunta, "erro": str(e)})
            continue

        latencia_ms = int((time.monotonic() - inicio) * 1000)

        if verbose:
            print(f"\n{'='*60}")
            print(f"[{pid}] {pergunta}")
            print(f"Contextos recuperados: {len(contextos)}")
            print(f"Resposta: {resposta[:300]}...")
            print(f"Latência: {latencia_ms}ms")

        # Aguarda entre perguntas para respeitar rate limit do Groq (free tier)
        if idx < total:
            log.info("  Aguardando 4s (rate limit Groq)...")
            time.sleep(4)

        amostra = SingleTurnSample(
            user_input=pergunta,
            response=resposta,
            retrieved_contexts=contextos if contextos else [""],
        )
        amostras_ragas.append(amostra)

        registros_detalhados.append({
            "id": pid,
            "categoria": item["categoria"].value if hasattr(item["categoria"], "value") else item["categoria"],
            "tipo_busca": item["tipo_busca"].value if hasattr(item["tipo_busca"], "value") else item["tipo_busca"],
            "tipo_esperado": item["tipo_esperado"].value if hasattr(item["tipo_esperado"], "value") else item["tipo_esperado"],
            "dificuldade": item["dificuldade"].value if hasattr(item["dificuldade"], "value") else item["dificuldade"],
            "pergunta": pergunta,
            "resposta": resposta,
            "n_contextos": len(contextos),
            "latencia_ms": latencia_ms,
            "fontes": json.dumps(
                [
                    {
                        "numero": c.get("numero"),
                        "tipo": c.get("tipo_nome"),
                        "ano": c.get("ano"),
                        "score": round(c.get("score_final", 0), 4),
                    }
                    for c in chunks_meta
                ],
                ensure_ascii=False,
            ),
            "faithfulness": None,
            "response_relevancy": None,
            "context_precision": None,
        })

    if not amostras_ragas:
        log.error("Nenhuma amostra coletada — verifique erros acima.")
        return {"erro": "Sem amostras para avaliar"}

    log.info(f"\nExecutando RAGAS evaluate em {len(amostras_ragas)} amostras...")
    log.info("Usando max_workers=1 para respeitar rate limit do Groq...")
    dataset = EvaluationDataset(samples=amostras_ragas)

    try:
        if RunConfig is not None:
            cfg = RunConfig(max_workers=1, max_wait=60, timeout=120)
            resultado_ragas = evaluate(
                dataset=dataset,
                metrics=metricas,
                raise_exceptions=False,
                run_config=cfg,
            )
        else:
            resultado_ragas = evaluate(
                dataset=dataset,
                metrics=metricas,
                raise_exceptions=False,
            )
    except Exception as e:
        log.error(f"Erro no RAGAS evaluate: {e}")
        raise

    import pandas as pd

    df_ragas = resultado_ragas.to_pandas()

    col_faith = "faithfulness" if "faithfulness" in df_ragas.columns else None
    col_relev = "response_relevancy" if "response_relevancy" in df_ragas.columns else None
    col_prec  = "llm_context_precision_without_reference" if "llm_context_precision_without_reference" in df_ragas.columns else None

    for i, reg in enumerate(registros_detalhados):
        if i < len(df_ragas):
            row = df_ragas.iloc[i]
            if col_faith: reg["faithfulness"]       = round(float(row[col_faith]), 4)
            if col_relev: reg["response_relevancy"] = round(float(row[col_relev]), 4)
            if col_prec:  reg["context_precision"]  = round(float(row[col_prec]),  4)

    df_resultado = pd.DataFrame(registros_detalhados)

    if saida is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        saida = RAIZ / "docs" / f"ragas_resultado_{ts}.csv"

    saida = Path(saida)
    saida.parent.mkdir(parents=True, exist_ok=True)
    df_resultado.to_csv(saida, index=False, encoding="utf-8-sig")
    log.info(f"Resultado salvo em: {saida}")

    metricas_agg: dict = {}
    for col in ["faithfulness", "response_relevancy", "context_precision"]:
        vals = df_resultado[col].dropna()
        if len(vals) > 0:
            metricas_agg[col] = {
                "media": round(float(vals.mean()), 4),
                "min":   round(float(vals.min()),  4),
                "max":   round(float(vals.max()),  4),
                "n":     len(vals),
            }

    _imprimir_sumario(metricas_agg, df_resultado, erros, saida)

    return {"metricas": metricas_agg, "arquivo": str(saida), "erros": erros}


# ---------------------------------------------------------------------------
# Utilitários de apresentação
# ---------------------------------------------------------------------------

def _imprimir_sumario(
    metricas_agg: dict,
    df,
    erros: list[dict],
    saida: Path,
) -> None:
    """Imprime o sumário de avaliação no terminal."""
    print("\n" + "=" * 65)
    print("SUMÁRIO — AVALIAÇÃO RAGAS · RAG ANEEL")
    print("=" * 65)

    print(f"\nAmostras avaliadas : {len(df)}")
    print(f"Erros              : {len(erros)}")

    if metricas_agg:
        print("\nMétricas globais:")
        nomes = {
            "faithfulness":       "Faithfulness          ",
            "response_relevancy": "Response Relevancy    ",
            "context_precision":  "Context Precision     ",
        }
        for metrica, stats in metricas_agg.items():
            nome = nomes.get(metrica, metrica)
            print(
                f"  {nome}  média={stats['media']:.4f}  "
                f"min={stats['min']:.4f}  max={stats['max']:.4f}  (n={stats['n']})"
            )

    if "categoria" in df.columns and "faithfulness" in df.columns:
        print("\nFaithfulness por categoria:")
        for cat, grupo in df.groupby("categoria"):
            vals = grupo["faithfulness"].dropna()
            if len(vals) > 0:
                print(f"  {cat:<20}  {vals.mean():.4f}  (n={len(vals)})")

    if "tipo_esperado" in df.columns and "faithfulness" in df.columns:
        print("\nFaithfulness por tipo esperado:")
        for tipo, grupo in df.groupby("tipo_esperado"):
            vals = grupo["faithfulness"].dropna()
            if len(vals) > 0:
                print(f"  {tipo:<30}  {vals.mean():.4f}  (n={len(vals)})")

    if erros:
        print(f"\nPerguntas com erro ({len(erros)}):")
        for e in erros:
            print(f"  [{e['id']}] {e['erro'][:80]}")

    print(f"\nResultado completo em: {saida}")
    print("=" * 65)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parsear_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Avaliação RAGAS do sistema RAG ANEEL",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--limite", type=int, default=None,
        help="Número máximo de perguntas a avaliar (padrão: todas)",
    )
    parser.add_argument(
        "--categoria",
        choices=[c.value for c in Categoria],
        default=None,
        help="Filtrar por categoria temática",
    )
    parser.add_argument(
        "--tipo",
        choices=[t.value for t in TipoResposta],
        default=None,
        help="Filtrar por tipo de resposta esperada",
    )
    parser.add_argument(
        "--top-k", type=int, default=5,
        help="Número de chunks recuperados por pergunta (padrão: 5)",
    )
    parser.add_argument(
        "--saida", type=str, default=None,
        help="Caminho do CSV de resultado (padrão: docs/ragas_resultado_<timestamp>.csv)",
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Imprime pergunta/resposta de cada amostra",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    args = _parsear_args()

    perguntas = list(BANCO)

    if args.categoria:
        cat = Categoria(args.categoria)
        perguntas = [p for p in perguntas if p["categoria"] == cat]
        log.info(f"Filtro categoria={args.categoria} → {len(perguntas)} perguntas")

    if args.tipo:
        tipo = TipoResposta(args.tipo)
        perguntas = [p for p in perguntas if p["tipo_esperado"] == tipo]
        log.info(f"Filtro tipo={args.tipo} → {len(perguntas)} perguntas")

    if args.limite:
        perguntas = perguntas[: args.limite]
        log.info(f"Limite={args.limite} → {len(perguntas)} perguntas")

    if not perguntas:
        log.error("Nenhuma pergunta selecionada com os filtros aplicados.")
        sys.exit(1)

    log.info(f"Total de perguntas para avaliação: {len(perguntas)}")

    executar_avaliacao(
        perguntas=perguntas,
        n_resultados=args.top_k,
        saida=Path(args.saida) if args.saida else None,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()
