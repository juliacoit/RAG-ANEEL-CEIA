"""
conftest.py
===========
Fixtures compartilhadas para os testes pytest do RAG ANEEL.

Uso:
    pytest tests/
    pytest tests/ -k "banco_perguntas"
    pytest tests/ -v
"""

import sys
from pathlib import Path

import pytest

# Garante que a raiz do projeto está no path, independente de onde pytest é chamado
RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))


# ---------------------------------------------------------------------------
# Fixtures: banco de perguntas
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def banco():
    """Retorna o banco completo de 45+ perguntas."""
    from src.evaluation.banco_perguntas import BANCO
    return BANCO


@pytest.fixture(scope="session")
def perguntas_fallback(banco):
    """Retorna apenas as perguntas de fallback (devem ser recusadas pelo RAG)."""
    from src.evaluation.banco_perguntas import TipoResposta
    return [p for p in banco if p["tipo_esperado"] == TipoResposta.FALLBACK]


@pytest.fixture(scope="session")
def perguntas_faceis(banco):
    """Retorna apenas as perguntas de dificuldade fácil."""
    from src.evaluation.banco_perguntas import Dificuldade
    return [p for p in banco if p["dificuldade"] == Dificuldade.FACIL]


# ---------------------------------------------------------------------------
# Fixtures: conexão Qdrant (skip automático se Docker não estiver rodando)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def qdrant_client():
    """
    Tenta conectar ao Qdrant local. Pula os testes de integração se não estiver disponível.
    Garante que o mesmo cliente é reutilizado em toda a sessão.
    """
    pytest.importorskip("qdrant_client", reason="qdrant-client não instalado")

    try:
        import urllib.request
        urllib.request.urlopen("http://localhost:6333/collections", timeout=3)
    except Exception:
        pytest.skip("Qdrant não está rodando — inicie com: docker compose up -d")

    from src.search.indexar import conectar_qdrant
    return conectar_qdrant()


@pytest.fixture(scope="session")
def indices_bm25(qdrant_client):
    """Carrega os índices BM25 e retorna (bm25, bm25_ids). Depende do Qdrant estar rodando."""
    from src.search.indexar import carregar_indices
    return carregar_indices(qdrant_client)


# ---------------------------------------------------------------------------
# Fixtures: configuração de avaliação
# ---------------------------------------------------------------------------

@pytest.fixture
def eval_config() -> dict:
    """Configuração padrão para rodar avaliações de forma reproduzível."""
    return {
        "n_resultados": 5,
        "limite": 5,        # Avalia apenas 5 perguntas por padrão nos testes
        "verbose": False,
    }
