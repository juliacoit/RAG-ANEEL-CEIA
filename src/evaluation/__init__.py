"""
src/evaluation
==============
Módulo de avaliação do sistema RAG ANEEL.

Componentes:
  banco_perguntas — 45+ perguntas curadas em 10 categorias temáticas
  eval_runner     — executa avaliação RAGAS e persiste resultados em CSV
"""

from .banco_perguntas import BANCO, Categoria, TipoBusca, TipoResposta, Dificuldade, Metrica

__all__ = [
    "BANCO",
    "Categoria",
    "TipoBusca",
    "TipoResposta",
    "Dificuldade",
    "Metrica",
]
