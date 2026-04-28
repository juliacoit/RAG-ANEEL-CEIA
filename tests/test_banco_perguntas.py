"""
test_banco_perguntas.py
=======================
Testes unitários para o banco de perguntas.
Não requerem Docker, LLM ou conexão externa — rodam sempre.
"""

import pytest
from src.evaluation.banco_perguntas import (
    BANCO,
    Categoria,
    Dificuldade,
    Metrica,
    TipoBusca,
    TipoResposta,
    com_filtros,
    por_categoria,
    por_dificuldade,
    por_metrica,
    por_tipo_busca,
    por_tipo_resposta,
    resumo,
)


# ---------------------------------------------------------------------------
# Estrutura e integridade do banco
# ---------------------------------------------------------------------------

def test_banco_nao_vazio():
    assert len(BANCO) > 0, "O banco de perguntas não pode estar vazio"


def test_banco_ids_unicos():
    ids = [p["id"] for p in BANCO]
    assert len(ids) == len(set(ids)), "IDs duplicados no banco de perguntas"


def test_banco_campos_obrigatorios():
    campos = {"id", "pergunta", "categoria", "tipo_busca", "tipo_esperado",
              "dificuldade", "metricas_alvo", "notas"}
    for p in BANCO:
        faltando = campos - p.keys()
        assert not faltando, f"[{p['id']}] campos faltando: {faltando}"


def test_banco_pergunta_nao_vazia():
    for p in BANCO:
        assert p["pergunta"].strip(), f"[{p['id']}] pergunta vazia"


def test_banco_metricas_alvo_lista():
    for p in BANCO:
        assert isinstance(p["metricas_alvo"], list), f"[{p['id']}] metricas_alvo deve ser lista"
        assert len(p["metricas_alvo"]) > 0, f"[{p['id']}] metricas_alvo não pode ser vazia"


def test_banco_tipos_enum():
    for p in BANCO:
        assert isinstance(p["categoria"],    Categoria),    f"[{p['id']}] categoria inválida"
        assert isinstance(p["tipo_busca"],   TipoBusca),    f"[{p['id']}] tipo_busca inválido"
        assert isinstance(p["tipo_esperado"],TipoResposta), f"[{p['id']}] tipo_esperado inválido"
        assert isinstance(p["dificuldade"],  Dificuldade),  f"[{p['id']}] dificuldade inválida"
        for m in p["metricas_alvo"]:
            assert isinstance(m, Metrica), f"[{p['id']}] métrica inválida: {m}"


def test_banco_filtros_dict_ou_none():
    for p in BANCO:
        assert p.get("filtros") is None or isinstance(p["filtros"], dict), \
            f"[{p['id']}] filtros deve ser dict ou None"


# ---------------------------------------------------------------------------
# Utilitários de filtragem
# ---------------------------------------------------------------------------

def test_por_categoria_microgeracao():
    resultado = por_categoria(Categoria.MICROGERACAO)
    assert len(resultado) >= 1
    assert all(p["categoria"] == Categoria.MICROGERACAO for p in resultado)


def test_por_tipo_resposta_fallback():
    resultado = por_tipo_resposta(TipoResposta.FALLBACK)
    assert len(resultado) >= 1
    assert all(p["tipo_esperado"] == TipoResposta.FALLBACK for p in resultado)


def test_por_dificuldade():
    for dif in Dificuldade:
        resultado = por_dificuldade(dif)
        assert all(p["dificuldade"] == dif for p in resultado)


def test_por_metrica():
    resultado = por_metrica(Metrica.FAITHFULNESS)
    assert len(resultado) >= 1
    assert all(Metrica.FAITHFULNESS in p["metricas_alvo"] for p in resultado)


def test_por_tipo_busca():
    for tb in TipoBusca:
        resultado = por_tipo_busca(tb)
        assert all(p["tipo_busca"] == tb for p in resultado)


def test_com_filtros():
    resultado = com_filtros()
    assert len(resultado) >= 1
    assert all(p["filtros"] is not None for p in resultado)


def test_resumo_estrutura():
    stats = resumo()
    assert "total" in stats
    assert "por_categoria" in stats
    assert "por_tipo_resposta" in stats
    assert "por_dificuldade" in stats
    assert "por_tipo_busca" in stats
    assert "com_filtros" in stats
    assert stats["total"] == len(BANCO)


def test_resumo_soma_categoria():
    stats = resumo()
    total_por_cat = sum(stats["por_categoria"].values())
    assert total_por_cat == stats["total"]


def test_resumo_soma_tipo_resposta():
    stats = resumo()
    total_por_tipo = sum(stats["por_tipo_resposta"].values())
    assert total_por_tipo == stats["total"]
