"""The headline numbers beside "O que é o Aleph Quant" (the structural redesign of 2026-11-06): read from the same
files the sections below show, each with its source, never typed in."""

import pandas as pd
import pytest

from ui import destaques


def _comparacao() -> pd.DataFrame:
    dias = pd.to_datetime(["2010-01-04", "2015-01-05", "2020-01-06", "2026-01-05"])
    return pd.DataFrame({"Composite": [100.0, 150.0, 120.0, 400.0], "BOVA11": [100.0, 90, 110, 150]}, index=dias)


def test_the_composite_return_a_year_and_its_worst_fall_come_from_the_comparison_curve():
    numeros = destaques.numeros(_comparacao(), conta=10067.6, acoes=15)
    # the same convention as the tables below (core.metrics, 252 trading days a year): one number, one value
    from core.metrics import cagr
    assert numeros[0].valor == pytest.approx(cagr(_comparacao()["Composite"]))
    assert numeros[1].valor == pytest.approx(120 / 150 - 1)
    assert [n.rotulo for n in numeros] == ["Composite ao ano", "Queda máxima", "Conta Composite (R$)",
                                           "Ações na carteira"]
    assert "2010" in numeros[0].fonte and "2026" in numeros[0].fonte


def test_the_numbers_read_in_brazilian_format():
    textos = [n.texto for n in destaques.numeros(_comparacao(), conta=10067.6, acoes=15)]
    assert textos[1].startswith("\u2212") and textos[1].endswith("%") and "," in textos[1]
    assert textos[2] == "10.067,60" and textos[3] == "15"                   # the unit lives in the label


def test_missing_sources_drop_their_tiles_instead_of_inventing_numbers():
    assert destaques.numeros(None, conta=None, acoes=None) == []
    assert [n.rotulo for n in destaques.numeros(None, conta=None, acoes=15)] == ["Ações na carteira"]


def test_the_tiles_say_their_period_and_terms_on_screen_and_round_like_the_simulator():
    # the second finish review: a real-money tool's first viewport keeps each claim's period and terms visible
    itens = destaques.numeros(_comparacao(), conta=10067.62, acoes=15)
    assert itens[2].texto == "10.067,62" and itens[2].rotulo == "Conta Composite (R$)"   # the simulator's rounding
    legenda = destaques.legenda(_comparacao())
    assert "04/01/2010" in legenda and "05/01/2026" in legenda
    assert "depois de custos" in legenda and "antes do Imposto de Renda" in legenda and "03/11/2026" in legenda
    assert destaques.legenda(None).startswith("Conta fictícia")


def test_the_backtest_section_says_after_costs_like_the_engine_that_made_its_curves():
    # the finish review's data check (2026-11-06): the live curves come from attempt 51's loop with CFG_BT, which charges
    # 10 bps a side, yet the section said "antes de custos"; the comparison (same engine) said "depois"
    from pathlib import Path
    from scripts.avaliar_universo_dinamico import CFG_BT
    fonte = (Path(__file__).resolve().parents[1] / "ui" / "alepha.py").read_text(encoding="utf-8")
    assert CFG_BT.custo_bps > 0
    assert "antes de custos" not in fonte and fonte.count("depois de custos") >= 2
