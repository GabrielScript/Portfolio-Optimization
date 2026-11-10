"""Tests for the forward-holdout ledger helpers (no lake, no model)."""

from __future__ import annotations

import pandas as pd
import pytest

from scripts.qml_holdout import (comprar, primeira_sessao, rebalancear, rebalanceio_devido,
                                 valorizar)


def test_primeira_sessao_is_the_first_on_or_after_the_start() -> None:
    sessoes = pd.bdate_range("2026-11-02", periods=10)
    assert primeira_sessao(sessoes, pd.Timestamp("2026-11-09")) == pd.Timestamp("2026-11-09")
    assert primeira_sessao(sessoes, pd.Timestamp("2026-11-07")) == pd.Timestamp("2026-11-09")


def test_primeira_sessao_raises_when_the_window_has_not_started() -> None:
    sessoes = pd.bdate_range("2026-11-02", periods=3)
    with pytest.raises(ValueError):
        primeira_sessao(sessoes, pd.Timestamp("2027-01-01"))


def test_rebalanceio_devido_counts_sessions_not_days() -> None:
    sessoes = pd.bdate_range("2026-11-02", periods=200)
    assert not rebalanceio_devido(sessoes, sessoes[62], sessoes[0], 63)
    assert rebalanceio_devido(sessoes, sessoes[63], sessoes[0], 63)


def test_buying_charges_the_cost_and_keeps_the_weights() -> None:
    pesos = pd.Series({"AAAA3": 0.5, "BBBB3": 0.5})
    precos = pd.Series({"AAAA3": 10.0, "BBBB3": 20.0})
    cotas = comprar(pesos, precos, 10_000.0, custo_bps=10.0)
    assert valorizar(cotas, precos) == pytest.approx(10_000.0 * (1 - 10 / 10_000))
    assert cotas["AAAA3"] * 10.0 == pytest.approx(cotas["BBBB3"] * 20.0)


def test_buying_raises_on_a_missing_or_zero_price() -> None:
    pesos = pd.Series({"AAAA3": 1.0})
    with pytest.raises(ValueError):
        comprar(pesos, pd.Series({"AAAA3": 0.0}), 1_000.0)
    with pytest.raises(ValueError):
        comprar(pesos, pd.Series(dtype=float), 1_000.0)


def test_rebalancing_to_the_same_book_costs_nothing() -> None:
    pesos = pd.Series({"AAAA3": 0.5, "BBBB3": 0.5})
    precos = pd.Series({"AAAA3": 10.0, "BBBB3": 20.0})
    cotas = comprar(pesos, precos, 1_000.0, custo_bps=0.0)
    novas, giro = rebalancear(cotas, pesos, precos, custo_bps=10.0)
    assert giro == pytest.approx(0.0, abs=1e-9)
    assert valorizar(novas, precos) == pytest.approx(1_000.0)


def test_rebalancing_charges_only_on_the_turnover() -> None:
    precos = pd.Series({"AAAA3": 10.0, "BBBB3": 20.0})
    cotas = comprar(pd.Series({"AAAA3": 1.0, "BBBB3": 0.0}), precos, 1_000.0, custo_bps=0.0)
    alvo = pd.Series({"AAAA3": 0.5, "BBBB3": 0.5})
    novas, giro = rebalancear(cotas, alvo, precos, custo_bps=10.0)
    assert giro == pytest.approx(1.0)                       # half sold, half bought
    assert valorizar(novas, precos) == pytest.approx(1_000.0 * (1 - 10 / 10_000))
