"""MARS on the B3 (docs/superpowers/specs/2026-11-08-mars-design.md): the overlay, the risk score and the reward."""

from __future__ import annotations

import numpy as np
import pytest

from qpo_kaggle.mars import Execucao, executar, penalidade, risco_env

CFG = Execucao(passo_maximo=0.10, teto=0.20, colchao=0.02, custo=0.001)


def test_a_sale_is_capped_at_the_holding():
    x = np.array([0.05, 0.0])
    novo, caixa, custo, _ = executar(x, 0.95, np.array([-1.0, 0.0]), np.ones(2, bool), np.zeros(2, bool), CFG)
    assert novo[0] == 0.0
    assert caixa == pytest.approx(0.95 + 0.05 * (1 - 0.001))
    assert custo == pytest.approx(0.05 * 0.001)


def test_a_buy_stops_at_the_position_cap():
    x = np.array([0.15, 0.0])
    novo, *_ = executar(x, 0.85, np.array([1.0, 0.0]), np.ones(2, bool), np.zeros(2, bool), CFG)
    assert novo[0] == pytest.approx(0.20)


def test_buys_shrink_to_keep_the_cash_buffer_after_costs():
    nomes = 12                                    # 12 buys of 10% of V each ask for 120% of V
    _, caixa, *_ = executar(np.zeros(nomes), 1.0, np.ones(nomes), np.ones(nomes, bool), np.zeros(nomes, bool), CFG)
    assert caixa == pytest.approx(0.02)


def test_a_name_leaving_tomorrow_is_sold_whole():
    x = np.array([0.10, 0.0])
    novo, *_ = executar(x, 0.90, np.array([1.0, 0.0]), np.ones(2, bool), np.array([True, False]), CFG)
    assert novo[0] == 0.0


def test_a_name_not_listed_today_is_not_bought():
    novo, *_ = executar(np.zeros(2), 1.0, np.ones(2), np.array([False, True]), np.zeros(2, bool), CFG)
    assert novo[0] == 0.0


def test_the_executed_action_is_in_the_action_units():
    _, _, _, feita = executar(np.zeros(2), 1.0, np.array([0.5, 0.0]), np.ones(2, bool), np.zeros(2, bool), CFG)
    assert feita[0] == pytest.approx(0.5)


def test_the_risk_score_of_all_cash_is_zero():
    assert risco_env(np.zeros(4), 1.0, np.zeros(4), np.eye(4) * 1e-4, CFG) == 0.0


def test_the_risk_score_counts_a_single_name_as_full_concentration():
    x = np.array([1.0, 0.0, 0.0, 0.0])
    assert risco_env(x, 0.0, np.zeros(4), np.zeros((4, 4)), CFG) == pytest.approx(0.3)


def test_the_risk_score_reads_buys_beyond_cash_as_leverage():
    x = np.full(10, 0.1)                          # fully invested, then +50% of V in buys: weights 0.15 each
    concentracao = (10 * 0.15**2 - 0.1) / 0.9
    assert risco_env(x, 0.0, np.full(10, 0.5), np.zeros((10, 10)), CFG) == pytest.approx(0.3 * 0.5 + 0.3 * concentracao)


def test_the_risk_score_scales_volatility_by_half():
    x = np.array([0.5, 0.0])                      # half in a name of 40% a year: 20% / 50% = 0.4
    cov = np.diag([0.4**2 / 252, 0.0])
    assert risco_env(x, 0.5, np.zeros(2), cov, CFG) == pytest.approx(0.4 * 0.4)   # HHI 0.25 < 1/D: no concentration


def test_the_penalty_adds_half_the_volatility_and_twice_the_drawdown():
    assert penalidade(np.array([1.0, 1.1, 0.99])) == pytest.approx(0.5 * 0.1 + 2.0 * 0.1)
