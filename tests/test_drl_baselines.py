from __future__ import annotations

import numpy as np
import pytest

from drl import baselines


def test_equal_weights_sum_to_one_and_are_long_only():
    w = baselines.pesos_iguais(4)
    assert w.sum() == pytest.approx(1.0)
    assert (w >= 0).all()


def test_the_shrunk_covariance_is_positive_definite_when_the_window_is_short():
    ret = np.array([[0.01, 0.02, np.nan], [0.0, 0.01, np.nan], [0.02, 0.0, np.nan]])
    ret = np.nan_to_num(ret)
    cov = baselines.covariancia_encolhida(ret)
    assert np.all(np.linalg.eigvalsh(cov) > 0)


def test_minimum_variance_puts_most_weight_on_the_quiet_name():
    rng = np.random.default_rng(0)
    ret = np.column_stack([rng.normal(0, 0.01, 500), rng.normal(0, 0.04, 500), rng.normal(0, 0.02, 500)])
    w = baselines.pesos_min_variancia(ret)
    assert w.sum() == pytest.approx(1.0)
    assert w[0] == w.max()
    assert np.argmax(w) == 0


def test_the_sharpe_portfolio_prefers_the_better_risk_adjusted_name():
    rng = np.random.default_rng(1)
    ret = np.column_stack([rng.normal(0.0010, 0.01, 800), rng.normal(0.0001, 0.01, 800)])
    w = baselines.pesos_sharpe(ret)
    assert w.sum() == pytest.approx(1.0)
    assert w[0] > 5 * w[1]                       # the name with ten times the mean return gets the bulk of it


def test_the_curve_applies_the_returns_and_the_turnover_cost():
    retornos = np.array([[0.01], [0.01]])
    pesos = [np.array([1.0]), np.array([1.0])]
    curva = baselines.curva(pesos, retornos)
    assert curva[-1] == pytest.approx(1.01 ** 2)
    curva_com_custo = baselines.curva(pesos, retornos, custo_bps=10.0)
    assert curva_com_custo[-1] < curva[-1]                       # the first trade from cash costs 2 x 0.10%


def test_a_long_only_weights_path_costs_only_its_own_turnover():
    retornos = np.array([[0.0], [0.0], [0.0]])
    pesos = [np.array([0.5]), np.array([0.5]), np.array([0.5])]
    curva = baselines.curva(pesos, retornos, custo_bps=10.0)
    assert curva[-1] == pytest.approx(1.0 - 0.5 * 0.001)          # one trade of half the portfolio and nothing after
