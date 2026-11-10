from __future__ import annotations

import numpy as np
import pytest

from drl import relatorio


def _curva(crescimento: float, sessoes: int = 252) -> np.ndarray:
    return 100.0 * (1.0 + crescimento) ** np.arange(sessoes + 1)


def test_a_curve_that_grows_one_percent_a_session_annualises_to_the_compounded_rate():
    m = relatorio.metricas(_curva(0.01))
    assert m["retorno_anual"] == pytest.approx(1.01 ** 252 - 1.0, rel=1e-9)
    assert m["total"] == pytest.approx(1.01 ** 252 - 1.0, rel=1e-9)
    assert m["mdd"] == 0.0


def test_the_drawdown_is_measured_from_the_running_peak():
    curva = np.array([100.0, 120.0, 60.0, 90.0])
    m = relatorio.metricas(curva)
    assert m["mdd"] == pytest.approx(-0.5)          # 120 -> 60


def test_a_flat_curve_has_no_sharpe_and_no_volatility():
    m = relatorio.metricas(np.full(253, 100.0))
    assert m["vol_anual"] == 0.0
    assert m["sharpe"] == 0.0
    assert m["retorno_anual"] == 0.0


def test_the_sharpe_is_annualised_and_uses_the_risk_free_rate():
    rng = np.random.default_rng(3)
    r = rng.normal(0.0005, 0.01, 2520)
    curva = 100.0 * np.cumprod(np.append(1.0, 1.0 + r))
    m = relatorio.metricas(curva)
    esperado = (r.mean() - 0.0) / r.std(ddof=1) * np.sqrt(252)
    assert m["sharpe"] == pytest.approx(esperado, rel=1e-9)
    sem_taxa = m["sharpe"]
    assert relatorio.metricas(curva, taxa=0.0001)["sharpe"] < sem_taxa


def test_the_turnover_is_annualised_from_the_session_average():
    m = relatorio.metricas(_curva(0.0, 10), giros=np.full(10, 0.01))
    assert m["giro_anual"] == pytest.approx(0.01 * 252)


def test_an_empty_curve_returns_zeros_instead_of_raising():
    m = relatorio.metricas(np.array([100.0]))
    assert m["sharpe"] == 0.0 and m["mdd"] == 0.0
