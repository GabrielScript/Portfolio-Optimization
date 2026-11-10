from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from core import vol_target


def _retornos(valores: list[float]) -> pd.Series:
    return pd.Series(valores, index=pd.bdate_range("2020-01-02", periods=len(valores)), dtype=float)


def test_a_calm_market_keeps_full_exposure():
    """Realised vol below the target must not lever up: the portfolio is long-only and never borrows."""
    r = _retornos([0.0001] * 40)
    e = vol_target.exposicao(r, alvo=0.20, janela=10)
    assert (e <= 1.0 + 1e-12).all()
    assert e.iloc[-1] == pytest.approx(1.0)


def test_a_volatile_market_cuts_exposure_proportionally():
    rng = np.random.default_rng(7)
    r = _retornos(list(rng.normal(0.0, 0.05, 40)))
    e = vol_target.exposicao(r, alvo=0.20, janela=10)
    vol = r.rolling(10).std().iloc[-2] * np.sqrt(252)      # yesterday's window: today's decision cannot see today
    assert e.iloc[-1] == pytest.approx(0.20 / vol, rel=1e-9)


def test_the_exposure_uses_only_the_past():
    """The decision for session t is taken with returns up to t-1: changing a future return must not move it."""
    r = _retornos(list(np.random.default_rng(3).normal(0.001, 0.02, 40)))
    alt = r.copy()
    alt.iloc[-1] = 0.9                      # a huge return today
    assert vol_target.exposicao(r, 0.20, 10).iloc[-2] == pytest.approx(
        vol_target.exposicao(alt, 0.20, 10).iloc[-2])


def test_exposure_is_never_negative_and_never_levered():
    r = _retornos(list(np.random.default_rng(11).normal(0.0, 0.2, 60)))
    e = vol_target.exposicao(r, alvo=0.05, janela=5)
    assert (e >= 0.0).all() and (e <= 1.0 + 1e-12).all()


def test_a_flat_exposure_reproduces_the_original_returns():
    r = _retornos([0.01, -0.02, 0.03, 0.0])
    cdi = pd.Series(0.0003, index=r.index)
    x = vol_target.aplicar(r, cdi, alvo=10.0, janela=2, custo_bps=10.0)
    assert np.allclose(x.fillna(0.0), r)


def test_the_cash_leg_earns_the_cdi():
    r = _retornos([0.02] * 5)
    cdi = pd.Series(0.001, index=r.index)
    exposto = pd.Series([0.5] * 5, index=r.index)
    x = vol_target.aplicar(r, cdi, alvo=0.20, janela=2, custo_bps=0.0, caminho=exposto)
    assert x.iloc[2] == pytest.approx(0.5 * 0.02 + 0.5 * 0.001)


def test_changing_the_exposure_costs_turnover():
    r = _retornos([0.0] * 5)
    cdi = pd.Series(0.0, index=r.index)
    exposto = pd.Series([1.0, 0.5, 0.5, 0.5, 0.5], index=r.index)
    com_custo = vol_target.aplicar(r, cdi, alvo=1.0, janela=1, custo_bps=10.0, caminho=exposto)
    assert com_custo.iloc[1] == pytest.approx(-0.5 * 10.0 / 1e4)
    assert com_custo.iloc[2] == pytest.approx(0.0)


def test_a_target_above_half_a_percent_a_year_is_required():
    with pytest.raises(ValueError):
        vol_target.exposicao(_retornos([0.01] * 10), alvo=0.0, janela=5)


def test_the_overlay_on_an_engine_curve_keeps_the_curve_when_the_market_is_calm():
    valores = pd.Series([100.0, 100.1, 100.2, 100.3, 100.4], index=pd.bdate_range("2020-01-02", periods=5))
    cdi = pd.Series(0.0003, index=valores.index)
    curva, _ = vol_target.sobrepor(valores, cdi, alvo=5.0, janela=2)
    assert curva.iloc[0] == pytest.approx(100.0)
    assert curva.iloc[-1] == pytest.approx(100.4, rel=1e-9)


def test_the_tax_ceiling_only_charges_the_de_levering():
    e = pd.Series([1.0, 1.0, 0.5, 0.5], index=pd.bdate_range("2020-01-02", periods=4))
    valores = pd.Series([100.0, 100.0, 100.0, 100.0], index=e.index)
    assert vol_target.imposto_de_estimativa(valores, e) == pytest.approx(0.5 * 100.0 * 0.15)
