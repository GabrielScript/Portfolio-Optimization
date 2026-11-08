"""The four technical indicators of the MARS state (spec section 4): MACD, RSI, CCI, ADX."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from core.mars_indicadores import adx, cci, macd_relativo, rsi

DIAS = pd.bdate_range("2020-01-01", periods=120)


def test_rsi_of_a_series_that_only_rises_is_100():
    subindo = pd.Series(np.arange(1.0, 121.0), index=DIAS)
    assert rsi(subindo, 30).iloc[-1] == pytest.approx(100.0)


def test_macd_of_a_constant_price_is_zero():
    assert macd_relativo(pd.Series(10.0, index=DIAS)).iloc[-1] == pytest.approx(0.0)


def test_cci_of_a_constant_price_is_zero():
    preco = pd.Series(10.0, index=DIAS)
    assert cci(preco, preco, preco, 30).iloc[-1] == 0.0


def test_cci_matches_its_definition_on_a_short_window():
    fechamento = pd.Series([1.0, 2.0, 3.0], index=DIAS[:3])
    tipico = fechamento                                  # high = low = close
    desvio = np.mean(np.abs(tipico - tipico.mean()))
    assert cci(fechamento, fechamento, fechamento, 3).iloc[-1] == pytest.approx((3.0 - 2.0) / (0.015 * desvio))


def test_adx_of_a_steady_uptrend_tends_to_100():
    base = pd.Series(np.arange(1.0, 121.0), index=DIAS)
    assert adx(base + 0.5, base - 0.5, base, 30).iloc[-1] == pytest.approx(100.0, abs=0.5)
