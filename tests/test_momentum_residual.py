"""Residual momentum of spec 2026-11-16-momentum-residual."""

import numpy as np
import pandas as pd
import pytest

from core.momentum_residual import momentum_residual
from core.sinais import conservadora, momentum_12_1

MES = pd.Timestamp("2015-01-01")
MESES = pd.date_range(end=MES - pd.DateOffset(months=1), periods=48, freq="MS")    # 2011-01 .. 2014-12


def _mercado(semente: int = 0) -> pd.Series:
    return pd.Series(np.random.default_rng(semente).normal(0.01, 0.05, len(MESES)), index=MESES)


def _residuo_ortogonal(mercado: pd.Series, semente: int = 1) -> pd.Series:
    """Noise orthogonal to [1, market] over the 36-month fit window, so OLS returns it as the residual exactly."""
    janela = MESES[-36:]
    e = np.random.default_rng(semente).normal(0.0, 0.03, 36)
    x = np.column_stack([np.ones(36), mercado.loc[janela].to_numpy()])
    e = e - x @ np.linalg.lstsq(x, e, rcond=None)[0]
    return pd.Series(e, index=janela).reindex(MESES).fillna(0.0)


def test_score_is_the_sum_of_the_12_1_residuals_over_their_std():
    mercado = _mercado()
    e = _residuo_ortogonal(mercado)
    retornos = pd.DataFrame({"AAAA3": 0.004 + 1.5 * mercado + e})
    nota = momentum_residual(retornos, MES, mercado=mercado)
    janela = e.loc[MES - pd.DateOffset(months=12):MES - pd.DateOffset(months=2)]
    assert len(janela) == 11
    assert nota["AAAA3"] == pytest.approx(janela.sum() / janela.std(ddof=1), rel=1e-9)


def test_beta_does_not_change_the_score():
    mercado = _mercado()
    e = _residuo_ortogonal(mercado)
    retornos = pd.DataFrame({"BAIXO3": 0.5 * mercado + e, "ALTO3": 2.5 * mercado + e})
    nota = momentum_residual(retornos, MES, mercado=mercado)
    assert nota["BAIXO3"] == pytest.approx(nota["ALTO3"], rel=1e-9)


def test_a_missing_month_in_the_36_leaves_no_score():
    mercado = _mercado()
    retornos = pd.DataFrame({"AAAA3": mercado * 1.2 + 0.01, "FURO3": mercado * 0.8})
    retornos.loc[MES - pd.DateOffset(months=30), "FURO3"] = np.nan
    nota = momentum_residual(retornos, MES, mercado=mercado)
    assert np.isfinite(nota["AAAA3"]) and np.isnan(nota["FURO3"])


def test_the_decision_month_and_later_never_reach_the_score():
    gerador = np.random.default_rng(3)
    meses = pd.date_range(MESES[0], periods=len(MESES) + 3, freq="MS")
    retornos = pd.DataFrame(gerador.normal(0.01, 0.08, (len(meses), 6)), index=meses,
                            columns=[f"P{i}AA3" for i in range(6)])
    mexido = retornos.copy()
    mexido.loc[MES:] = 9.0
    pd.testing.assert_series_equal(momentum_residual(retornos, MES), momentum_residual(mexido, MES))


def test_default_market_is_the_equal_weighted_mean_of_the_candidates():
    gerador = np.random.default_rng(4)
    retornos = pd.DataFrame(gerador.normal(0.01, 0.08, (len(MESES), 5)), index=MESES,
                            columns=[f"P{i}AA3" for i in range(5)])
    retornos.iloc[5, 2] = np.nan                       # a month a name did not trade: the mean of the others
    pd.testing.assert_series_equal(momentum_residual(retornos, MES),
                                   momentum_residual(retornos, MES, mercado=retornos.mean(axis=1)))


def test_conservadora_default_momentum_is_still_12_1():
    gerador = np.random.default_rng(5)
    retornos = pd.DataFrame(gerador.normal(0.01, 0.08, (len(MESES), 8)), index=MESES,
                            columns=[f"P{i}AA3" for i in range(8)])
    proventos = pd.DataFrame({"security": ["P1AA3"], "ex_date": [MES - pd.DateOffset(months=3)], "yield": [0.05]})
    padrao = conservadora(retornos, proventos, MES, retornos.columns)
    explicito = conservadora(retornos, proventos, MES, retornos.columns, momentum=momentum_12_1)
    invertido = conservadora(retornos, proventos, MES, retornos.columns,
                             momentum=lambda r, m: -momentum_12_1(r, m))       # the hook is really used
    pd.testing.assert_series_equal(padrao, explicito)
    assert not padrao.equals(invertido)
    assert conservadora(retornos, proventos, MES, retornos.columns, momentum=momentum_residual).notna().all()
