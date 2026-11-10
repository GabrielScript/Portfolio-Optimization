"""Tests for core.multivariate: PCA factor covariance, Fama-MacBeth and the GRS test (synthetic data only)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from core.multivariate import fama_macbeth, grs_test, newey_west_mean_se, pca_covariance, pca_factor_model


def _factor_returns(n_obs: int = 1_000, n_assets: int = 60, n_factors: int = 2, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    fatores = rng.normal(0.0, 0.012, (n_obs, n_factors))
    cargas = rng.normal(1.0, 0.5, (n_factors, n_assets)) if n_factors else np.zeros((0, n_assets))
    ruido = rng.normal(0.0, 0.010, (n_obs, n_assets))
    return pd.DataFrame(fatores @ cargas + ruido, columns=[f"A{i:02d}" for i in range(n_assets)])


# --- PCA factor covariance ---------------------------------------------------------------------------


def test_marchenko_pastur_finds_the_planted_factors() -> None:
    assert pca_factor_model(_factor_returns(n_factors=2)).n_factors == 2
    assert pca_factor_model(_factor_returns(n_factors=0)).n_factors == 0


def test_pca_covariance_keeps_the_sample_variances_and_is_psd() -> None:
    retornos = _factor_returns()
    cov = pca_covariance(retornos)
    assert list(cov.index) == list(retornos.columns) == list(cov.columns)
    np.testing.assert_allclose(np.diag(cov.to_numpy()), retornos.var(ddof=1).to_numpy(), rtol=1e-10)
    assert np.linalg.eigvalsh(cov.to_numpy()).min() > -1e-12
    np.testing.assert_allclose(cov.to_numpy(), cov.to_numpy().T)


def test_pure_noise_gives_a_diagonal_covariance() -> None:
    cov = pca_covariance(_factor_returns(n_factors=0)).to_numpy()
    assert np.allclose(cov - np.diag(np.diag(cov)), 0.0)


def test_pca_covariance_rejects_nan_and_constant_columns() -> None:
    retornos = _factor_returns(n_obs=300, n_assets=10)
    com_buraco = retornos.copy()
    com_buraco.iloc[5, 2] = np.nan
    with pytest.raises(ValueError):
        pca_covariance(com_buraco)
    constante = retornos.copy()
    constante["A03"] = 0.001
    with pytest.raises(ValueError):
        pca_covariance(constante)


# --- Newey-West and Fama-MacBeth ---------------------------------------------------------------------


def test_newey_west_with_zero_lags_is_the_plain_standard_error() -> None:
    serie = pd.Series(np.random.default_rng(1).normal(0.5, 2.0, 400))
    assert newey_west_mean_se(serie, 0) == pytest.approx(serie.std(ddof=0) / np.sqrt(len(serie)))


def test_newey_west_grows_with_positive_autocorrelation() -> None:
    rng = np.random.default_rng(2)
    choques = rng.normal(0.0, 1.0, 2_000)
    serie = pd.Series(np.convolve(choques, np.ones(3), mode="valid"))        # MA(2): overlapping sums
    assert newey_west_mean_se(serie, 3) > 1.3 * newey_west_mean_se(serie, 0)


def _painel(n_dates: int = 200, n_names: int = 80, beta: float = 0.02, seed: int = 3):
    rng = np.random.default_rng(seed)
    datas = pd.bdate_range("2012-01-02", periods=n_dates, freq="21B")
    nomes = [f"N{i:03d}" for i in range(n_names)]
    sinal = pd.DataFrame(rng.normal(size=(n_dates, n_names)), index=datas, columns=nomes)
    ruido_x = pd.DataFrame(rng.normal(size=(n_dates, n_names)), index=datas, columns=nomes)
    alvo = beta * sinal + pd.DataFrame(rng.normal(0.0, 0.10, (n_dates, n_names)), index=datas, columns=nomes)
    return alvo, sinal, ruido_x


def test_fama_macbeth_recovers_a_planted_slope_and_rejects_noise() -> None:
    alvo, sinal, ruido_x = _painel()
    resultado = fama_macbeth(alvo, {"signal": sinal, "noise": ruido_x}, lags=3)
    assert list(resultado.coefficients.columns) == ["intercept", "signal", "noise"]
    assert resultado.mean["signal"] == pytest.approx(0.02, abs=0.004)       # z-scored x has unit spread
    assert resultado.t_stat["signal"] > 5.0
    assert abs(resultado.t_stat["noise"]) < 3.0
    assert len(resultado.coefficients) == 200


def test_fama_macbeth_skips_thin_dates_and_uses_only_complete_names() -> None:
    alvo, sinal, ruido_x = _painel(n_dates=30, n_names=40)
    alvo.iloc[0, 5:] = np.nan                                                # date 0: only 5 names left
    sinal.iloc[1, :3] = np.nan                                               # date 1: three names dropped
    resultado = fama_macbeth(alvo, {"signal": sinal}, lags=1, min_names=10)
    assert alvo.index[0] not in resultado.coefficients.index
    assert resultado.n_names.loc[alvo.index[1]] == 37


# --- GRS ----------------------------------------------------------------------------------------------


def test_grs_with_one_portfolio_is_the_squared_ols_t_of_alpha() -> None:
    rng = np.random.default_rng(4)
    fator = rng.normal(0.0004, 0.01, 1_500)
    carteira = 0.0003 + 0.9 * fator + rng.normal(0.0, 0.006, 1_500)
    regressao = stats.linregress(fator, carteira)
    resultado = grs_test(pd.DataFrame({"p": carteira}), pd.DataFrame({"f": fator}))
    assert resultado.f_stat == pytest.approx((regressao.intercept / regressao.intercept_stderr) ** 2, rel=1e-8)
    assert resultado.alphas["p"] == pytest.approx(regressao.intercept, rel=1e-10)


def test_grs_detects_a_joint_alpha_and_not_its_absence() -> None:
    rng = np.random.default_rng(5)
    fator = rng.normal(0.0004, 0.01, 3_000)
    betas = np.array([0.8, 1.0, 1.2, 0.9])
    sem_alfa = fator[:, None] * betas + rng.normal(0.0, 0.006, (3_000, 4))
    com_alfa = sem_alfa + 0.0008
    f = pd.DataFrame({"f": fator})
    assert grs_test(pd.DataFrame(com_alfa), f).p_value < 0.001
    assert grs_test(pd.DataFrame(sem_alfa), f).p_value > 0.01


def test_grs_needs_more_observations_than_portfolios_plus_factors() -> None:
    with pytest.raises(ValueError):
        grs_test(pd.DataFrame(np.zeros((4, 3))), pd.DataFrame(np.ones((4, 1))))


def test_fama_macbeth_slope_is_per_cross_sectional_standard_deviation() -> None:
    """A signal in other units (x10, shifted) gives the same slope: each date's regressor is z-scored."""
    alvo, sinal, _ = _painel()
    padrao = fama_macbeth(alvo, {"signal": sinal}, lags=3).mean["signal"]
    outra_escala = fama_macbeth(alvo, {"signal": 10.0 * sinal + 4.0}, lags=3).mean["signal"]
    assert outra_escala == pytest.approx(padrao, rel=1e-6)
