"""
multivariate.py - Multivariate statistics for the research arms (G4 of the QML line; test branch only).

* ``pca_factor_model`` / ``pca_covariance``: a statistical factor model of the covariance. PCA on the
  correlation matrix; the number of factors is the number of eigenvalues above the Marchenko-Pastur edge
  (1 + sqrt(N/T))^2, so it is never chosen by hand. The idiosyncratic part is diagonal and set so every
  variance equals the sample variance (Palomar ch. 3, "noise cleaning").
* ``fama_macbeth``: one cross-sectional OLS per date, coefficients averaged over time, Newey-West errors on
  the coefficient series (the labels of a monthly-sampled 63-session return overlap).
* ``grs_test``: Gibbons, Ross and Shanken (1989): are the alphas of N portfolios against K factors jointly
  zero? It assumes iid normal errors; daily returns violate that, so the result is descriptive.

No module of the app imports this file.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

Z_CLIP = 3.0


# --- PCA factor covariance ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FactorCovariance:
    """Daily covariance of a statistical factor model and the number of factors kept."""

    cov: pd.DataFrame
    n_factors: int


def marchenko_pastur_edge(n_assets: int, n_obs: int) -> float:
    """Upper edge of the eigenvalue spectrum of a pure-noise correlation matrix (N assets, T observations)."""
    return (1.0 + np.sqrt(n_assets / n_obs)) ** 2


def pca_factor_model(returns: pd.DataFrame) -> FactorCovariance:
    """
    Correlation = U_K L_K U_K' + Psi, with K the eigenvalues above the Marchenko-Pastur edge and Psi diagonal
    so that the diagonal is exactly 1; covariance = D C D with the sample standard deviations D.
    """
    valores = returns.to_numpy(dtype=float)
    n_obs, n_assets = valores.shape
    if n_obs < 3 or n_assets < 2:
        raise ValueError(f"need at least 3 observations and 2 assets: {valores.shape}")
    if not np.isfinite(valores).all():
        raise ValueError("returns contain NaN or inf: drop the incomplete columns first")
    constantes = np.ptp(valores, axis=0) == 0                               # exact; a float std is ~1e-19 here
    if constantes.any():
        raise ValueError(f"constant return series: {list(returns.columns[constantes])}")
    desvios = valores.std(axis=0, ddof=1)
    correlacao = np.corrcoef(valores, rowvar=False)
    autovalores, autovetores = np.linalg.eigh(correlacao)                   # ascending
    ordem = np.argsort(autovalores)[::-1]
    autovalores, autovetores = autovalores[ordem], autovetores[:, ordem]
    k = int((autovalores > marchenko_pastur_edge(n_assets, n_obs)).sum())
    cargas = autovetores[:, :k] * np.sqrt(autovalores[:k])
    comum = cargas @ cargas.T
    idiossincratico = np.clip(1.0 - np.diag(comum), 0.0, None)
    limpa = comum + np.diag(idiossincratico)
    cov = limpa * np.outer(desvios, desvios)
    return FactorCovariance(pd.DataFrame(cov, index=returns.columns, columns=returns.columns), k)


def pca_covariance(returns: pd.DataFrame) -> pd.DataFrame:
    """The daily covariance of ``pca_factor_model``: a drop-in for ``covariancia_ledoit_wolf``."""
    return pca_factor_model(returns).cov


# --- Fama-MacBeth ------------------------------------------------------------------------------------


def newey_west_mean_se(series: pd.Series, lags: int) -> float:
    """Standard error of the mean of ``series`` with a Bartlett kernel of ``lags`` lags."""
    x = np.asarray(series, dtype=float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 2:
        return float("nan")
    centrado = x - x.mean()
    variancia_longa = centrado @ centrado / n
    for lag in range(1, lags + 1):
        variancia_longa += 2.0 * (1.0 - lag / (lags + 1)) * (centrado[lag:] @ centrado[:-lag]) / n
    return float(np.sqrt(max(variancia_longa, 0.0) / n))


@dataclass(frozen=True)
class FamaMacBethResult:
    """Per-date coefficients, their time means and Newey-West t statistics, and names used per date."""

    coefficients: pd.DataFrame
    mean: pd.Series
    t_stat: pd.Series
    n_names: pd.Series


def _zscore(coluna: pd.Series) -> pd.Series | None:
    desvio = coluna.std(ddof=0)
    if not np.isfinite(desvio) or desvio <= 0:
        return None
    return ((coluna - coluna.mean()) / desvio).clip(-Z_CLIP, Z_CLIP)


def fama_macbeth(target: pd.DataFrame, regressors: Mapping[str, pd.DataFrame], lags: int,
                 min_names: int = 10) -> FamaMacBethResult:
    """
    For each date of ``target`` (dates x names): OLS of the target on an intercept and the cross-sectionally
    z-scored regressors (clipped to +-3), over the names where every value exists. A date with fewer than
    ``min_names`` names, or a regressor without dispersion, is skipped.
    """
    nomes_regressores = list(regressors)
    linhas: dict[pd.Timestamp, np.ndarray] = {}
    contagem: dict[pd.Timestamp, int] = {}
    for data in target.index:
        quadro = pd.concat({"y": target.loc[data],
                            **{nome: regressors[nome].reindex(index=[data]).iloc[0] if data in regressors[nome].index
                               else pd.Series(dtype=float) for nome in nomes_regressores}}, axis=1).dropna()
        if len(quadro) < min_names:
            continue
        colunas = [_zscore(quadro[nome]) for nome in nomes_regressores]
        if any(coluna is None for coluna in colunas):
            continue
        x = np.column_stack([np.ones(len(quadro)), *colunas])
        coeficientes, *_ = np.linalg.lstsq(x, quadro["y"].to_numpy(dtype=float), rcond=None)
        linhas[data] = coeficientes
        contagem[data] = len(quadro)
    colunas_saida = ["intercept", *nomes_regressores]
    coeficientes = pd.DataFrame.from_dict(linhas, orient="index", columns=colunas_saida).sort_index()
    media = coeficientes.mean()
    erros = pd.Series({nome: newey_west_mean_se(coeficientes[nome], lags) for nome in colunas_saida})
    return FamaMacBethResult(coeficientes, media, media / erros, pd.Series(contagem).sort_index())


# --- GRS ----------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class GRSResult:
    f_stat: float
    p_value: float
    alphas: pd.Series        # per period, in the units of the inputs
    n_obs: int


def grs_test(portfolio_excess: pd.DataFrame, factor_excess: pd.DataFrame) -> GRSResult:
    """
    GRS = (T - N - K) / N * a' S^-1 a / (1 + m' W^-1 m) ~ F(N, T - N - K), with S the residual covariance and W
    the factor covariance, both maximum likelihood (divided by T), and m the factor means.
    """
    comum = portfolio_excess.index.intersection(factor_excess.index)
    r = portfolio_excess.loc[comum].to_numpy(dtype=float)
    f = factor_excess.loc[comum].to_numpy(dtype=float)
    t, n = r.shape
    k = f.shape[1]
    if t <= n + k:
        raise ValueError(f"GRS needs T > N + K: T={t}, N={n}, K={k}")
    x = np.column_stack([np.ones(t), f])
    coeficientes, *_ = np.linalg.lstsq(x, r, rcond=None)
    residuos = r - x @ coeficientes
    alfas = coeficientes[0]
    sigma = residuos.T @ residuos / t
    media_f = f.mean(axis=0)
    centrado = f - media_f
    omega = centrado.T @ centrado / t
    estatistica = ((t - n - k) / n) * float(alfas @ np.linalg.solve(sigma, alfas)) \
        / (1.0 + float(media_f @ np.linalg.solve(omega, media_f)))
    return GRSResult(float(estatistica), float(stats.f.sf(estatistica, n, t - n - k)),
                     pd.Series(alfas, index=portfolio_excess.columns), t)
