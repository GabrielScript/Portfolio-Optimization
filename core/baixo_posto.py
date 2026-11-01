"""
baixo_posto.py - The engine's Ledoit-Wolf covariance as Sigma = a I + F'F, with F of T rows, and the optimizer on that form. With
T rows and N >> T names (the US test: 36 x 3,000) the dense conic program costs minutes per solve and this one seconds, with the
same optimum: the tests hold both to ``core.estimation.estimar_momentos`` and ``core.optimization.otimizar``.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import cvxpy as cp
import numpy as np
import pandas as pd

from core.config import DIAS_UTEIS_ANO, ConfigOtimizacao, Objetivo
from core.optimization import (ErroInviavel, ErroOtimizacao, _checar_capacidade, _checar_capacidade_grupos, _limpar_pesos,
                               _resolver, _restricoes, ajustar_aos_tetos, otimizar)


@dataclass(frozen=True)
class CovarianciaBaixoPosto:
    """Annual Sigma = diagonal x I + fator' fator; ``fator`` has one row per training row and one column per name."""

    diagonal: float
    fator: np.ndarray
    nomes: list[str]


def _encolhimento(centrados: np.ndarray) -> tuple[float, float]:
    """Ledoit-Wolf's intensity and the identity's scale, from T x T products, so nothing of size N x N is ever built. The same
    formula as ``sklearn.covariance.ledoit_wolf_shrinkage``, whose Frobenius norm of X'X equals that of XX'."""
    linhas, colunas = centrados.shape
    quadrados = centrados ** 2
    traco = quadrados.sum() / linhas
    escala = traco / colunas
    delta = float(((centrados @ centrados.T) ** 2).sum()) / linhas ** 2
    beta = (float((quadrados.sum(axis=1) ** 2).sum()) / linhas - delta) / (colunas * linhas)
    distancia = (delta - 2.0 * escala * traco + colunas * escala ** 2) / colunas
    if distancia <= 0.0:
        return 0.0, escala
    return min(beta, distancia) / distancia, escala


def ledoit_wolf_baixo_posto(retornos: pd.DataFrame, periodos: int = DIAS_UTEIS_ANO) -> CovarianciaBaixoPosto:
    """``estimar_momentos(retornos).cov`` in low-rank form, annualized by ``periodos``: the engine's 252 for daily returns, 12 for
    monthly ones."""
    if retornos.isna().any().any():
        raise ValueError("retornos contains NaN")
    valores = retornos.to_numpy(dtype=float)
    centrados = valores - valores.mean(axis=0)
    intensidade, escala = _encolhimento(centrados)
    fator = np.sqrt(periodos * (1.0 - intensidade) / len(valores)) * centrados
    return CovarianciaBaixoPosto(periodos * intensidade * escala, fator, list(retornos.columns))


def densa(sigma: CovarianciaBaixoPosto, indices: np.ndarray | None = None) -> pd.DataFrame:
    """The dense covariance of those names (all of them by default), for what needs a matrix: the risk rule, a small second stage."""
    indices = np.arange(len(sigma.nomes)) if indices is None else indices
    fator = sigma.fator[:, indices]
    nomes = [sigma.nomes[i] for i in indices]
    return pd.DataFrame(sigma.diagonal * np.eye(len(indices)) + fator.T @ fator, index=nomes, columns=nomes)


def _risco_e_desvio(w: cp.Variable, sigma: CovarianciaBaixoPosto) -> tuple[cp.Expression, cp.Expression]:
    return (sigma.diagonal * cp.sum_squares(w) + cp.sum_squares(sigma.fator @ w),
            cp.norm(cp.hstack([np.sqrt(sigma.diagonal) * w, sigma.fator @ w]), 2))


def _min_vol(sigma: CovarianciaBaixoPosto, cfg: ConfigOtimizacao) -> np.ndarray:
    w = cp.Variable(len(sigma.nomes))
    return _resolver(cp.Problem(cp.Minimize(_risco_e_desvio(w, sigma)[0]), _restricoes(w, sigma.nomes, cfg)), w)[0]


def _kelly(mu: pd.Series, sigma: CovarianciaBaixoPosto, cfg: ConfigOtimizacao) -> np.ndarray:
    """The dense ``_kelly``: with the volatility cap; if no portfolio meets it, the minimum variance."""
    w = cp.Variable(len(mu))
    risco, desvio = _risco_e_desvio(w, sigma)
    restricoes = _restricoes(w, sigma.nomes, cfg)
    if cfg.vol_maxima is not None:
        restricoes.append(desvio <= cfg.vol_maxima)
    try:
        return _resolver(cp.Problem(cp.Maximize(mu.to_numpy() @ w - (cfg.aversao_risco / 2.0) * risco), restricoes), w)[0]
    except ErroInviavel:
        if cfg.vol_maxima is None:
            raise
    return _min_vol(sigma, cfg)


def _max_sharpe(mu: pd.Series, sigma: CovarianciaBaixoPosto, cfg: ConfigOtimizacao) -> np.ndarray:
    """The dense ``_max_sharpe`` (Charnes-Cooper): the minimum variance when no feasible portfolio beats the risk-free rate or
    none with a positive Sharpe meets the volatility cap."""
    w = cp.Variable(len(mu))
    _resolver(cp.Problem(cp.Maximize(mu.to_numpy() @ w), _restricoes(w, sigma.nomes, cfg)), w)
    if float(mu.to_numpy() @ w.value) <= cfg.taxa_livre_risco + 1e-10:
        return _min_vol(sigma, cfg)
    y = cp.Variable(len(mu))
    kappa = cp.Variable(nonneg=True)
    risco, desvio = _risco_e_desvio(y, sigma)
    restricoes = [*_restricoes(y, sigma.nomes, cfg, escala=kappa), (mu.to_numpy() - cfg.taxa_livre_risco) @ y == 1]
    if cfg.vol_maxima is not None:
        restricoes.append(desvio <= cfg.vol_maxima * kappa)
    try:
        return _resolver(cp.Problem(cp.Minimize(risco), restricoes), y)[0] / float(kappa.value)
    except ErroInviavel:
        return _min_vol(sigma, cfg)


def _relaxado(mu: pd.Series, sigma: CovarianciaBaixoPosto, cfg: ConfigOtimizacao) -> np.ndarray:
    if cfg.objetivo is Objetivo.KELLY:
        return _kelly(mu, sigma, cfg)
    if cfg.objetivo is Objetivo.MAX_SHARPE:
        return _max_sharpe(mu, sigma, cfg)
    if cfg.objetivo is Objetivo.MIN_VOL:
        return _min_vol(sigma, cfg)
    raise ValueError(f"{cfg.objetivo} has no low-rank form here")


def otimizar_baixo_posto(mu: pd.Series, sigma: CovarianciaBaixoPosto, cfg: ConfigOtimizacao) -> pd.Series:
    """``otimizar(mu, cov, cfg).pesos`` for Kelly, max Sharpe and minimum variance, with its checks, its caps and its two-stage
    cardinality; the second stage holds at most ``n_ativos_max`` names, so it runs on the dense optimizer itself. Raises as
    ``otimizar`` does."""
    if list(mu.index) != sigma.nomes:
        raise ValueError("mu and the covariance need the same names in the same order")
    _checar_capacidade(len(mu), cfg)
    _checar_capacidade_grupos(sigma.nomes, cfg)
    pesos = _limpar_pesos(_relaxado(mu, sigma, cfg))
    if cfg.n_ativos_max is not None and int((pesos > 0).sum()) > cfg.n_ativos_max:
        selecionados = np.argsort(pesos)[::-1][:cfg.n_ativos_max]
        try:
            segunda = otimizar(mu.iloc[selecionados], densa(sigma, selecionados), replace(cfg, n_ativos_max=None)).pesos
        except ErroOtimizacao:      # as the dense optimizer: the cardinality stays unapplied rather than the portfolio lost
            return ajustar_aos_tetos(pd.Series(pesos, index=mu.index), cfg)
        return segunda.reindex(mu.index).fillna(0.0)
    return ajustar_aos_tetos(pd.Series(pesos, index=mu.index), cfg)
