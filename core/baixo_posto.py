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


def ledoit_wolf_baixo_posto(retornos: pd.DataFrame) -> CovarianciaBaixoPosto:
    """``estimar_momentos(retornos).cov`` in low-rank form: annualized by the same 252, from the same scale of returns."""
    if retornos.isna().any().any():
        raise ValueError("retornos contains NaN")
    valores = retornos.to_numpy(dtype=float)
    centrados = valores - valores.mean(axis=0)
    intensidade, escala = _encolhimento(centrados)
    fator = np.sqrt(DIAS_UTEIS_ANO * (1.0 - intensidade) / len(valores)) * centrados
    return CovarianciaBaixoPosto(DIAS_UTEIS_ANO * intensidade * escala, fator, list(retornos.columns))


def densa(sigma: CovarianciaBaixoPosto, indices: np.ndarray | None = None) -> pd.DataFrame:
    """The dense covariance of those names (all of them by default), for what needs a matrix: the risk rule, a small second stage."""
    indices = np.arange(len(sigma.nomes)) if indices is None else indices
    fator = sigma.fator[:, indices]
    nomes = [sigma.nomes[i] for i in indices]
    return pd.DataFrame(sigma.diagonal * np.eye(len(indices)) + fator.T @ fator, index=nomes, columns=nomes)


def _risco_e_desvio(w: cp.Variable, sigma: CovarianciaBaixoPosto) -> tuple[cp.Expression, cp.Expression]:
    return (sigma.diagonal * cp.sum_squares(w) + cp.sum_squares(sigma.fator @ w),
            cp.norm(cp.hstack([np.sqrt(sigma.diagonal) * w, sigma.fator @ w]), 2))


def _kelly_relaxado(mu: pd.Series, sigma: CovarianciaBaixoPosto, cfg: ConfigOtimizacao) -> np.ndarray:
    """The dense ``_kelly``'s first stage: with the volatility cap; if no portfolio meets it, the minimum variance."""
    w = cp.Variable(len(mu))
    risco, desvio = _risco_e_desvio(w, sigma)
    restricoes = _restricoes(w, sigma.nomes, cfg)
    com_teto = restricoes if cfg.vol_maxima is None else [*restricoes, desvio <= cfg.vol_maxima]
    try:
        return _resolver(cp.Problem(cp.Maximize(mu.to_numpy() @ w - (cfg.aversao_risco / 2.0) * risco), com_teto), w)[0]
    except ErroInviavel:
        if cfg.vol_maxima is None:
            raise
    return _resolver(cp.Problem(cp.Minimize(risco), restricoes), w)[0]


def otimizar_baixo_posto(mu: pd.Series, sigma: CovarianciaBaixoPosto, cfg: ConfigOtimizacao) -> pd.Series:
    """``otimizar(mu, cov, cfg).pesos`` for Kelly, with its checks, its caps and its two-stage cardinality; the second stage holds
    at most ``n_ativos_max`` names, so it runs on the dense optimizer itself. Raises as ``otimizar`` does."""
    if cfg.objetivo is not Objetivo.KELLY:
        raise ValueError(f"only Kelly has a low-rank form here, not {cfg.objetivo}")
    if list(mu.index) != sigma.nomes:
        raise ValueError("mu and the covariance need the same names in the same order")
    _checar_capacidade(len(mu), cfg)
    _checar_capacidade_grupos(sigma.nomes, cfg)
    pesos = _limpar_pesos(_kelly_relaxado(mu, sigma, cfg))
    if cfg.n_ativos_max is not None and int((pesos > 0).sum()) > cfg.n_ativos_max:
        selecionados = np.argsort(pesos)[::-1][:cfg.n_ativos_max]
        try:
            segunda = otimizar(mu.iloc[selecionados], densa(sigma, selecionados), replace(cfg, n_ativos_max=None)).pesos
        except ErroOtimizacao:      # as the dense optimizer: the cardinality stays unapplied rather than the portfolio lost
            return ajustar_aos_tetos(pd.Series(pesos, index=mu.index), cfg)
        return segunda.reindex(mu.index).fillna(0.0)
    return ajustar_aos_tetos(pd.Series(pesos, index=mu.index), cfg)
