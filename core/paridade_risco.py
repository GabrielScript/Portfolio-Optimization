"""
paridade_risco.py - Risk parity as equal risk contribution (ERC, Maillard, Roncalli and Teiletche 2010): long only, fully
invested, each asset's share of the portfolio variance w_i (Sigma w)_i / w'Sigma w equal to 1/N. Solved by cyclical
coordinate descent on the convex problem min 1/2 x'Sigma x - (1/N) sum log x_i (Griveau-Billion, Richard and Roncalli
2013), then normalized: the solution is unique for a positive definite Sigma. The strategies comparison of 2026-11-05
uses it; the HRP of ``core.hierarchical`` is a different, tree-based allocation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TOLERANCIA = 1e-10
MAX_VOLTAS = 10_000


def otimizar_paridade_de_risco(cov: pd.DataFrame) -> pd.Series:
    """ERC weights (sum 1, all positive) of the covariance ``cov`` (assets x assets, positive definite)."""
    sigma = cov.to_numpy(dtype=float)
    n = len(sigma)
    orcamento = 1.0 / n
    x = 1.0 / np.sqrt(np.diag(sigma))              # inverse volatility: exact without correlation, a close start
    x = x / x.sum()
    for _ in range(MAX_VOLTAS):
        anterior = x.copy()
        for i in range(n):
            resto = sigma[i] @ x - sigma[i, i] * x[i]
            x[i] = (-resto + np.sqrt(resto * resto + 4.0 * sigma[i, i] * orcamento)) / (2.0 * sigma[i, i])
        if np.max(np.abs(x - anterior)) < TOLERANCIA * np.max(np.abs(x)):
            break
    return pd.Series(x / x.sum(), index=cov.index)


def contribuicoes_de_risco(pesos: pd.Series, cov: pd.DataFrame) -> pd.Series:
    """Each asset's share of the portfolio variance (they sum to 1)."""
    w = pesos.reindex(cov.index).to_numpy(dtype=float)
    marginais = cov.to_numpy(dtype=float) @ w
    return pd.Series(w * marginais / (w @ marginais), index=cov.index)
