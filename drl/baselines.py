"""The classical rules the agents are measured against, as policies of the same environment: equal weights, long-only
minimum variance and long-only maximum Sharpe, the last two on a Ledoit-Wolf covariance (P1's comparator).

The first version ran these in a separate loop: rebalanced every session while the agents traded monthly, charged no
cost for bringing drifted weights back, skipped the return across each year end, and called a fixed 10% diagonal blend
"Ledoit-Wolf". As policies of ``PortfolioEnv`` they now share the agents' timing, cost, drift and eligibility.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize
from sklearn.covariance import LedoitWolf

from drl.ambiente import Policy, PortfolioEnv

WINDOW = 252                     # sessions of returns behind a covariance (P1 uses 60 for 11 assets; we hold ~150)
MIN_WINDOW = 60                  # fewer realized sessions than this: equal weights


def _with_cash(pesos: np.ndarray, tradeable: np.ndarray) -> np.ndarray:
    """Weights of the tradeable names spread back over all names, cash last (zero unless nothing is tradeable)."""
    completo = np.zeros(len(tradeable) + 1)
    if tradeable.any():
        completo[:-1][tradeable] = pesos
    else:
        completo[-1] = 1.0
    return completo


def equal_weight(env: PortfolioEnv) -> np.ndarray:
    tradeable = env.tradeable()
    return _with_cash(np.full(int(tradeable.sum()), 1.0 / max(int(tradeable.sum()), 1)), tradeable)


def ledoit_wolf(returns: np.ndarray) -> np.ndarray:
    """Ledoit and Wolf's (2004) shrinkage covariance of ``returns`` (sessions x names)."""
    return LedoitWolf().fit(returns).covariance_


def _on_simplex(objective, gradient, n: int) -> np.ndarray:
    resultado = minimize(objective, np.full(n, 1.0 / n), jac=gradient, method="SLSQP", bounds=[(0.0, 1.0)] * n,
                         constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1.0, "jac": lambda w: np.ones(n)}],
                         options={"maxiter": 500, "ftol": 1e-12})
    pesos = np.clip(resultado.x, 0.0, None)
    return pesos / pesos.sum()


def min_variance_weights(cov: np.ndarray) -> np.ndarray:
    """argmin w' S w on the simplex. The objective is scaled so the solver's tolerance means the same on any tape."""
    escala = float(np.trace(cov)) / len(cov)
    return _on_simplex(lambda w: float(w @ cov @ w) / escala, lambda w: 2.0 * (cov @ w) / escala, len(cov))


def max_sharpe_weights(mu: np.ndarray, cov: np.ndarray) -> np.ndarray:
    """argmax (w' mu) / sqrt(w' S w) on the simplex; ``mu`` is already in excess of the risk-free rate. When no name
    has a positive excess mean the ratio has no sensible maximum and the minimum-variance book is returned."""
    if not (mu > 0).any():
        return min_variance_weights(cov)

    def negativo(w: np.ndarray) -> float:
        return -float(w @ mu) / float(np.sqrt(w @ cov @ w))

    def gradiente(w: np.ndarray) -> np.ndarray:
        vol = float(np.sqrt(w @ cov @ w))
        return -(mu / vol - float(w @ mu) * (cov @ w) / vol ** 3)

    return _on_simplex(negativo, gradiente, len(cov))


def _optimized(build) -> Policy:
    def policy(env: PortfolioEnv) -> np.ndarray:
        tradeable = env.tradeable()
        retornos = env.past_returns(WINDOW)[:, tradeable]
        if retornos.shape[1] < 2 or retornos.shape[0] < MIN_WINDOW:
            return equal_weight(env)
        return _with_cash(build(retornos), tradeable)
    return policy


min_variance: Policy = _optimized(lambda retornos: min_variance_weights(ledoit_wolf(retornos)))
max_sharpe: Policy = _optimized(lambda retornos: max_sharpe_weights(retornos.mean(axis=0), ledoit_wolf(retornos)))
"""Sample means as P1's MVO uses them, with no risk-free adjustment: the comparator of the paper, not a house method."""

RULES: dict[str, Policy] = {"equal_weight": equal_weight, "min_variance": min_variance, "max_sharpe": max_sharpe}
