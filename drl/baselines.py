"""The comparators every agent is measured against, on the same tape and with the same turnover cost: equal weights,
minimum variance and a Ledoit-Wolf mean-variance portfolio maximising the Sharpe ratio. The papers' own benchmarks
(P1's 60-session sample means + shrinkage MVO, P2's mean-variance and equal weights) map onto these three, so one table
puts the agents, the classical optimisers and the Composite side by side.

Numpy only: the projection onto the simplex keeps every solution long-only and fully invested, exactly as the papers
constrain their agents, and there is no solver to converge differently on different machines."""

from __future__ import annotations

import numpy as np
import pandas as pd

JANELA = 252          # sessions of history a baseline needs before it can hold a view
ENCOLHIMENTO = 0.1    # Ledoit-Wolf's idea, fixed: 10% of the average variance on the diagonal


def _simplex(v: np.ndarray) -> np.ndarray:
    """Euclidean projection of ``v`` onto {w >= 0, sum w = 1} (sort-based, no solver)."""
    u = np.sort(v)[::-1]
    c = np.cumsum(u) - 1.0
    rho = np.nonzero(u - c / np.arange(1, len(u) + 1) > 0)[0][-1]
    theta = c[rho] / (rho + 1.0)
    return np.maximum(v - theta, 0.0)


def _retornos(fechamentos: pd.DataFrame, ate: int, janela: int = JANELA) -> np.ndarray:
    """The window of simple returns ending at row ``ate`` (the session's own return included): n x sessions."""
    bloco = fechamentos.iloc[max(0, ate - janela + 1): ate + 1]
    return np.nan_to_num(bloco.pct_change(fill_method=None).to_numpy(dtype=float), nan=0.0)


def covariancia_encolhida(ret: np.ndarray) -> np.ndarray:
    """Ledoit-Wolf-style shrinkage: the sample covariance pulled ``ENCOLHIMENTO`` of the way to a diagonal of the same
    average variance. Keeps the matrix invertible when the window is short or a name never moved."""
    s = np.cov(ret, rowvar=False)
    s = np.atleast_2d(s)
    media = float(np.trace(s) / s.shape[0])
    return (1.0 - ENCOLHIMENTO) * s + ENCOLHIMENTO * media * np.eye(s.shape[0])


def pesos_iguais(n: int) -> np.ndarray:
    return np.full(n, 1.0 / n) if n else np.zeros(0)


def pesos_min_variancia(ret: np.ndarray, passos: int = 400) -> np.ndarray:
    """Long-only minimum variance by projected gradient descent: deterministic and convex."""
    n = ret.shape[1]
    if n == 0:
        return np.zeros(0)
    cov = covariancia_encolhida(ret)
    escala = float(np.trace(cov)) or 1.0
    w = pesos_iguais(n)
    for _ in range(passos):
        w = _simplex(w - 0.5 * (2.0 * cov @ w) / escala)
    return w


def pesos_sharpe(ret: np.ndarray, passos: int = 800, taxa: float = 0.0, passo: float = 1.0) -> np.ndarray:
    """Long-only mean-variance maximising the Sharpe ratio, by exponentiated-gradient ascent: the weights stay on the
    simplex by construction, so no step can jump a corner and park the portfolio on the wrong name."""
    n = ret.shape[1]
    if n == 0:
        return np.zeros(0)
    mu, cov = ret.mean(axis=0) - taxa, covariancia_encolhida(ret)
    w = pesos_iguais(n)
    for _ in range(passos):
        vol = float(np.sqrt(max(w @ cov @ w, 1e-18)))
        grad = mu / vol - float(w @ mu) * (cov @ w) / vol ** 3
        maior = float(np.abs(grad).max())
        if maior <= 0.0:
            break
        w = w * np.exp(passo * grad / maior)
        w = w / w.sum()
    return w


def curva(pesos: list[np.ndarray], retornos: np.ndarray, custo_bps: float = 0.0) -> np.ndarray:
    """The value of a weight path applied to ``retornos`` (t x n, the returns each choice earns), net of the turnover
    cost. Starts at 1.0."""
    valor, anterior = 1.0, np.zeros(pesos[0].shape[0]) if pesos else np.zeros(0)
    saida = [valor]
    for w, r in zip(pesos, retornos):
        giro = float(np.abs(w - anterior).sum())
        valor *= 1.0 + float(w @ r) - custo_bps / 1e4 * giro
        anterior = w
        saida.append(valor)
    return np.array(saida)
