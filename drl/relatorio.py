"""The metrics every curve is judged by, in the house convention: the Sharpe and Sortino ratios are over the CDI of
each session, and the drawdown is measured on the daily curve.

The first version used a zero risk-free rate, which on B3 adds about half a point of Sharpe to anything that merely
earns the CDI, and its summary table put a drawdown measured on yearly points next to daily ones.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np

SESSIONS = 252
FIELDS = ("cagr", "vol", "sharpe", "sortino", "mdd", "calmar", "turnover", "total")
CAPM_FIELDS = ("alpha", "beta", "t_alpha")


def session_returns(values: Sequence[float] | np.ndarray) -> np.ndarray:
    v = np.asarray(values, dtype=float)
    return v[1:] / v[:-1] - 1.0 if len(v) > 1 else np.zeros(0)


def metrics(values: Sequence[float] | np.ndarray, risk_free: Sequence[float] | np.ndarray | None = None,
            turnover: Sequence[float] | np.ndarray | None = None) -> dict[str, float]:
    """
    One table line. ``values``: one point per session. ``risk_free``: the rate of each session return (same length as
    the returns; None = zero, the papers' convention). ``turnover``: the notional traded at each decision; reported as
    a yearly sum.
    """
    v = np.asarray(values, dtype=float)
    r = session_returns(v)
    if len(r) < 2:
        raise ValueError(f"a curve needs at least three points: {len(v)}")
    rf = np.zeros(len(r)) if risk_free is None else np.asarray(risk_free, dtype=float)
    if rf.shape != r.shape:
        raise ValueError(f"risk_free has {rf.shape[0]} rates for {len(r)} returns")
    anos = len(r) / SESSIONS
    total = float(v[-1] / v[0] - 1.0)
    cagr = float((1.0 + total) ** (1.0 / anos) - 1.0) if total > -1.0 else -1.0
    excesso = r - rf
    desvio = float(np.std(excesso, ddof=1))
    queda = float(np.sqrt(np.mean(np.minimum(excesso, 0.0) ** 2)))
    mdd = float((v / np.maximum.accumulate(v) - 1.0).min())
    return {"cagr": cagr,
            "vol": float(np.std(r, ddof=1) * math.sqrt(SESSIONS)),
            "sharpe": float(np.mean(excesso) / desvio * math.sqrt(SESSIONS)) if desvio > 0 else 0.0,
            "sortino": float(np.mean(excesso) / queda * math.sqrt(SESSIONS)) if queda > 0 else 0.0,
            "mdd": mdd,
            "calmar": float(cagr / abs(mdd)) if mdd < 0 else 0.0,
            "turnover": float(np.sum(turnover) / anos) if turnover is not None else 0.0,
            "total": total}


def capm(values: Sequence[float] | np.ndarray, benchmark: Sequence[float] | np.ndarray,
         risk_free: Sequence[float] | np.ndarray) -> dict[str, float]:
    """
    The house's CAPM line (``core.metrics.comparar_benchmark``): least squares of the curve's session return over the
    risk-free rate on the benchmark's. ``alpha`` is a year's (252 x the intercept); ``t_alpha`` divides the intercept
    by its Newey-West (1987) error, Bartlett kernel with ``floor(4 (T / 100) ** (2 / 9))`` lags, because the residual
    of a daily portfolio return is autocorrelated and the plain error understates it.
    """
    r, b, rf = session_returns(values), session_returns(benchmark), np.asarray(risk_free, dtype=float)
    if not r.shape == b.shape == rf.shape or len(r) < 3:
        raise ValueError(f"the curve, the benchmark and the rate must cover the same sessions (at least three "
                         f"returns): {len(r)}, {len(b)}, {len(rf)}")
    regressores = np.column_stack([np.ones(len(r)), b - rf])
    (intercepto, beta), *_ = np.linalg.lstsq(regressores, r - rf, rcond=None)
    momentos = regressores * (r - rf - regressores @ np.array([intercepto, beta]))[:, None]
    meio = momentos.T @ momentos
    defasagens = int(np.floor(4 * (len(r) / 100) ** (2 / 9)))
    for defasagem in range(1, defasagens + 1):
        cruzado = momentos[defasagem:].T @ momentos[:-defasagem]
        meio += (1 - defasagem / (defasagens + 1)) * (cruzado + cruzado.T)
    inversa = np.linalg.inv(regressores.T @ regressores)
    erro = float(np.sqrt(max((inversa @ meio @ inversa)[0, 0], 0.0)))
    return {"alpha": float(intercepto * SESSIONS), "beta": float(beta),
            "t_alpha": float(intercepto / erro) if erro > 0 else float("nan")}
