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
