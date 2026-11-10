"""The metrics every curve is judged by, identical for the agents, the classical baselines and the Composite: annual
return, annual volatility, Sharpe, maximum drawdown, Calmar, annual turnover. Same definitions as the papers (252
sessions a year, risk-free 0 by default) so their tables and ours can be read side by side."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

SESSOES = 252


def returns(valores: np.ndarray | pd.Series) -> np.ndarray:
    """The session returns of a value curve."""
    v = np.asarray(valores, dtype=float)
    return v[1:] / v[:-1] - 1.0 if len(v) > 1 else np.zeros(0)


def metricas(valores: np.ndarray | pd.Series, giros: np.ndarray | pd.Series | None = None,
             taxa: float = 0.0) -> dict[str, float]:
    """The table line of one curve. ``taxa`` is the risk-free return per session."""
    r = returns(valores)
    if len(r) == 0:
        return {k: 0.0 for k in ("retorno_anual", "vol_anual", "sharpe", "sortino", "mdd", "calmar", "giro_anual",
                                 "total")}
    v = np.asarray(valores, dtype=float)
    anos = len(r) / SESSOES
    total = float(v[-1] / v[0] - 1.0)
    anual = float((1.0 + total) ** (1.0 / anos) - 1.0) if anos > 0 and total > -1.0 else -1.0
    vol = float(np.std(r, ddof=1) * math.sqrt(SESSOES)) if len(r) > 1 else 0.0
    excedente = float(np.mean(r) - taxa)
    sharpe = float(excedente / np.std(r, ddof=1) * math.sqrt(SESSOES)) if len(r) > 1 and np.std(r) > 0 else 0.0
    queda = np.minimum(r, 0.0)
    sortino = (float(excedente / np.sqrt(np.mean(queda ** 2)) * math.sqrt(SESSOES))
               if len(r) > 1 and np.any(queda < 0) else 0.0)
    pico = np.maximum.accumulate(v)
    mdd = float((v / pico - 1.0).min())
    calmar = float(anual / abs(mdd)) if mdd < 0 else 0.0
    giro = float(np.mean(np.asarray(giros, dtype=float)) * SESSOES) if giros is not None and len(giros) else 0.0
    return {"retorno_anual": anual, "vol_anual": vol, "sharpe": sharpe, "sortino": sortino, "mdd": mdd,
            "calmar": calmar, "giro_anual": giro, "total": total}
