"""
momentum_residual.py - Blitz-Huij-Martens residual momentum (spec
``docs/superpowers/specs/2026-11-16-momentum-residual-design.md``): momentum measured on the part of each name's return
the market does not explain, scaled by the residuals' own volatility. Same calendar windows as ``core.sinais``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from core.sinais import MESES_DA_VOL, MESES_DO_MOMENTUM, _meses_antes


def momentum_residual(retornos_mensais: pd.DataFrame, mes: pd.Timestamp,
                      mercado: pd.Series | None = None) -> pd.Series:
    """
    Score of ``mes`` (= m) per column of ``retornos_mensais`` (the month's candidates): OLS of the name's monthly return
    on a constant and ``mercado`` over the ``MESES_DA_VOL`` months m-36 .. m-1, then the sum of its residuals over
    m-12 .. m-2 divided by their sample standard deviation. ``mercado`` defaults to the equal-weighted mean of the
    columns with a return that month. A name missing any of the 36 months has no score (NaN), as the 36-month vol.
    """
    janela = _meses_antes(retornos_mensais, mes, MESES_DA_VOL)
    fator = (janela.mean(axis=1) if mercado is None else mercado.reindex(janela.index)).to_numpy()
    completos = janela.columns[janela.notna().all()]
    nota = pd.Series(np.nan, index=retornos_mensais.columns, dtype=float)
    if len(completos) == 0 or not np.isfinite(fator).all():
        return nota
    x = np.column_stack([np.ones(len(fator)), fator])
    y = janela[completos].to_numpy()
    residuos = y - x @ np.linalg.lstsq(x, y, rcond=None)[0]
    ultimos = residuos[-(MESES_DO_MOMENTUM + 1):-1]            # m-12 .. m-2: the last row, m-1, stays out
    nota[completos] = ultimos.sum(axis=0) / ultimos.std(axis=0, ddof=1)
    return nota
