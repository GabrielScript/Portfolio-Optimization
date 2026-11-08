"""
pead.py - SUE, signal 1 of the pre-registered test (docs/superpowers/specs/2026-11-08-pead-recompra-design.md,
section 3): this quarter's net income minus the same quarter's a year before, over the deviation of the 8 such changes
before it (Bernard and Thomas's seasonal random walk), point in time from the CVM statements. The statements hold
12-month flows, so the change of the 12-month figure between consecutive quarter ends is exactly that surprise.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from core.risco import emissor
from core.valor import NotaDoMes

TRIMESTRES_ANTES = 8        # the changes whose deviation scales the last one
MINIMO_ANTES = 6            # of them present


def variacoes_trimestrais(fundamentos: pd.DataFrame, mes: pd.Timestamp) -> pd.DataFrame:
    """Issuer x quarter end: NI(q) - NI(q - 1 quarter) of the 12-month net income received before ``mes``; empty where
    either quarter end is missing."""
    conhecidos = fundamentos[(fundamentos["account"] == "net_income") & (fundamentos["available_from"] < mes)]
    if conhecidos.empty:
        return pd.DataFrame()
    lucros = conhecidos.pivot_table(index="issuer", columns="dt_refer", values="value", aggfunc="last")
    lucros.columns = pd.DatetimeIndex(lucros.columns)
    trimestres = pd.date_range(lucros.columns.min(), lucros.columns.max(), freq="QE")
    return lucros.reindex(columns=trimestres).diff(axis=1)


def sue(fundamentos: pd.DataFrame, mes: pd.Timestamp) -> pd.Series:
    """Issuer -> SUE at the latest quarter end with a change: that change over the sample deviation of the
    ``TRIMESTRES_ANTES`` quarter ends before it, with at least ``MINIMO_ANTES`` present and a deviation > 0."""
    variacoes = variacoes_trimestrais(fundamentos, mes)
    resultado = {}
    for emissor_, linha in zip(variacoes.index, variacoes.to_numpy()):
        validas = np.flatnonzero(~np.isnan(linha))
        if validas.size == 0:
            continue
        ultima = validas[-1]
        antes = linha[max(ultima - TRIMESTRES_ANTES, 0):ultima]
        antes = antes[~np.isnan(antes)]
        if antes.size < MINIMO_ANTES:
            continue
        desvio = float(np.std(antes, ddof=1))
        if desvio > 0.0:
            resultado[emissor_] = float(linha[ultima]) / desvio
    return pd.Series(resultado, dtype=float)


def nota_sue(fundamentos: pd.DataFrame) -> NotaDoMes:
    """``sue`` of each ticker among the month's candidates (an issuer's classes share it)."""
    def _nota(mes: pd.Timestamp, candidatos: Sequence[str]) -> pd.Series:
        valores = sue(fundamentos, mes)
        papeis = [papel for papel in candidatos if emissor(papel) in valores.index]
        return pd.Series([valores[emissor(papel)] for papel in papeis], index=papeis, dtype=float)
    return _nota
