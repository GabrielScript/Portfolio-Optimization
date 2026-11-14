"""
bdr_fundamentos.py - B3 BDR returns for the Composite's score on SEC fundamentals
(spec docs/superpowers/specs/2026-11-14-composto-bdr-fundamentos-design.md).

B3 data only: a session's return is its close over the BDR's previous traded close; a session the lake flags as an
unresolved opening jump (``gold/jumps.parquet``: unadjusted splits, spin-offs) returns 0; a move of -30% or worse, or
+40% or better, that the next traded session undoes (two-day product within 10% of 1) is a print error and both return 0.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

QUEDA, ALTA, DESFEITO = -0.30, 0.40, 0.10


def _do_papel(precos: pd.Series, saltos: set[pd.Timestamp]) -> pd.Series:
    negociado = precos.dropna()
    r = negociado.pct_change()
    valores = r.to_numpy(copy=True)
    for i in range(1, len(valores) - 1):
        extremo = valores[i] <= QUEDA or valores[i] >= ALTA
        if extremo and abs((1.0 + valores[i]) * (1.0 + valores[i + 1]) - 1.0) <= DESFEITO:
            valores[i] = valores[i + 1] = 0.0
    r = pd.Series(valores, index=negociado.index)
    r[r.index.isin(list(saltos))] = 0.0
    return r.reindex(precos.index)


def retornos_corrigidos(precos: pd.DataFrame, saltos: pd.DataFrame) -> pd.DataFrame:
    """Daily returns of ``precos`` (B3 calendar x BDR, NaN on a session without a trade) with the two corrections;
    ``saltos``: the lake's jump rows (security, date)."""
    por_papel = {papel: set(pd.to_datetime(grupo["date"])) for papel, grupo in saltos.groupby("security")}
    return pd.DataFrame({papel: _do_papel(precos[papel], por_papel.get(papel, set())) for papel in precos.columns},
                        index=precos.index)


def retornos_mensais(retornos: pd.DataFrame) -> pd.DataFrame:
    """Each month's compounded return (index: the month's first day); a month without any trade is empty."""
    meses = retornos.index.to_period("M").to_timestamp()
    compostos = (1.0 + retornos).groupby(meses).prod(min_count=1) - 1.0
    return compostos.where(retornos.notna().groupby(meses).any())


def juntar(tabela: pd.DataFrame, velho: str, novo: str) -> pd.Series:
    """One series for a renamed BDR: ``velho`` before ``novo``'s first value, ``novo`` from it on."""
    corte = tabela[novo].first_valid_index()
    if corte is None:
        return tabela[velho].copy()
    return pd.concat([tabela.loc[tabela.index < corte, velho], tabela.loc[tabela.index >= corte, novo]])


def indice(retornos: pd.DataFrame) -> pd.DataFrame:
    """A price index from the corrected returns, NaN before a BDR's first trade (sessions without a trade keep the
    last level, so a held name is priced every session)."""
    nivel = (1.0 + retornos.fillna(0.0)).cumprod()
    return nivel.where(retornos.notna().cummax() | retornos.notna(), np.nan)
