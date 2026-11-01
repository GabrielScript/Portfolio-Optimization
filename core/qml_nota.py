"""
qml_nota.py - A forecast table as the monthly score of the Composite's portfolio rule, ``core.composto.estrategia_por_nota``
(docs/superpowers/specs/2026-11-01-qpo-hibrido-design.md, section 4). The models run in another environment, so the app only
ever reads their numbers.
"""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd

from core.valor import NotaDoMes


def nota_da_previsao(tabela: pd.DataFrame) -> NotaDoMes:
    """``tabela``: month, security, previsao. A month with no forecast is a KeyError: a backtest must not skip it."""
    por_mes = {mes: parte.set_index("security")["previsao"] for mes, parte in tabela.groupby("month")}

    def _nota(mes: pd.Timestamp, candidatos: Sequence[str]) -> pd.Series:
        if mes not in por_mes:
            raise KeyError(f"{mes:%Y-%m}: no forecast")
        return por_mes[mes].reindex(list(candidatos))
    return _nota
