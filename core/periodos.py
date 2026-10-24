"""
periodos.py - The Conservadora's frozen backtest results by period (whole run, last 5 and 3 years) and by calendar
year, from the saved daily value curves of the pre-registered rerun (`scripts/curvas_conservadora.py`). Descriptive
only: the same configuration as the registered run, so N does not change (DECISOES 122).

Kept out of `core/metrics.py` and `core/lago.py`, which are in the paper guards' paths.
"""

from __future__ import annotations

import pandas as pd

from core.metrics import cagr, drawdown
from core.volatilidade import volatilidades


def retornos_por_ano(curvas: pd.DataFrame) -> pd.DataFrame:
    """Each calendar year's simple return of every column: last session of the year over the last session of the
    year before (the first year starts at the curve's first value)."""
    fechamentos = curvas.groupby(curvas.index.year).last()
    anteriores = fechamentos.shift(1)
    anteriores.iloc[0] = curvas.iloc[0]
    return fechamentos / anteriores - 1


def _medidas(curvas: pd.DataFrame) -> dict[tuple[str, str], float]:
    medidas = {}
    for nome in curvas.columns:
        valores = curvas[nome].dropna()
        medidas[(nome, "CAGR")] = cagr(valores)
        medidas[(nome, "Vol anual")] = volatilidades(valores).anual
        medidas[(nome, "MDD")] = drawdown(valores)[1]
    return medidas


def medidas_por_periodo(curvas: pd.DataFrame, anos: tuple[int, ...] = (5, 3)) -> pd.DataFrame:
    """CAGR, annualized volatility and maximum drawdown of every column over the whole curve and over the last
    ``anos`` years (sessions strictly after the last date minus n years), one row per period."""
    fim = curvas.index[-1]
    trechos = {"Período inteiro": curvas,
               **{f"Últimos {n} anos": curvas.loc[curvas.index > fim - pd.DateOffset(years=n)] for n in anos}}
    linhas = {rotulo: {("Início", ""): trecho.index[0], ("Fim", ""): trecho.index[-1], **_medidas(trecho)}
              for rotulo, trecho in trechos.items()}
    tabela = pd.DataFrame(linhas).T
    tabela.columns = pd.MultiIndex.from_tuples(tabela.columns)
    return tabela
