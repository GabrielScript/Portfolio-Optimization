"""
bdr_dividendos.py - BDR dividends taken from the US underlying (spec docs/superpowers/specs/2026-11-14-bdr-setores-design.md,
addendum 2).

B3 lists only a BDR's last few cash events, so the lake has them from 11/2025 only. Before that, a dividend is the jump
of the underlying's adjusted-over-close ratio (stockanalysis.com files of the US lake): daily where the daily file
reaches, monthly before it (dated the month's last day). The BDR gets 70% of it (30% US withholding) from its first B3
session on or after the US date; a jump above 20% is a spin-off or a return of capital and counts in full.
"""

from __future__ import annotations

import pandas as pd

RETENCAO = 0.30          # US withholding on a BDR's dividend
LIMITE = 0.20            # above it the jump is a corporate action, not a dividend

SUBJACENTE = {
    "MUTC34": "MU", "NEXT34": "NEE", "TSMC34": "TSM", "LILY34": "LLY", "GOGL34": "GOOGL", "TSLA34": "TSLA",
    "C2RW34": "CRWD", "P2LT34": "PLTR", "FCXO34": "FCX", "UNHH34": "UNH", "S1NP34": "SNPS", "I1SR34": "ISRG",
    "TMOS34": "TMO", "DEEC34": "DE", "BOTZ39": "BOTZ", "PYPL34": "PYPL", "D1EL34": "DELL", "BURA39": "URA",
    "A1MT34": "AMAT", "N1VO34": "NVO",
}


def _saltos(precos: pd.DataFrame) -> pd.Series:
    """The jumps of adjusted over close, by the file's date (``t``, ``c`` close, ``a`` adjusted). Both signs are kept:
    the adjusted series is rounded apart from the close, and that noise must cancel, not add up."""
    if precos.empty:
        return pd.Series(dtype=float, index=pd.DatetimeIndex([]))
    tabela = precos.assign(t=pd.to_datetime(precos["t"])).sort_values("t").set_index("t")
    razao = tabela["a"] / tabela["c"]
    salto = (razao / razao.shift(1) - 1.0).iloc[1:]
    return salto[salto != 0.0]


def eventos(diario: pd.DataFrame, mensal: pd.DataFrame) -> pd.Series:
    """The underlying's dividend factors by US date: the daily file's, and the monthly file's for the months before
    the daily file starts (each on its month's last day)."""
    do_dia = _saltos(diario)
    inicio = pd.to_datetime(diario["t"]).min() if not diario.empty else None
    do_mes = _saltos(mensal)
    do_mes.index = do_mes.index + pd.offsets.MonthEnd(0)
    if inicio is not None:
        do_mes = do_mes[do_mes.index < inicio.to_period("M").to_timestamp()]
    return pd.concat([do_mes, do_dia]).sort_index()


def aplicar(tr: pd.Series, eventos_eua: pd.Series, corte: pd.Timestamp | None, retencao: float = RETENCAO,
            limite: float = LIMITE) -> pd.Series:
    """``tr`` (the BDR's B3 total-return close) with each US event before ``corte`` (the BDR's first B3 cash event)
    added from the first B3 session on or after its date."""
    ajustado = tr.astype(float).copy()
    for data, salto in eventos_eua.items():
        if corte is not None and data >= corte:
            continue
        liquido = salto if salto > limite else (1.0 - retencao) * salto
        ajustado[ajustado.index >= data] *= 1.0 + liquido
    return ajustado
