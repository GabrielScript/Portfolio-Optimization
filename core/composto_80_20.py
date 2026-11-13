"""
composto_80_20.py - The Composite 80/20 (the user, 2026-11-13; docs/superpowers/specs/2026-11-13-composto-80-20-design.md):
80% the Composite's record in force (``paper-v3/composto.csv``, its weights scaled by 0.8) and 20% a Bitcoin ETF,
back to 80/20 at the first session of each month and at each new Composite record. A ghost account of the simulator
only: the user kept the pure Composite official on MT5 and in the app (2026-11-13 ~16:00). Every Bitcoin mix failed
its gate.

The ghost's record (``simulador/composto_80_20.csv``) has the Top-5's columns (``core.top5``): ``data`` is the day it
trades at (the simulator takes the first session on or after it), ``base`` the Composite record it scales.
"""

from __future__ import annotations

import pandas as pd

CARTEIRA = "Composto 80/20"
ETF = "BITH11"                            # Hashdex Bitcoin: the most liquid Bitcoin-only ETF on B3 (spec)
PESO_ETF = 0.20
INICIO = pd.Timestamp("2026-11-13")       # registration day: its close is the first trade
COLUNAS = ["data", "ticker", "peso", "base"]


def vazio() -> pd.DataFrame:
    return pd.DataFrame({"data": pd.Series(dtype="datetime64[ns]"), "ticker": pd.Series(dtype=object),
                         "peso": pd.Series(dtype=float), "base": pd.Series(dtype="datetime64[ns]")})


def pesos_mistos(pesos_do_composto: pd.Series, peso_etf: float = PESO_ETF) -> pd.Series:
    """The Composite's weights scaled to ``1 - peso_etf`` plus the ETF at ``peso_etf``."""
    if ETF in pesos_do_composto.index:
        raise ValueError(f"the Composite's record already holds {ETF}")
    acoes = pesos_do_composto[pesos_do_composto > 0].astype(float)
    return pd.concat([acoes / acoes.sum() * (1.0 - peso_etf), pd.Series({ETF: peso_etf})])


def _base_em(composto: pd.DataFrame, dia: pd.Timestamp) -> pd.Timestamp:
    datas = pd.to_datetime(composto["data"])
    anteriores = datas[datas <= dia]
    if anteriores.empty:
        raise ValueError(f"no Composite record on or before {dia:%Y-%m-%d}")
    return anteriores.max()


def _pesos_em(composto: pd.DataFrame, base: pd.Timestamp) -> pd.Series:
    linhas = composto[pd.to_datetime(composto["data"]) == base]
    return pesos_mistos(linhas.set_index(linhas["ticker"].str.removesuffix(".SA"))["peso"])


def livro_do_dia(composto: pd.DataFrame, hoje: pd.Timestamp) -> tuple[str, pd.Series]:
    """The MT5 book's record key (``<Composite date>|<YYYY-MM>``: a new month or a new Composite record is a new
    record) and its 80/20 weights on ``hoje``."""
    base = _base_em(composto, hoje)
    return f"{base:%Y-%m-%d}|{hoje:%Y-%m}", _pesos_em(composto, base)


def _datas_devidas(bases: pd.DatetimeIndex, hoje: pd.Timestamp, inicio: pd.Timestamp) -> list[pd.Timestamp]:
    if hoje < inicio:
        return []
    meses = pd.date_range(inicio + pd.offsets.MonthBegin(1), hoje, freq="MS")
    novas = bases[(bases > inicio) & (bases <= hoje)]
    return sorted({inicio, *meses, *novas})


def registros_devidos(composto: pd.DataFrame, feito: pd.DataFrame, hoje: pd.Timestamp,
                      inicio: pd.Timestamp = INICIO) -> pd.DataFrame:
    """The record rows to append on ``hoje``: the registration day, each month's first day and each Composite record
    date after it, up to ``hoje``, not yet in ``feito["data"]``; each the 80/20 mix of the Composite record in force."""
    bases = pd.DatetimeIndex(pd.to_datetime(composto["data"]).unique())
    feitas = set(pd.to_datetime(feito["data"]))
    linhas = []
    for data in _datas_devidas(bases, hoje, inicio):
        if data in feitas:
            continue
        base = _base_em(composto, data)
        pesos = _pesos_em(composto, base)
        linhas.append(pd.DataFrame({"data": data, "ticker": pesos.index, "peso": pesos.to_numpy(), "base": base}))
    return pd.concat(linhas, ignore_index=True)[COLUNAS] if linhas else vazio()
