"""The point-in-time panel the DRL work trades on: the lake's daily closes, the eligible names at each session and the
returns between sessions. Pure pandas: the environment and the agents read it, nothing here talks to the lake SQL or to
the paper accounts."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ARQUIVO = Path("lago/precos_diarios.parquet")
HISTORICO = 252          # sessions of valid closes a name needs before it may be traded


def painel(caminho: Path = ARQUIVO, inicio: str | None = None, fim: str | None = None) -> pd.DataFrame:
    """The lake's wide closes: sessions as the index, one column per name, delisted names still present. ``inicio`` and
    ``fim`` cut the sessions, never the columns (a name that dies mid-window stays, so its last close is visible)."""
    fechamentos = pd.read_parquet(caminho)
    if "date" in fechamentos.columns:          # the lake's file carries the session as a column, not the index
        fechamentos = fechamentos.set_index("date")
    fechamentos.index = pd.DatetimeIndex(pd.to_datetime(fechamentos.index)).as_unit("ns")
    fechamentos = fechamentos.sort_index().dropna(axis=1, how="all")   # over the whole file, never the window
    if inicio is not None:
        fechamentos = fechamentos.loc[inicio:]
    if fim is not None:
        fechamentos = fechamentos.loc[:fim]
    return fechamentos


def retornos(fechamentos: pd.DataFrame) -> pd.DataFrame:
    """The one-period simple returns; a session after a gap (or the name's first session) has no return and is NaN."""
    return fechamentos.pct_change(fill_method=None)


def elegiveis(fechamentos: pd.DataFrame, sessoes: int = HISTORICO) -> pd.DataFrame:
    """Whether each name may be traded at each session: it must carry ``sessoes`` valid closes up to and including that
    session, and its last close must be the previous session's (so a name that stopped trading is not offered again).
    The count uses the past only; nothing here looks at a later session."""
    validos = fechamentos.notna()
    historico = validos.rolling(sessoes, min_periods=sessoes).sum() == sessoes
    negociavel = validos & validos.shift(1).fillna(False)
    return (historico & negociavel).astype(bool)


def janela_retornos(retornos_: pd.DataFrame, ate: int, tamanho: int) -> np.ndarray:
    """The ``tamanho`` log returns ending at row ``ate`` (the session's own return is included, as the papers' lagged
    window is the data observed at the close); names with a NaN anywhere are zero-filled, never dropped, so the column
    order never changes under the agent."""
    bloco = retornos_.iloc[max(0, ate - tamanho + 1): ate + 1]
    if len(bloco) < tamanho:                  # the first sessions of the tape have no full window yet
        return np.zeros((retornos_.shape[1], tamanho))
    return np.nan_to_num(np.log1p(bloco.to_numpy(dtype=float)).T, nan=0.0, posinf=0.0, neginf=0.0)
