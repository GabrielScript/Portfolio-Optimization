"""
rotacao.py - H2, the BR/US/gold sleeve rotation with a CDI filter (docs/superpowers/specs/2026-10-29-h2-rotacao-design.md).

Antonacci's dual momentum on sleeves in reais: on the last B3 session of each month, the sleeve with the highest
252-session return, if it beats the CDI's over the same sessions, else the CDI; held from the next session. 100% in one
sleeve, no leverage; ``custo`` per side of each switch (the CDI is cash: no side).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

JANELA = 252           # 12 months of B3 sessions
CAIXA = "CDI"
CUSTO = 0.001          # 10 bps per side, as CFG_BT


def sleeve_em_reais(fechamentos: pd.Series, ptax: pd.Series, sessoes: pd.DatetimeIndex, taxa_anual: float) -> pd.Series:
    """A US sleeve on B3 sessions: the close of the latest US session strictly before each B3 session, times that
    day's PTAX (the latest published on or before it), less the fund fee at 1/252 of ``taxa_anual`` per session."""
    fechamentos = fechamentos.sort_index()
    posicao = fechamentos.index.searchsorted(sessoes, side="left") - 1
    em_dolar = np.where(posicao >= 0, fechamentos.to_numpy()[np.clip(posicao, 0, None)], np.nan)
    cambio = ptax.sort_index().reindex(ptax.index.union(sessoes)).ffill().reindex(sessoes).to_numpy()
    taxa = (1.0 - taxa_anual) ** (np.arange(len(sessoes)) / JANELA)
    return pd.Series(em_dolar * cambio * taxa, index=sessoes)


def escolha(valores: pd.DataFrame, cdi_diario: pd.Series, data: pd.Timestamp) -> str | None:
    """The sleeve to hold after ``data``: the best 252-session return if above the CDI's, else ``CAIXA``; None
    without 252 sessions of every sleeve."""
    fim = valores.index.get_loc(data)
    if fim < JANELA:
        return None
    janela = valores.iloc[fim - JANELA:fim + 1]
    if janela.isna().any().any():
        return None
    retornos = janela.iloc[-1] / janela.iloc[0] - 1.0
    cdi = float(np.prod(1.0 + cdi_diario.reindex(janela.index[1:]).fillna(0.0).to_numpy()) - 1.0)
    melhor = str(retornos.idxmax())
    return melhor if retornos[melhor] > cdi else CAIXA


def rotacao(sleeves: pd.DataFrame, cdi_diario: pd.Series, custo: float = CUSTO) -> tuple[pd.Series, pd.Series]:
    """(value from 1.0 at the first decision, the choice of each month-end decision). The value of a session is after
    its return and before a switch made at its close; a switch costs ``custo`` on each risky side."""
    datas = sleeves.index
    fins_de_mes = set(pd.Series(datas, index=datas).groupby(datas.to_period("M")).max())
    cdi = cdi_diario.reindex(datas).fillna(0.0)
    valor, posse, valores, escolhas = None, None, {}, {}
    for anterior, data in zip([None, *datas[:-1]], datas):
        if posse is not None:
            fator = 1.0 + cdi[data] if posse == CAIXA else sleeves.at[data, posse] / sleeves.at[anterior, posse]
            valor *= fator
            valores[data] = valor
        if data not in fins_de_mes:
            continue
        nova = escolha(sleeves, cdi, data)
        if nova is None:
            continue
        if posse is None:
            valor = 1.0
            valores[data] = valor
            lados = int(nova != CAIXA)
        else:
            lados = int(posse != CAIXA) + int(nova != CAIXA) if nova != posse else 0
        valor *= (1.0 - custo) ** lados
        posse = nova
        escolhas[data] = nova
    return pd.Series(valores, dtype=float), pd.Series(escolhas, dtype=object)
