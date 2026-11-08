"""
bdr_conjunto.py - One ranking for B3 stocks and US BDRs (docs/superpowers/specs/2026-11-08-bdr-conjunto-design.md):
the Stouffer score over the joint pool and the equal-weight simulator that runs it outside the engine (BDRs lack the
engine's 1,260 sessions).
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd

from core.mix_bdr import momentum_12_1
from core.views import escore_normal

SESSOES_DE_RISCO = 252


def nota_conjunta(composto: pd.Series, precos: pd.DataFrame, data: pd.Timestamp) -> pd.Series:
    """Per column of ``precos`` (the pool: B3 names with a Composite score and the eligible US names): the sum of its
    available normal scores over sqrt(their count). Price scores (momentum 12-1, minus the deviation of the last
    ``SESSOES_DE_RISCO`` daily returns up to ``data``) are taken over the whole pool; ``composto``'s over its names."""
    ate = precos.loc[:data]
    componentes = pd.DataFrame({
        "momento": escore_normal(momentum_12_1(ate, data).dropna()),
        "risco": escore_normal(-ate.pct_change(fill_method=None).iloc[-SESSOES_DE_RISCO:].std().dropna()),
        "fundamento": escore_normal(composto.dropna()),
    }).reindex(precos.columns)
    contagem = componentes.notna().sum(axis=1)
    return (componentes.sum(axis=1, min_count=1) / np.sqrt(contagem.where(contagem > 0))).dropna()


def simular(alvos: Mapping[pd.Timestamp, pd.Series], precos: pd.DataFrame, custo_bps: float) -> pd.Series:
    """Value (1 at the start) from the first date of ``alvos``: at each date the portfolio is reset to its target
    weights, paying ``custo_bps`` on the traded weight (the first date buys it all); in between each holding drifts
    with its price, a price that stops carried forward."""
    custo = custo_bps / 1e4
    datas = sorted(alvos)
    carregados = precos.ffill()
    dias = precos.index[precos.index >= datas[0]]
    valores = pd.Series(index=dias, dtype=float)
    valor, unidades = 1.0, pd.Series(dtype=float)
    for dia in dias:
        if not unidades.empty:
            valor = float((unidades * carregados.loc[dia, unidades.index]).sum())
        if dia in alvos:
            alvo = alvos[dia]
            atuais = (unidades * carregados.loc[dia, unidades.index] / valor) if not unidades.empty else pd.Series(
                dtype=float)
            negociado = float(alvo.sub(atuais, fill_value=0.0).abs().sum())
            valor *= 1.0 - custo * negociado
            unidades = valor * alvo / carregados.loc[dia, alvo.index]
        valores[dia] = valor
    return valores
