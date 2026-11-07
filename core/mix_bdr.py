"""
mix_bdr.py - The fixed BR/BDR mixes (docs/superpowers/specs/2026-11-07-mix-bdr-design.md, section 2): the US sleeve's
eligibility and picks, and the two-sleeve curve reset to (15 - k)/15 and k/15 at each rebalance.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import pandas as pd

NOMES = 15
PISO_VOLUME = 1e6
COBERTURA = 0.9
MESES = 12
LONGE, PERTO = 252, 21


def momentum_12_1(precos: pd.DataFrame, data: pd.Timestamp) -> pd.Series:
    """Total return from ``LONGE`` to ``PERTO`` sessions before ``data`` (its session counts as the last), per column."""
    ate = precos.loc[:data]
    if len(ate) <= LONGE:
        return pd.Series(float("nan"), index=precos.columns)
    return ate.iloc[-1 - PERTO] / ate.iloc[-1 - LONGE] - 1.0


def elegiveis_no_mes(volume: pd.DataFrame, mes: pd.Timestamp) -> list[str]:
    """Traded in >= ``COBERTURA`` of the previous ``MESES`` months' sessions, median daily volume >= ``PISO_VOLUME``."""
    periodo = volume.index.to_period("M")
    atual = pd.Timestamp(mes).to_period("M")
    janela = volume[(periodo >= atual - MESES) & (periodo < atual)]
    if janela.empty:
        return []
    negociou = janela.notna().mean()
    return sorted(janela.columns[(negociou >= COBERTURA) & (janela.median() >= PISO_VOLUME)])


def escolher_eua(precos: pd.DataFrame, volume: pd.DataFrame, data: pd.Timestamp, k: int) -> list[str]:
    """The ``k`` eligible US names with the highest 12-1 momentum at ``data`` (fewer if fewer qualify)."""
    elegiveis = elegiveis_no_mes(volume, data.to_period("M").to_timestamp())
    momentum = momentum_12_1(precos[elegiveis], data).dropna() if elegiveis else pd.Series(dtype=float)
    return momentum.sort_values(ascending=False, kind="mergesort").index[:k].tolist()


def curva_mista(br: pd.Series, eua: pd.DataFrame, escolhas: Mapping[pd.Timestamp, Sequence[str]], k: int,
                custo_bps: float) -> pd.Series:
    """Value from the first rebalance on: at each date of ``escolhas`` the BR sleeve (curve ``br``) gets 1 - n/15 and
    each of the n <= k US picks 1/15; both drift until the next date. The traded share pays ``custo_bps`` (the BR
    sleeve's own trades are inside its curve; the first date buys only the US picks)."""
    custo = custo_bps / 1e4
    datas = sorted(escolhas)
    precos = eua.reindex(br.index).ffill()
    dias = br.index[br.index >= datas[0]]
    valores = pd.Series(index=dias, dtype=float)
    valor, unidades_br, unidades = 1.0, 0.0, pd.Series(dtype=float)
    proximas = set(datas)
    for dia in dias:
        valor = unidades_br * br[dia] + float((unidades * precos.loc[dia, unidades.index]).sum()) if unidades_br else valor
        if dia in proximas:
            picks = list(escolhas[dia])[:k]
            alvo_eua = pd.Series(1.0 / NOMES, index=picks)
            alvo_br = 1.0 - alvo_eua.sum()
            if unidades_br:
                atual_eua = unidades * precos.loc[dia, unidades.index] / valor
                atual_br = unidades_br * br[dia] / valor
                negociado = abs(alvo_br - atual_br) + float(alvo_eua.sub(atual_eua, fill_value=0.0).abs().sum())
            else:
                negociado = float(alvo_eua.sum())
            valor *= 1.0 - custo * negociado
            unidades_br = valor * alvo_br / br[dia]
            unidades = valor * alvo_eua / precos.loc[dia, picks] if picks else pd.Series(dtype=float)
        valores[dia] = valor
    return valores
