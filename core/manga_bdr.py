"""
manga_bdr.py - The Composite with a BDR sleeve (spec docs/superpowers/specs/2026-11-14-composto-bdr-design.md).

At each month's first session, with data through its close, the sleeve takes the ``k`` eligible BDRs of best
252-session return over volatility, equal weights, and the book goes back to (1 - w) Composite and w sleeve. Costs
on the traded value of each leg; 15% tax on each BDR sale's gain over its average cost (no exemption, losses not
carried). Pure functions: the script ``scripts/composto_bdr_avaliar.py`` reads the lake and the Composite curves.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd

SESSOES = 252
PRESENCA_MINIMA = 0.90            # share of the window's sessions with a trade (the 12/3 ghost's rule, DECISOES 299)
VOLUME_MINIMO = 1_000_000.0       # median daily volume in R$
CUSTO_BDR, CUSTO_COMPOSTO, IMPOSTO = 0.005, 0.001, 0.15


def decisoes(datas: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """The first session of each month in ``datas``."""
    serie = datas.to_series()
    return pd.DatetimeIndex(serie.groupby(datas.to_period("M")).min().to_numpy())


def elegiveis(precos: pd.DataFrame, volumes: pd.DataFrame, ate: pd.Timestamp, sessoes: int = SESSOES,
              presenca: float = PRESENCA_MINIMA, volume_minimo: float = VOLUME_MINIMO) -> list[str]:
    """BDRs traded on at least ``presenca`` of the last ``sessoes`` sessions through ``ate`` (the frames' index is the
    B3 calendar) with a median daily volume, on the sessions they traded, of at least ``volume_minimo``."""
    janela = volumes.loc[:ate].iloc[-sessoes:]
    negociou = janela.notna() & (janela > 0)
    mediana = janela.where(negociou).median()
    ok = (negociou.mean() >= presenca) & (mediana >= volume_minimo) & precos.loc[:ate].iloc[-1].notna()
    return sorted(ok.index[ok.reindex(ok.index, fill_value=False)])


def nota(precos: pd.DataFrame, ate: pd.Timestamp, sessoes: int = SESSOES) -> pd.Series:
    """The ``sessoes``-session return over the annualized volatility of the daily returns, through ``ate``."""
    janela = precos.loc[:ate].iloc[-(sessoes + 1):].ffill()
    retornos = janela.pct_change(fill_method=None).iloc[1:]
    total = janela.iloc[-1] / janela.iloc[0] - 1.0
    return total / (retornos.std() * np.sqrt(252))


def top(notas: pd.Series, candidatos: Sequence[str], k: int) -> list[str]:
    """The ``k`` best scores among ``candidatos`` (ties by ticker)."""
    validas = notas.reindex(list(candidatos)).dropna()
    ordem = sorted(validas.index, key=lambda papel: (-float(validas[papel]), papel))
    return ordem[:k]


def simular(composto: pd.Series, precos: pd.DataFrame, escolhas: Mapping[pd.Timestamp, Sequence[str]], peso: float,
            custo_bdr: float = CUSTO_BDR, custo_composto: float = CUSTO_COMPOSTO,
            imposto: float = IMPOSTO) -> tuple[pd.Series, pd.Series]:
    """The book's values (start 1) net of the BDR tax and before it: ``composto`` is the Composite's value curve,
    ``precos`` the BDRs' closes, ``escolhas`` the sleeve's names at each decision session (traded at its close)."""
    return (_livro(composto, precos, escolhas, peso, custo_bdr, custo_composto, imposto),
            _livro(composto, precos, escolhas, peso, custo_bdr, custo_composto, 0.0))


def _livro(composto: pd.Series, precos: pd.DataFrame, escolhas: Mapping[pd.Timestamp, Sequence[str]], peso: float,
           custo_bdr: float, custo_composto: float, imposto: float) -> pd.Series:
    datas = composto.index
    px = precos.reindex(precos.index.union(datas)).ffill().reindex(datas)
    comp, valores, bases, saida = 1.0, {}, {}, []
    for i, data in enumerate(datas):
        if i > 0:
            comp *= float(composto.iloc[i] / composto.iloc[i - 1])
            for papel in valores:
                valores[papel] *= float(px.at[data, papel] / px.at[datas[i - 1], papel])
        if data in escolhas:
            comp = _rebalancear(comp, valores, bases, [p for p in escolhas[data] if pd.notna(px.at[data, p])], peso,
                                custo_bdr, custo_composto, imposto)
        saida.append(comp + sum(valores.values()))
    return pd.Series(saida, index=datas)


def _rebalancear(comp: float, valores: dict[str, float], bases: dict[str, float], nomes: Sequence[str], peso: float,
                 custo_bdr: float, custo_composto: float, imposto: float) -> float:
    """Sells, then buys, each sleeve name to ``peso`` / len(``nomes``) of the book; returns the Composite leg after the
    costs and the tax (the cash of both goes through it)."""
    total = comp + sum(valores.values())
    alvo = peso * total / len(nomes) if nomes else 0.0
    custos = impostos = 0.0
    for papel in list(valores):
        meta = alvo if papel in nomes else 0.0
        if valores[papel] > meta:
            venda = valores[papel] - meta
            ganho = venda * (1.0 - bases[papel] / valores[papel])
            impostos += imposto * max(ganho, 0.0)
            custos += custo_bdr * venda
            bases[papel] *= meta / valores[papel]
            valores[papel] = meta
        if valores[papel] <= 0.0:
            del valores[papel], bases[papel]
    for papel in nomes:
        compra = alvo - valores.get(papel, 0.0)
        if compra > 0:
            custos += custo_bdr * compra
            valores[papel] = valores.get(papel, 0.0) + compra
            bases[papel] = bases.get(papel, 0.0) + compra
    novo = total - sum(valores.values())
    custos += custo_composto * abs(novo - comp)
    return novo - custos - impostos
