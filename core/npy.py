"""
npy.py - Net payout yield in the Conservadora (docs/superpowers/specs/2026-10-29-npy-conservadora-design.md).

Blitz and van Vliet (2018, p. 4): the dividend yield plus the net change in shares outstanding, "calculated as the
latest level divided by the 24-month average" (Boudoukh et al., 2007). Counts are the lake's ``shares_today_units``
(ON + PN, split-adjusted, row m known before m). ``core.sinais`` is guarded by paper-v2, so the score's half-ranking
is repeated here; ``tests/test_npy.py`` checks it equals ``core.sinais.conservadora`` when no issuer issues.
"""

from __future__ import annotations

from collections.abc import Iterable

import pandas as pd

from core.risco import emissor
from core.sinais import dividend_yield_12m, momentum_12_1, volatilidade_36m

MESES_DA_EMISSAO = 24      # the paper's 24-month average


def contagens_por_mes(valor_de_mercado: pd.DataFrame) -> pd.DataFrame:
    """Month (1st day) x issuer of ``shares_today_units`` from the lake's ``market_caps`` file."""
    return (valor_de_mercado.assign(month=pd.to_datetime(valor_de_mercado["month"]))
            .pivot_table(index="month", columns="issuer", values="shares_today_units", aggfunc="last")
            .sort_index())


def emissao_liquida(contagens: pd.DataFrame, mes: pd.Timestamp) -> pd.Series:
    """Issuer -> count(m) / mean(count(m-23) .. count(m)) - 1, only issuers with all 24 counts present and > 0 (the
    rest are left out; the caller treats them as 0). Rows after ``mes`` are never read."""
    janela = contagens.reindex(pd.date_range(end=mes, periods=MESES_DA_EMISSAO, freq="MS"))
    completas = janela.notna().all() & (janela > 0).all()
    janela = janela.loc[:, completas]
    return janela.iloc[-1] / janela.mean() - 1.0


def net_payout_yield(proventos: pd.DataFrame, contagens: pd.DataFrame, mes: pd.Timestamp,
                     papeis: Iterable[str]) -> pd.Series:
    """Security -> 12-month dividend yield minus its issuer's net issuance (0 without 24 counts): an issue lowers the
    yield, a buyback raises it; both classes of an issuer get the same issuance."""
    papeis = list(papeis)
    emissao = emissao_liquida(contagens, mes)
    por_papel = pd.Series([emissao.get(emissor(papel), 0.0) for papel in papeis], index=papeis, dtype=float)
    return dividend_yield_12m(proventos, mes, papeis) - por_papel


def conservadora_npy(retornos_mensais: pd.DataFrame, proventos: pd.DataFrame, contagens: pd.DataFrame,
                     mes: pd.Timestamp, candidatos: Iterable[str]) -> pd.Series:
    """``core.sinais.conservadora`` with the net payout yield in the place of the dividend yield: 1 for the low-vol
    half plus the mean of the percentile ranks of NPY and 12-1 momentum inside each half."""
    papeis = [papel for papel in candidatos if papel in retornos_mensais.columns]
    medidas = pd.DataFrame({"vol": volatilidade_36m(retornos_mensais[papeis], mes),
                            "momentum": momentum_12_1(retornos_mensais[papeis], mes)}).dropna()
    medidas["npy"] = net_payout_yield(proventos, contagens, mes, medidas.index)
    menor_vol = medidas["vol"].rank(method="first") <= len(medidas) // 2
    postos = pd.Series(index=medidas.index, dtype=float)
    for metade in (menor_vol, ~menor_vol):
        grupo = medidas[metade]
        postos[grupo.index] = (grupo["npy"].rank(pct=True) + grupo["momentum"].rank(pct=True)) / 2.0
    return postos + menor_vol.astype(float)
