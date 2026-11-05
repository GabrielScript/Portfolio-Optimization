"""
Option signals as a veto on the Conservadora (spec: ``docs/superpowers/specs/2026-10-29-opcoes-veto-design.md``):
the issuer's signals from the lake's chains, their direction, the partial IC given the Conservadora score and the
bottom-quintile veto. Pure functions; the scripts freeze the panel and read the test once.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd

from core.poder import Portao, ic_por_mes

# three signals read at once: t threshold Bonferroni-adjusted (spec, section 4)
PORTAO = Portao("opções: IC >= 0,02 e t >= 2,4 (Bonferroni, 3 sinais)", 0.02, 2.4)
SINAIS = {"S1": "iv_atm", "S2": "skew", "S3": "pc_ratio"}     # each ranked as minus the value: higher = better
EMISSORES_MINIMOS = 20
QUINTIL = 5


def sinais_por_emissor(cadeias: pd.DataFrame) -> pd.DataFrame:
    """Lake chains (month, underlying_isin, underlying_issuer, iv_atm, skew, pc_ratio, contracts) -> one row per
    (month, issuer): iv_atm and skew of the chain with the most contracts, pc_ratio over every chain."""
    cadeias = cadeias.assign(puts=cadeias["pc_ratio"] * cadeias["contracts"])
    principal = (cadeias.sort_values("contracts", ascending=False)
                 .drop_duplicates(["month", "underlying_issuer"]).set_index(["month", "underlying_issuer"]))
    somas = cadeias.groupby(["month", "underlying_issuer"])[["puts", "contracts"]].sum(min_count=1)
    pc = (somas["puts"] / somas["contracts"]).where(somas["contracts"] > 0)
    saida = principal[["iv_atm", "skew"]].assign(pc_ratio=pc)
    return saida.rename_axis(["month", "issuer"]).reset_index()


def pontuar(sinais: pd.DataFrame) -> pd.DataFrame:
    """S1 = -iv_atm, S2 = -skew, S3 = -pc_ratio (higher = expected better, spec section 3)."""
    return sinais[["month", "issuer"]].assign(**{nome: -sinais[coluna] for nome, coluna in SINAIS.items()})


def ic_parcial_por_mes(sinal: pd.DataFrame, retornos: pd.DataFrame, nota: pd.DataFrame) -> pd.Series:
    """Monthly partial Spearman IC of ``sinal`` with ``retornos`` given ``nota`` (months x securities), on the names
    with all three: (r_sy - r_sc r_yc) / sqrt((1 - r_sc^2)(1 - r_yc^2))."""
    validos = sinal.notna() & retornos.notna() & nota.notna()
    s, y, c = (quadro.where(validos) for quadro in (sinal, retornos, nota))
    r_sy, r_sc, r_yc = ic_por_mes(s, y), ic_por_mes(s, c), ic_por_mes(y, c)
    return (r_sy - r_sc * r_yc) / np.sqrt((1 - r_sc**2) * (1 - r_yc**2))


def vetar(candidatos: pd.Series, pontos: pd.Series, emissor: Callable[[str], str]) -> pd.Series:
    """``candidatos`` (security -> score) without every class of the bottom floor(n/5) issuers by ``pontos`` (issuer ->
    signal, higher = better), only when 20 or more issuers have the signal; names without it are kept."""
    pontos = pontos.dropna()
    if len(pontos) < EMISSORES_MINIMOS:
        return candidatos
    vetados = set(pontos.sort_values(kind="stable").index[:len(pontos) // QUINTIL])
    return candidatos[[emissor(papel) not in vetados for papel in candidatos.index]]
