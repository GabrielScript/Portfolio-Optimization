"""
universo.py - Aleph Bot's universe by month (docs/superpowers/specs/2026-11-12-aleph-bot-design.md): the house's
stocks and units whose median daily volume over the 63 sessions before the month is R$ 20 million or more, plus the
index ETFs. Nothing of the month itself is read.
"""

from __future__ import annotations

import pandas as pd

MIN_VOLUME = 20e6
SESSIONS = 63
ETFS = ("BOVA11", "SMAL11", "IVVB11")


def of_month(volume: pd.DataFrame, house: pd.DataFrame, month: pd.Timestamp) -> list[str]:
    """``volume``: daily BRL volume, dates x securities. ``house``: the lake's ``universe_monthly`` (month,
    security). A name needs all ``SESSIONS`` sessions before ``month`` to be judged."""
    antes = volume.loc[volume.index < month].tail(SESSIONS)
    if len(antes) < SESSIONS:
        return []
    da_casa = set(house.loc[pd.to_datetime(house["month"]) == month, "security"])
    mediana = antes.median(skipna=False)
    candidatos = [nome for nome in mediana.index if nome in da_casa or nome in ETFS]
    return sorted(nome for nome in candidatos if mediana[nome] >= MIN_VOLUME)
