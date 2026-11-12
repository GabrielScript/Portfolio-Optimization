"""Aleph Bot, stage 0: the universe by month (docs/superpowers/specs/2026-11-12-aleph-bot-design.md)."""

from __future__ import annotations

import pandas as pd

from core.bot import universo


def test_a_name_enters_by_the_median_volume_of_the_63_sessions_before_the_month() -> None:
    dias = pd.bdate_range("2024-01-01", "2024-06-28")
    volume = pd.DataFrame({"AAAA3": 30e6, "BBBB3": 10e6, "CCCC3": 30e6, "XPTO34": 90e6, "BOVA11": 500e6, "FIIX11": 50e6},
                          index=dias)
    volume.loc[dias >= "2024-06-01", "BBBB3"] = 900e6          # June's own volume must not count for June
    casa = pd.DataFrame({"month": pd.Timestamp("2024-06-01"), "security": ["AAAA3", "BBBB3"]})   # CCCC3 not in the house
    nomes = universo.of_month(volume, casa, pd.Timestamp("2024-06-01"))
    assert nomes == ["AAAA3", "BOVA11"]                         # house names by volume, plus the listed index ETFs
    julho = universo.of_month(volume, casa.assign(month=pd.Timestamp("2024-07-01")), pd.Timestamp("2024-07-01"))
    assert "BBBB3" not in julho                                 # 20 sessions of 900 mi do not move a 63-session median


def test_too_little_history_keeps_a_name_out() -> None:
    dias = pd.bdate_range("2024-05-01", "2024-05-31")
    volume = pd.DataFrame({"AAAA3": 30e6}, index=dias)
    casa = pd.DataFrame({"month": pd.Timestamp("2024-06-01"), "security": ["AAAA3"]})
    assert universo.of_month(volume, casa, pd.Timestamp("2024-06-01")) == []
