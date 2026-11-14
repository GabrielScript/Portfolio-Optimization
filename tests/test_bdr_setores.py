"""Fifteen BDRs, one per sector (spec 2026-11-14-bdr-setores): liquidity tiers, the growth pick, the sector order."""

from __future__ import annotations

import numpy as np
import pandas as pd

from core import bdr_setores as bs
from core.bdr_setores import Setor

SESSOES = pd.bdate_range("2024-01-01", periods=100)
INICIO, FIM = SESSOES[0], SESSOES[-1]


def _livro(**series: tuple[float, float, float]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Each name: (first close, last close, daily volume); prices move in a straight line."""
    precos = pd.DataFrame({p: np.linspace(a, b, len(SESSOES)) for p, (a, b, _) in series.items()}, index=SESSOES)
    volumes = pd.DataFrame({p: v for p, (_, _, v) in series.items()}, index=SESSOES)
    return precos, volumes


def test_the_pick_is_the_best_growth_among_the_liquid_names():
    precos, volumes = _livro(AAAA34=(10, 15, 1e5), BBBB34=(10, 20, 1e5), CCCC34=(10, 30, 2e4))
    escolha = bs.escolher(precos, volumes, (Setor("chips", "A", ("AAAA34", "BBBB34", "CCCC34")),), INICIO, FIM, quantos=1)
    assert list(escolha["papel"]) == ["BBBB34"]          # CCCC34 grew more but is thin, and a liquid name exists
    assert escolha["retorno"].iloc[0] == 1.0
    assert escolha["nivel"].iloc[0] == "liquid"


def test_a_sector_without_a_liquid_name_takes_the_best_thin_one_flagged():
    precos, volumes = _livro(AAAA34=(10, 12, 2e4), BBBB34=(10, 14, 1.5e4), CCCC34=(10, 30, 5e3))
    escolha = bs.escolher(precos, volumes, (Setor("robots", "C", ("AAAA34", "BBBB34", "CCCC34")),), INICIO, FIM, quantos=1)
    assert list(escolha["papel"]) == ["BBBB34"]          # CCCC34 is under the R$ 10 thousand floor
    assert escolha["nivel"].iloc[0] == "thin"


def test_a_name_traded_on_less_than_90_percent_of_the_sessions_is_out():
    precos, volumes = _livro(AAAA34=(10, 12, 1e5), BBBB34=(10, 20, 1e5))
    volumes.iloc[:15, 1] = np.nan                        # BBBB34 traded on 85% of the sessions
    escolha = bs.escolher(precos, volumes, (Setor("cloud", "A", ("AAAA34", "BBBB34")),), INICIO, FIM)
    assert list(escolha["papel"]) == ["AAAA34"]


def test_sectors_go_in_order_skip_the_empty_ones_and_stop_at_the_count():
    precos, volumes = _livro(AAAA34=(10, 11, 1e5), BBBB34=(10, 12, 1e5), CCCC34=(10, 13, 1e5))
    setores = (Setor("one", "A", ("AAAA34",)), Setor("empty", "A", ("ZZZZ34",)), Setor("two", "A", ("BBBB34",)),
               Setor("three", "B", ("CCCC34",)))
    escolha = bs.escolher(precos, volumes, setores, INICIO, FIM, quantos=2)
    assert list(escolha["setor"]) == ["one", "two"]
    assert list(escolha["ordem"]) == [1, 3]


def test_past_one_name_per_sector_a_second_round_takes_each_sectors_next_best_in_order():
    precos, volumes = _livro(AAAA34=(10, 11, 1e5), BBBB34=(10, 15, 1e5), CCCC34=(10, 13, 1e5), DDDD34=(10, 30, 2e4),
                             EEEE34=(10, 12, 1e5))
    setores = (Setor("one", "A", ("AAAA34", "BBBB34", "DDDD34")), Setor("two", "A", ("CCCC34", "EEEE34")))
    escolha = bs.escolher(precos, volumes, setores, INICIO, FIM, quantos=4)
    assert list(escolha["papel"]) == ["BBBB34", "CCCC34", "AAAA34", "EEEE34"]   # liquid AAAA34 before thin DDDD34
    assert list(escolha["rodada"]) == [1, 1, 2, 2]


def test_every_bdr_belongs_to_one_sector_only():
    todos = [p for setor in bs.SETORES for p in setor.candidatos]
    assert len(todos) == len(set(todos))
    assert len(bs.SETORES) == 20
