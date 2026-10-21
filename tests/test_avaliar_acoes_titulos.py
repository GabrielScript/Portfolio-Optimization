"""Pre-registered stocks x bonds split (``scripts/avaliar_acoes_titulos.py``): inputs and decision rule."""

import pandas as pd
import pytest

from scripts.avaliar_acoes_titulos import anbima_levels

HEADER = ["Índice", "Data de Referência", "Número Índice", "Variação Diária (%)", "Variação no Mês (%)"]


def test_anbima_history_becomes_the_daily_index_level_whatever_the_label_spelling():
    # the label changes spelling inside one file ("IMA-B" early, "IMA - B" later) and dates come as text
    grid = pd.DataFrame([HEADER,
                         ["IMA-B", "2003-09-16 00:00:00", "723.070317", None, None],
                         ["IMA - B", "2026-10-20 00:00:00", "12562.899543", "0.3964", "4.3093"]])
    levels = anbima_levels(grid)
    assert levels.index.equals(pd.DatetimeIndex(["2003-09-16", "2026-10-20"]))
    assert levels.tolist() == pytest.approx([723.070317, 12562.899543])


def test_the_etf_stand_in_loses_its_annual_fee_spread_over_252_sessions():
    from scripts.avaliar_acoes_titulos import net_of_fee
    level = pd.Series(100.0, index=pd.bdate_range("2020-01-01", periods=253))
    net = net_of_fee(level, 0.0025)
    assert net.iloc[0] == 100.0 and net.iloc[-1] == pytest.approx(100.0 * (1 - 0.0025))


def test_a_level_on_a_day_the_b3_was_closed_reaches_the_next_session():
    # 25/01/2021: São Paulo holiday, B3 closed, the CDI still accrued
    from scripts.avaliar_acoes_titulos import cdi_index, on_sessions
    cdi = pd.Series(0.001, index=pd.DatetimeIndex(["2021-01-22", "2021-01-25", "2021-01-26"]))
    sessions = pd.DatetimeIndex(["2021-01-21", "2021-01-22", "2021-01-26"])
    level = on_sessions(cdi_index(cdi), sessions)
    assert pd.isna(level.iloc[0])
    assert level.iloc[2] / level.iloc[1] == pytest.approx(1.001 ** 2)


def test_the_bond_leg_is_ima_b_only_when_it_is_the_sole_candidate_or_beats_the_cdi_mix_in_every_phase():
    from scripts.avaliar_acoes_titulos import bond_leg
    assert [bond_leg(True, True, [0.96, 0.97, 0.95]), bond_leg(True, True, [0.96, 0.94, 0.99]),
            bond_leg(True, False, [0.10] * 3), bond_leg(False, True, [0.99] * 3),
            bond_leg(False, False, [0.99] * 3)] == ["IMA-B", "CDI", "IMA-B", "CDI", None]
