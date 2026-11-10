"""The DRL tape: holder returns across gaps, point-in-time eligibility, causal market features."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from drl.dados import build_panel, expanding_z, market_features, unbroken_run
from tests.drl_fixtures import panel_of, random_prices, universe_of


def test_the_move_across_a_gap_lands_on_the_session_the_name_trades_again() -> None:
    datas = pd.bdate_range("2020-01-01", periods=6)
    precos = pd.DataFrame({"A": [10.0, 10.0, np.nan, np.nan, 7.0, 7.0], "B": 5.0}, index=datas)
    forward = panel_of(precos).forward[:, 0]
    assert forward == pytest.approx([0.0, 0.0, 0.0, -0.3, 0.0])


def test_a_delisted_name_stays_at_its_last_close() -> None:
    datas = pd.bdate_range("2020-01-01", periods=5)
    precos = pd.DataFrame({"A": [10.0, 11.0, np.nan, np.nan, np.nan], "B": 5.0}, index=datas)
    assert panel_of(precos).forward[:, 0] == pytest.approx([0.1, 0.0, 0.0, 0.0])


def test_a_name_is_eligible_only_while_listed_trading_and_with_an_unbroken_history() -> None:
    precos = random_prices(sessions=80, names=3)
    precos.iloc[40, 1] = np.nan                                    # N01 misses one session
    painel = build_panel(precos, universe_of(precos, ["N00", "N01"]), pd.Series(0.0, index=precos.index), history=10)
    assert not painel.eligible[:, 2].any()                         # N02 is never on the lake's list
    assert not painel.eligible[:9, 0].any() and painel.eligible[9:, 0].all()
    assert not painel.eligible[40:50, 1].any() and painel.eligible[50, 1]


def test_the_list_read_is_the_one_of_the_month_the_return_is_earned_in() -> None:
    datas = pd.DatetimeIndex(["2020-01-30", "2020-01-31", "2020-02-03", "2020-02-04"])
    precos = pd.DataFrame({"A": 1.0, "B": 1.0}, index=datas)
    universo = pd.DataFrame({"month": [pd.Timestamp("2020-01-01"), pd.Timestamp("2020-02-01")], "security": ["A", "B"]})
    painel = build_panel(precos, universo, pd.Series(0.0, index=datas), history=1)
    assert painel.eligible[:, 0].tolist() == [True, False, False, False]     # 31/01 already trades for February
    assert painel.eligible[:, 1].tolist() == [False, True, True, True]       # 04/02 earns on 05/02: still February


def test_cash_accrues_the_rate_dated_on_the_session_the_return_ends() -> None:
    datas = pd.bdate_range("2020-01-01", periods=3)
    precos = pd.DataFrame({"A": 1.0, "B": 1.0}, index=datas)
    painel = build_panel(precos, universe_of(precos), pd.Series([0.01, 0.02, 0.03], index=datas), history=1)
    assert painel.cdi == pytest.approx([0.02, 0.03])


def test_unbroken_run_restarts_after_a_hole() -> None:
    valido = np.array([[True], [True], [False], [True]])
    assert unbroken_run(valido)[:, 0].tolist() == [1, 2, 0, 1]


def test_expanding_z_uses_only_the_values_up_to_each_point() -> None:
    serie = np.array([1.0, 2.0, 3.0, 10.0])
    mexida = np.array([1.0, 2.0, 3.0, -50.0])
    assert expanding_z(serie)[:3] == pytest.approx(expanding_z(mexida)[:3])
    assert expanding_z(serie)[2] == pytest.approx((3.0 - 2.0) / np.std([1.0, 2.0, 3.0]))


def test_market_features_at_a_row_ignore_later_returns() -> None:
    painel = panel_of(random_prices(sessions=200, names=6))
    mexido = painel.forward.copy()
    mexido[120:] *= 3.0
    assert market_features(mexido, painel.eligible)[:121] == pytest.approx(painel.market[:121])
    assert not np.allclose(market_features(mexido, painel.eligible)[140:], painel.market[140:])
