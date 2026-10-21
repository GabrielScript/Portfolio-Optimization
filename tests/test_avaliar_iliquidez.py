"""Pieces of scripts/avaliar_iliquidez.py (step b: small caps below the universe, R$ 10 thousand, against SMAL11)."""

import pandas as pd
import pytest

from core.execucao import Refeita
from scripts.avaliar_iliquidez import close_far_outside_range, given_way, held_rows, measures, one_session_later

SESSIONS = pd.bdate_range("2015-01-05", periods=6)


def test_held_rows_are_those_of_each_name_while_its_weight_holds():
    # AAAA3 held from the 2nd session, BBBB3 from the 4th on (AAAA3 sold then)
    daily = pd.DataFrame({"security": ["AAAA3"] * 6 + ["BBBB3"] * 6, "date": list(SESSIONS) * 2})
    weights = pd.DataFrame({"AAAA3": [1.0, 0.0], "BBBB3": [0.0, 1.0]}, index=SESSIONS[[1, 3]])
    rows = held_rows(daily, weights)
    assert sorted(zip(rows["security"], rows["date"])) == [("AAAA3", SESSIONS[1]), ("AAAA3", SESSIONS[2]),
                                                           ("BBBB3", SESSIONS[3]), ("BBBB3", SESSIONS[4]),
                                                           ("BBBB3", SESSIONS[5])]


def test_close_more_than_the_tolerance_outside_the_day_range_fails_the_gate():
    # SULA3 2020-06-08 closed 1.4% below its low (known, within 2%); a close 3% above the high is a data error
    rows = pd.DataFrame({"security": ["SULA3", "AAAA3"], "date": SESSIONS[:2], "close": [20.76, 10.30],
                         "low": [21.05, 9.80], "high": [23.99, 10.00]})
    assert close_far_outside_range(rows, tolerance=0.02)["security"].tolist() == ["AAAA3"]


def test_one_session_later_moves_each_decision_to_the_next_session():
    weights = pd.DataFrame({"AAAA3": [1.0, 0.5]}, index=SESSIONS[[1, 3]])
    assert one_session_later(weights, SESSIONS).index.tolist() == [SESSIONS[2], SESSIONS[4]]


def test_given_way_are_the_picks_the_capacity_rule_passed_over():
    # 2 picks of R$ 5 thousand: AAAA3 (highest Amihud) trades R$ 50 thousand a day, 5% of it is R$ 2.5 thousand
    months = pd.date_range("2014-01-01", "2014-12-01", freq="MS")
    monthly = pd.DataFrame([{"month": month, "security": security, "distress": "", "traded_ratio": 1.0,
                             "median_dtv_brl": 50_000.0 if security == "AAAA3" else 1e6, "amihud": amihud}
                            for security, amihud in (("AAAA3", 3e-7), ("BBBB3", 2e-7), ("CCCC3", 1e-7))
                            for month in months])
    universe = pd.DataFrame({"month": pd.to_datetime([]), "security": pd.Series([], dtype=str)})
    assert given_way(monthly, universe, pd.Timestamp("2015-01-02"), n=2, capital=10_000.0) == ["AAAA3"]


def test_measures_put_the_cost_on_the_mean_equity_per_year_and_the_tax_due_at_exit():
    # a year of sessions at R$ 100 thousand, R$ 2 thousand of orders, R$ 1 thousand of tax if sold at the last close
    sessions = pd.bdate_range("2015-01-02", periods=253)
    values = pd.Series(100_000.0, index=sessions)
    exit_values = values.copy()
    exit_values.iloc[-1] = 99_000.0
    account = Refeita(values, exit_values, pd.Series(dtype=float), pd.Series([2_000.0], index=sessions[:1]))
    row = measures(account, account)
    assert (row["Cost per year"], row["Tax at exit (R$)"]) == (pytest.approx(0.02), pytest.approx(1_000.0))
