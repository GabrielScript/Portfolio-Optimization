"""Small-cap / illiquidity study (step b): bucket, picks, mid-price returns, forced exits."""

import pandas as pd
import pytest

from core.iliquidez import (apply_forced_exits, bucket, decision_dates, liquidity, mask_suspect_days, mid_returns,
                            picks, targets, traded_sessions, value_without_costs)

DECISION = pd.Timestamp("2015-01-02")
NO_UNIVERSE = pd.DataFrame({"month": pd.to_datetime([]), "security": pd.Series([], dtype=str)})


def _monthly(rows: dict[str, dict]) -> pd.DataFrame:
    """12 months (2014-01 to 2014-12) per security; ``rows`` maps security -> column overrides."""
    base = {"asset_type": "stock", "distress": "", "traded_ratio": 0.95, "median_dtv_brl": 300_000.0,
            "amihud": 1e-7, "quoted_spread": 0.01, "corwin_schultz": 0.008, "vol_daily": 0.02}
    months = pd.date_range("2014-01-01", "2014-12-01", freq="MS")
    return pd.DataFrame([{"month": month, "security": security, **base, **overrides}
                         for security, overrides in rows.items() for month in months])


def test_bucket_leaves_out_a_security_in_the_universe_of_the_decision_month():
    monthly = _monthly({"AAAA3": {}, "BBBB3": {}})
    universe = pd.DataFrame({"month": [pd.Timestamp("2015-01-01")], "security": ["BBBB3"]})
    assert bucket(monthly, universe, DECISION) == ["AAAA3"]


def test_bucket_leaves_out_a_security_traded_on_less_than_80_percent_of_sessions():
    monthly = _monthly({"AAAA3": {}, "BBBB3": {"traded_ratio": 0.7}})
    assert bucket(monthly, NO_UNIVERSE, DECISION) == ["AAAA3"]


def test_bucket_leaves_out_a_security_listed_for_only_part_of_the_12_months():
    monthly = _monthly({"AAAA3": {}, "BBBB3": {}})
    monthly = monthly[(monthly["security"] == "AAAA3") | (monthly["month"] >= pd.Timestamp("2014-07-01"))]
    assert bucket(monthly, NO_UNIVERSE, DECISION) == ["AAAA3"]


def test_bucket_leaves_out_a_security_in_distress_in_the_month_before_the_decision():
    monthly = _monthly({"AAAA3": {}, "BBBB3": {}})
    monthly.loc[(monthly["security"] == "BBBB3") & (monthly["month"] == pd.Timestamp("2014-12-01")), "distress"] = "RJ"
    assert bucket(monthly, NO_UNIVERSE, DECISION) == ["AAAA3"]


def test_bucket_keeps_the_most_traded_securities_up_to_its_size():
    monthly = _monthly({"AAAA3": {"median_dtv_brl": 100_000.0}, "BBBB3": {"median_dtv_brl": 900_000.0},
                        "CCCC3": {"median_dtv_brl": 500_000.0}})
    assert bucket(monthly, NO_UNIVERSE, DECISION, size=2) == ["BBBB3", "CCCC3"]


def test_picks_are_the_bucket_names_with_the_highest_amihud():
    monthly = _monthly({"AAAA3": {"amihud": 3e-7}, "BBBB3": {"amihud": 1e-7}, "CCCC3": {"amihud": 2e-7}})
    assert picks(monthly, ["AAAA3", "BBBB3", "CCCC3"], DECISION, n=2, capital=10_000.0) == ["AAAA3", "CCCC3"]


def test_pick_too_big_for_its_daily_volume_gives_way_to_the_next():
    # R$ 100 thousand in 2 names = R$ 50 thousand each, above 5% of AAAA3's R$ 500 thousand a day
    monthly = _monthly({"AAAA3": {"amihud": 3e-7, "median_dtv_brl": 500_000.0}, "BBBB3": {"amihud": 1e-7},
                        "CCCC3": {"amihud": 2e-7}, "DDDD3": {"amihud": 0.5e-7}})
    monthly.loc[monthly["security"] != "AAAA3", "median_dtv_brl"] = 2_000_000.0
    assert picks(monthly, ["AAAA3", "BBBB3", "CCCC3", "DDDD3"], DECISION, n=2, capital=100_000.0) == ["BBBB3", "CCCC3"]


def _daily(rows: list[tuple]) -> pd.DataFrame:
    """(security, date, close, best_bid, best_ask, factor_change, ret_total) rows."""
    frame = pd.DataFrame(rows, columns=["security", "date", "close", "best_bid", "best_ask", "factor_change",
                                        "ret_total"])
    return frame.assign(date=pd.to_datetime(frame["date"]), return_break=0)


def test_mid_return_is_the_midpoint_ratio_times_the_factor_change():
    # ex-dividend day: the midpoint falls 10.00 -> 9.80 and the factor carries the 2% paid
    daily = _daily([("AAAA3", "2015-01-05", 10.0, 9.9, 10.1, 1.0, None),
                    ("AAAA3", "2015-01-06", 9.85, 9.7, 9.9, 1.02, -0.0045)])
    assert mid_returns(daily).loc[pd.Timestamp("2015-01-06"), "AAAA3"] == pytest.approx(9.8 / 10.0 * 1.02 - 1)


def test_mid_return_falls_back_to_the_close_return_without_a_valid_quote():
    # no ask on the second day: the lake's close-to-close total return is used
    daily = _daily([("AAAA3", "2015-01-05", 10.0, 9.9, 10.1, 1.0, None),
                    ("AAAA3", "2015-01-06", 10.3, 10.2, 0.0, 1.0, 0.03)])
    assert mid_returns(daily).loc[pd.Timestamp("2015-01-06"), "AAAA3"] == pytest.approx(0.03)


def test_mid_return_is_unknown_where_the_lake_return_is_unknown():
    # a return break (or > 180 days without trading): the lake has no daily return and neither does the midpoint
    daily = _daily([("AAAA3", "2015-01-05", 10.0, 9.9, 10.1, 1.0, None),
                    ("AAAA3", "2015-01-06", 30.0, 29.9, 30.1, 1.0, None)])
    assert pd.isna(mid_returns(daily).loc[pd.Timestamp("2015-01-06"), "AAAA3"])


def test_return_on_a_suspect_day_is_masked():
    # TLCP4 2005-12-05: +89% on a rejected share event; kept by the lake, unknown to the study
    returns = pd.DataFrame({"TLCP4": [0.01, 0.89], "AAAA3": [0.02, 0.01]},
                           index=pd.to_datetime(["2005-12-02", "2005-12-05"]))
    suspect = pd.DataFrame({"security": ["TLCP4"], "date": pd.to_datetime(["2005-12-05"]), "kind": ["rejected_event"]})
    masked = mask_suspect_days(returns, suspect)
    assert (pd.isna(masked.loc["2005-12-05", "TLCP4"]), masked.loc["2005-12-05", "AAAA3"]) == (True, 0.01)


SESSIONS = pd.bdate_range("2015-01-05", periods=10)


def _held(last_trade: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """AAAA3 and BBBB3 held from the 2nd session; AAAA3 trades up to session ``last_trade`` and never again."""
    returns = pd.DataFrame(0.01, index=SESSIONS, columns=["AAAA3", "BBBB3"])
    traded = pd.DataFrame(True, index=SESSIONS, columns=["AAAA3", "BBBB3"])
    traded.loc[SESSIONS[last_trade + 1]:, "AAAA3"] = False
    returns.loc[SESSIONS[last_trade + 1]:, "AAAA3"] = float("nan")
    targets = pd.DataFrame({"AAAA3": [0.5], "BBBB3": [0.5]}, index=SESSIONS[1:2])
    return returns, traded, targets


def test_held_name_that_stops_trading_loses_the_haircut_on_the_session_after_its_last_trade():
    returns, traded, targets = _held(last_trade=4)
    exited, _ = apply_forced_exits(returns, traded, targets, distressed=set(), max_silence=3)
    assert exited.loc[SESSIONS[5], "AAAA3"] == pytest.approx(-0.5)


def test_return_of_a_name_after_its_forced_exit_is_not_credited_in_that_period():
    # back to trading 4 sessions later with +80%: the position was already sold off at the haircut
    returns, traded, targets = _held(last_trade=3)
    traded.loc[SESSIONS[8], "AAAA3"], returns.loc[SESSIONS[8], "AAAA3"] = True, 0.80
    exited, _ = apply_forced_exits(returns, traded, targets, distressed=set(), max_silence=3)
    assert exited.loc[SESSIONS[8], "AAAA3"] == 0.0


def test_name_in_distress_in_its_last_trading_month_is_written_off():
    returns, traded, targets = _held(last_trade=4)
    exited, _ = apply_forced_exits(returns, traded, targets, distressed={("AAAA3", pd.Timestamp("2015-01-01"))},
                                   max_silence=3)
    assert exited.loc[SESSIONS[5], "AAAA3"] == pytest.approx(-1.0)


def test_silence_of_max_silence_sessions_is_not_an_exit():
    returns, traded, targets = _held(last_trade=4)
    traded.loc[SESSIONS[8], "AAAA3"] = True  # 3 sessions without a trade, then back
    exited, exits = apply_forced_exits(returns, traded, targets, distressed=set(), max_silence=3)
    assert exits.empty


def test_silence_begun_before_the_period_exits_on_its_first_session():
    returns, traded, targets = _held(last_trade=2)
    targets = targets.set_axis(SESSIONS[5:6])
    exited, _ = apply_forced_exits(returns, traded, targets, distressed=set(), max_silence=3)
    assert exited.loc[SESSIONS[5], "AAAA3"] == pytest.approx(-0.5)


def test_half_spread_of_a_name_is_half_the_larger_of_quoted_and_corwin_schultz():
    monthly = _monthly({"AAAA3": {"quoted_spread": 0.010, "corwin_schultz": 0.030},
                        "BBBB3": {"quoted_spread": 0.020, "corwin_schultz": None}})
    half = liquidity(monthly).meio_spread.loc[pd.Timestamp("2014-12-01")]
    assert (half["AAAA3"], half["BBBB3"]) == (pytest.approx(0.015), pytest.approx(0.010))


def test_targets_weigh_the_picks_of_each_decision_equally():
    monthly = _monthly({"AAAA3": {"amihud": 3e-7}, "BBBB3": {"amihud": 1e-7}, "CCCC3": {"amihud": 2e-7}})
    weights = targets(monthly, NO_UNIVERSE, [DECISION], n=2, capital=10_000.0)
    assert weights.loc[DECISION].to_dict() == {"AAAA3": 0.5, "CCCC3": 0.5}


def test_decisions_start_at_the_phase_offset_and_repeat_every_given_sessions():
    assert decision_dates(SESSIONS, SESSIONS[1], every=3, offset=1) == [SESSIONS[2], SESSIONS[5], SESSIONS[8]]


def test_value_without_costs_compounds_each_name_from_its_decision():
    returns = pd.DataFrame({"AAAA3": [0.0, 0.10, 0.0], "BBBB3": [0.0, -0.10, 0.10]}, index=SESSIONS[:3])
    weights = pd.DataFrame({"AAAA3": [0.5], "BBBB3": [0.5]}, index=SESSIONS[1:2])
    assert value_without_costs(weights, returns, 1.0).tolist() == pytest.approx([1.0, 1.0, 1.045])


def test_session_without_a_row_in_a_month_the_export_kept_is_not_traded():
    daily = _daily([("AAAA3", "2015-01-05", 10.0, 9.9, 10.1, 1.0, None)])
    traded = traded_sessions(daily, NO_UNIVERSE, SESSIONS[:2])
    assert traded["AAAA3"].tolist() == [True, False]


def test_session_in_a_month_the_export_left_out_counts_as_traded():
    # AAAA3 in the universe from 2014-11 to 2015-02: the export keeps a month only for a name outside the universe in
    # some month of the 4 up to it (2015-01, not 2015-02), so a 2015-02 session without a row is not known to be silent
    daily = _daily([("AAAA3", "2015-01-30", 10.0, 9.9, 10.1, 1.0, None)])
    universe = pd.DataFrame({"month": pd.date_range("2014-11-01", "2015-02-01", freq="MS"), "security": "AAAA3"})
    traded = traded_sessions(daily, universe, pd.DatetimeIndex(["2015-01-30", "2015-02-02"]))
    assert traded["AAAA3"].tolist() == [True, True]
