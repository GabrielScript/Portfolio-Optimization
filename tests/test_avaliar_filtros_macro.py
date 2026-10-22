"""Pre-registered macro filters on BOVA11 (``scripts/avaliar_filtros_macro.py``): the equity risk premium, the credit
proxy, the meta-label and the decision rule."""

import pandas as pd
import pytest

from scripts.avaliar_filtros_macro import earnings_yield


def test_the_market_earnings_yield_sums_profits_and_values_of_issuers_that_have_both():
    fundamentals = pd.DataFrame({"net_income_parent": [10.0, -2.0, 5.0]}, index=["AAAA", "BBBB", "CCCC"])
    market_caps = pd.Series({"AAAA": 100.0, "BBBB": 50.0, "DDDD": 80.0})
    # CCCC has no market value and DDDD no profit: both out; a loss counts as it is
    assert earnings_yield(fundamentals, market_caps, ["AAAA", "BBBB", "CCCC", "DDDD"]) == pytest.approx(8.0 / 150.0)


def test_the_real_rate_of_a_month_is_the_last_10_year_quote_before_it_starts():
    from scripts.avaliar_filtros_macro import real_rate_before
    curve = pd.DataFrame({"date": pd.to_datetime(["2020-01-30", "2020-01-31", "2020-01-31", "2020-02-03"]),
                          "tenor_years": [10, 5, 10, 10], "real_rate": [0.031, 0.020, 0.032, 0.040]})
    assert real_rate_before(curve, pd.Timestamp("2020-02-01")) == 0.032


def test_the_credit_proxy_at_a_close_is_ida_di_over_cdi_up_to_the_session_before():
    # the IDA-DI of day d comes out ~19h, after the close: the decision at d only has it up to d-1
    from scripts.avaliar_filtros_macro import credit_excess
    sessions = pd.bdate_range("2020-01-01", periods=4)
    ida = pd.Series([100.0, 101.0, 102.0, 110.0], index=sessions)
    cdi = pd.Series([100.0, 100.5, 101.0, 101.5], index=sessions)
    out = credit_excess(ida, cdi, sessions=2)
    assert out.iloc[:3].isna().all() and out.iloc[3] == pytest.approx(0.02 - 0.01)


def test_the_premium_vetoes_stocks_when_below_the_median_of_the_months_before_and_never_with_under_36():
    from scripts.avaliar_filtros_macro import erp_veto
    months = pd.date_range("2011-07-01", periods=38, freq="MS")
    erp = pd.Series([float(i % 2) for i in range(36)] + [0.4, 0.5], index=months)   # median of the first 36: 0.5
    assert [erp_veto(erp, months[36]), erp_veto(erp, months[37]), erp_veto(erp, months[35])] == [True, False, False]


def test_a_months_label_says_whether_bova11_beat_the_cdi_from_the_last_close_before_it_to_its_last_close():
    from scripts.avaliar_filtros_macro import monthly_labels
    sessions = pd.DatetimeIndex(["2020-01-30", "2020-01-31", "2020-02-27", "2020-02-28", "2020-03-31"])
    bova = pd.Series([99.0, 100.0, 120.0, 101.0, 90.0], index=sessions)
    cdi = pd.Series([1.000, 1.000, 1.004, 1.005, 1.010], index=sessions)
    labels = monthly_labels(bova, cdi)
    assert labels.index.equals(pd.DatetimeIndex(["2020-02-01", "2020-03-01"])) and labels.tolist() == [1, 0]


def _history(n: int, erp: list[float], labels: list[int]) -> tuple[pd.DataFrame, pd.Series]:
    months = pd.date_range("2011-07-01", periods=n, freq="MS")
    features = pd.DataFrame({"erp": erp, "credit": [0.0] * n}, index=months)
    return features, pd.Series(labels, index=months)


def test_the_meta_label_learns_only_from_months_before_the_decision():
    from scripts.avaliar_filtros_macro import meta_label_probability
    features, labels = _history(40, [0.05] * 40, [1] * 39 + [0])
    # 39 months of "stocks won" before; the decision month's own label (0) is not known yet
    assert meta_label_probability(features, labels, features.index[39], features.iloc[39]) == 1.0


def test_the_meta_label_leans_to_stocks_when_the_premium_is_high_in_a_history_where_high_premiums_won():
    from scripts.avaliar_filtros_macro import meta_label_probability
    erp = [0.02 + 0.04 * (i % 2) for i in range(40)]
    features, labels = _history(40, erp, [i % 2 for i in range(40)])
    month = features.index[39]
    high, low = features.iloc[39].copy(), features.iloc[39].copy()
    high["erp"], low["erp"] = 0.06, 0.02
    assert meta_label_probability(features, labels, month, high) > 0.5 > meta_label_probability(features, labels,
                                                                                                month, low)


def test_the_strategy_decides_at_the_last_close_it_sees_and_holds_bova11_or_the_cdi_whole():
    from scripts.avaliar_filtros_macro import signal_strategy
    seen = []
    strategy = signal_strategy(lambda close: seen.append(close) or close.day == 3)
    returns = pd.DataFrame({"BOVA11": 0.0, "CDI": 0.0}, index=pd.bdate_range("2020-02-03", periods=2))
    weights = [strategy(returns.iloc[:1]).to_dict(), strategy(returns).to_dict()]
    assert seen == list(returns.index) and weights == [{"BOVA11": 1.0, "CDI": 0.0}, {"BOVA11": 0.0, "CDI": 1.0}]


def test_the_monthly_premium_is_the_universes_earnings_yield_minus_the_real_rate_known_before_the_month():
    from core.valor import Balancos
    from scripts.avaliar_filtros_macro import monthly_erp
    fundamentals = pd.DataFrame({"issuer": ["AAAA", "AAAA", "BBBB"], "account": ["net_income_parent"] * 3,
                                 "dt_refer": pd.to_datetime(["2019-12-31", "2020-03-31", "2019-12-31"]),
                                 "value": [10.0, 30.0, 4.0],
                                 "available_from": pd.to_datetime(["2020-03-20", "2020-05-01", "2020-03-25"])})
    caps = pd.DataFrame({"issuer": ["AAAA", "BBBB"], "month": pd.to_datetime(["2020-05-01"] * 2),
                         "market_cap": [100.0, 100.0]})
    universe = pd.DataFrame({"month": pd.to_datetime(["2020-05-01"]), "security": ["AAAA3"], "issuer": ["AAAA"]})
    curve = pd.DataFrame({"date": pd.to_datetime(["2020-04-30"]), "tenor_years": [10], "real_rate": [0.03]})
    erp = monthly_erp(Balancos(fundamentals, caps, {}), universe, curve, [pd.Timestamp("2020-05-01")])
    # AAAA's 1st-quarter document arrives on the 1st of May itself: May still uses its 2019 one; BBBB is not in the
    # universe
    assert erp.loc["2020-05-01"] == pytest.approx(10.0 / 100.0 - 0.03)


def test_the_timing_control_holds_the_filters_average_stock_weight_over_its_decisions():
    from scripts.avaliar_filtros_macro import mean_stock_weight
    weights = pd.DataFrame({"BOVA11": [1.0, 0.0, 1.0, 1.0], "CDI": [0.0, 1.0, 0.0, 0.0]})
    assert mean_stock_weight(weights) == 0.75


def test_the_premium_reads_the_market_values_rechecked_for_thousandfold_slips():
    # DECISOES 114: the 2026-10-20 values kept GOLL at ~R$ 40 trillion; same fundamentals, caps re-checked
    from scripts.avaliar_filtros_macro import CVM_FOLDER
    assert CVM_FOLDER.parts[-2:] == ("dados", "2026-10-22-cvm")
