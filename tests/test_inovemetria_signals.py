"""The five Inovemetria families on hand-made frames: each uses only what was public before the month."""

import numpy as np
import pandas as pd
import pytest

from core.inovemetria.data import InnovationData
from core.inovemetria.diffusion import step_logistic
from core.inovemetria.signals import (convexity_score, diffusion_score, effort_score, encroachment_score,
                                      field_counts, patent_components, patent_score)

T = pd.Timestamp


def _rows(issuer: str, account: str, ends, values, lag_days: int = 40) -> list[dict]:
    return [{"issuer": issuer, "account": account, "dt_refer": T(e), "value": float(v),
             "available_from": T(e) + pd.Timedelta(days=lag_days)} for e, v in zip(ends, values)]


def _data(rows: list[dict], patents=None, activity=None, roots=None) -> InnovationData:
    return InnovationData(
        fundamentals=pd.DataFrame(rows),
        patents=pd.DataFrame(patents or [], columns=["application", "cnpj_root", "filed_on", "known_from", "fields"]),
        subsidiaries=pd.DataFrame(columns=["issuer", "cnpj_root", "dt_refer", "known_from"]),
        roots=pd.Series(roots or {}, dtype=object),
        activity=pd.Series(activity or {}, dtype=object),
        ipca=pd.Series(1.0, index=pd.date_range("2000-01-01", "2027-01-01", freq="MS")))


QUARTERS = pd.date_range("2016-03-31", periods=16, freq="QE")


def _logistic(n: int, x0: float = 20.0) -> list[float]:
    xs = [x0]
    for _ in range(n - 1):
        xs.append(float(step_logistic(xs[-1], 0.35, 1000.0)))
    return xs


# S. Diffusion stage

def test_diffusion_score_is_the_forecast_log_growth_and_ignores_later_filings():
    xs = _logistic(16)
    rows = _rows("ACME", "revenue", QUARTERS, xs)
    rows += _rows("ACME", "revenue", [T("2020-03-31")], [5.0], lag_days=400)       # known only much later
    month = T("2020-03-01")
    score = diffusion_score(_data(rows))(month, ["ACME3"])
    truth = _logistic(20)[-1]
    assert score["ACME3"] == pytest.approx(np.log(truth / xs[-1]), abs=2e-3)


def test_no_diffusion_score_with_a_gap_or_a_non_positive_quarter():
    xs = _logistic(16)
    gap = _rows("GAPP", "revenue", QUARTERS.delete(8), xs[:8] + xs[9:])            # 14 pairs minus the gap's 2
    zero = _rows("ZERO", "revenue", QUARTERS, [0.0] + xs[1:])
    score = diffusion_score(_data(gap + zero))(T("2020-03-01"), ["GAPP3", "ZERO3"])
    assert "ZERO3" not in score.index
    assert "GAPP3" in score.index                    # 13 consecutive pairs remain: 12 needed
    short = _rows("SHRT", "revenue", QUARTERS[:13].delete(6), xs[:6] + xs[7:13])   # 11 pairs
    assert diffusion_score(_data(short))(T("2020-03-01"), ["SHRT3"]).empty


# E. Market encroachment

def test_encroachment_is_the_slope_of_the_logit_share():
    ends = pd.date_range("2017-03-31", periods=12, freq="QE")
    share = np.linspace(0.1, 0.2, 12)
    rows = _rows("RISE", "revenue", ends, 100 * share)
    rows += _rows("PEER", "revenue", ends, 100 * (1 - share) / 2) + _rows("OTHR", "revenue", ends, 100 * (1 - share) / 2)
    data = _data(rows, activity={"RISE": "Varejo", "PEER": "Varejo", "OTHR": "Varejo"})
    score = encroachment_score(data)(T("2020-03-01"), ["RISE3", "PEER3", "OTHR3"])
    years = (ends - ends[0]).days / 365.25
    logit = np.log(share / (1 - share))
    assert score["RISE3"] == pytest.approx(np.polyfit(years, logit, 1)[0], rel=1e-9)


def test_a_new_listing_does_not_change_the_incumbents_share():
    ends = pd.date_range("2017-03-31", periods=12, freq="QE")
    rows = _rows("AAAA", "revenue", ends, [10.0] * 12) + _rows("BBBB", "revenue", ends, [10.0] * 12)
    rows += _rows("CCCC", "revenue", ends, [10.0] * 12) + _rows("NEWW", "revenue", ends[6:], [50.0] * 6)
    data = _data(rows, activity={k: "Varejo" for k in ("AAAA", "BBBB", "CCCC", "NEWW")})
    score = encroachment_score(data)(T("2020-03-01"), ["AAAA3", "BBBB3", "CCCC3", "NEWW3"])
    assert score["AAAA3"] == pytest.approx(0.0, abs=1e-12)
    assert "NEWW3" not in score.index                # 6 quarters only


# B. Innovation effort

def _effort_rows(issuer: str, intangibles: tuple, fixed: tuple, dep: float, assets: tuple) -> list[dict]:
    ends = [T("2018-12-31"), T("2019-12-31")]
    return (_rows(issuer, "intangibles", ends, intangibles) + _rows(issuer, "fixed_assets", ends, fixed)
            + _rows(issuer, "depreciation", ends[1:], [dep]) + _rows(issuer, "total_assets", ends, assets))


def test_effort_is_the_change_over_lagged_assets_minus_the_sector_median():
    rows = (_effort_rows("AAAA", (10, 30), (50, 50), -5, (100, 120)) + _effort_rows("BBBB", (10, 20), (50, 50), -5, (100, 120))
            + _effort_rows("CCCC", (10, 10), (50, 50), -5, (100, 120)))
    sectors = {"AAAA3": "Bens Industriais", "BBBB3": "Bens Industriais", "CCCC3": "Bens Industriais"}
    score = effort_score(_data(rows), sectors)(T("2020-06-01"), list(sectors))
    # B1: 0.2, 0.1, 0.0 (median 0.1); B2: 0.25, 0.15, 0.05 (median 0.15): ranks agree, AAAA first
    assert score.sort_values().index.tolist() == ["CCCC3", "BBBB3", "AAAA3"]
    assert score["BBBB3"] == pytest.approx(0.0, abs=1e-12)


def test_financials_have_no_effort_score():
    rows = _effort_rows("BANK", (10, 30), (50, 50), -5, (100, 120))
    assert effort_score(_data(rows), {"BANK4": "Financeiro"})(T("2020-06-01"), ["BANK4"]).empty


# P. Patents

def test_patent_intensity_and_acceleration():
    month = T("2020-01-01")
    published = [T("2019-06-01"), T("2018-09-01"), T("2018-03-01"), T("2017-01-01")]
    patents = [(f"PI{i}", "11111111", d - pd.DateOffset(months=18), d, [4]) for i, d in enumerate(published)]
    rows = _rows("ACME", "total_assets", [T("2018-09-30"), T("2019-09-30")], [2e9, 3e9])
    rows += _rows("NONE", "total_assets", [T("2018-09-30"), T("2019-09-30")], [1e9, 1e9])
    data = _data(rows, patents=patents, roots={"ACME": "11111111", "NONE": "22222222"})
    p1, p2 = patent_components(data, month)
    assert p1["ACME"] == pytest.approx(4 / 2.0)            # 4 in 60 months per R$ bn of assets a year before
    assert p2["ACME"] == pytest.approx((3 - 1) / (1 + 3 + 1))
    assert p1["NONE"] == 0.0 and p2["NONE"] == 0.0


def test_an_issuer_without_patents_scores_zero_not_missing():
    rows = _rows("NONE", "total_assets", [T("2018-09-30"), T("2019-09-30")], [1e9, 1e9])
    rows += _rows("ACME", "total_assets", [T("2018-09-30"), T("2019-09-30")], [1e9, 1e9])
    patents = [("PI1", "11111111", T("2017-01-01"), T("2018-06-01"), [4])]
    data = _data(rows, patents=patents, roots={"ACME": "11111111", "NONE": "22222222"})
    score = patent_score(data)(T("2020-01-01"), ["NONE3", "ACME3"])
    assert set(score.index) == {"NONE3", "ACME3"} and score["ACME3"] > score["NONE3"]


def test_patents_after_the_snapshot_are_held_at_2026_01():
    rows = _rows("ACME", "total_assets", [T("2024-09-30"), T("2025-09-30")], [1e9, 1e9])
    patents = [("PI1", "11111111", T("2023-01-01"), T("2024-03-01"), [4])]   # recent at 2026-01, older by 2026-09
    data = _data(rows, patents=patents, roots={"ACME": "11111111"})
    assert patent_components(data, T("2026-09-01"))[1]["ACME"] == pytest.approx(
        patent_components(data, T("2026-01-01"))[1]["ACME"])


# X. Price convexity

def test_convexity_is_positive_for_an_accelerating_path():
    months = pd.date_range("2017-01-01", periods=36, freq="MS")
    t = np.arange(36)
    rng = np.random.default_rng(3)
    accelerating = np.expm1(0.001 + 0.0016 * t + rng.normal(0, 0.002, 36))   # growing increments: a quadratic path
    straight = np.expm1(np.full(36, 0.01) + rng.normal(0, 0.002, 36))
    returns = pd.DataFrame({"FAST3": accelerating, "LINE3": straight}, index=months)
    score = convexity_score(returns)(T("2020-01-01"), ["FAST3", "LINE3"])
    # the path is a random walk, so the OLS t is inflated: only the ranking is used (normal scores), not its scale
    assert score["FAST3"] > 3 and score["FAST3"] > score["LINE3"] + 10


def test_field_counts_add_one_per_distinct_field_in_the_patent_window():
    patents = [("PI1", "11111111", T("2017-01-01"), T("2018-06-01"), [4, 10]),
               ("PI2", "11111111", T("2017-01-01"), T("2019-06-01"), [4]),
               ("PI3", "11111111", T("2009-01-01"), T("2010-06-01"), [20])]           # older than 60 months
    counts = field_counts(_data([], patents=patents, roots={"ACME": "11111111"}), T("2020-01-01"))
    assert counts.loc["ACME", 4] == 2 and counts.loc["ACME", 10] == 1 and counts.loc["ACME", 20] == 0
    assert list(counts.columns) == list(range(1, 36))


def test_a_single_quarter_has_no_diffusion_score():
    rows = _rows("ONEE", "revenue", [T("2010-12-31")], [235249.0])
    assert diffusion_score(_data(rows))(T("2012-01-01"), ["ONEE3"]).empty
