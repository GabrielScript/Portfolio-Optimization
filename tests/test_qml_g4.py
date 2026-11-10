"""Tests for the G4 script helpers (no lake): decision dates of the cache and the top-quintile mask."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scripts.qml_g4_evaluate import decision_dates, top_share_mask


def test_decision_dates_keep_the_first_session_of_each_pair() -> None:
    sessoes = pd.bdate_range("2015-01-01", periods=120)
    cache = pd.DatetimeIndex([sessoes[i] for i in (10, 11, 31, 32, 52, 53, 73)])   # last pair has no follower
    assert list(decision_dates(cache[::-1], sessoes)) == [sessoes[i] for i in (10, 31, 52, 73)]


def test_decision_dates_refuse_a_date_outside_the_sessions() -> None:
    sessoes = pd.bdate_range("2015-01-01", periods=20)
    with pytest.raises(ValueError):
        decision_dates(pd.DatetimeIndex([pd.Timestamp("2015-01-03")]), sessoes)      # a Saturday


def test_top_share_mask_takes_the_top_fifth_of_each_date_and_ignores_missing_scores() -> None:
    datas = pd.bdate_range("2015-01-01", periods=2)
    score = pd.DataFrame([np.arange(10.0), np.r_[np.arange(9.0), np.nan]], index=datas,
                         columns=[f"N{i}" for i in range(10)])
    mascara = top_share_mask(score, 0.20)
    assert list(mascara.columns[mascara.iloc[0]]) == ["N8", "N9"]
    assert "N9" not in mascara.columns[mascara.iloc[1]]
    assert mascara.iloc[1].sum() == 2
