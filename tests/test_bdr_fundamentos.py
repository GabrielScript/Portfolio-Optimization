"""The Composite's score on B3 BDRs (spec 2026-11-14-composto-bdr-fundamentos): B3 price defects and monthly returns."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from core import bdr_fundamentos as bf

DIAS = pd.bdate_range("2024-09-30", periods=6)


def test_a_flagged_jump_session_returns_zero():
    precos = pd.DataFrame({"L1RC34": [100.0, 96.0, 10.1, 10.3, 10.2, 10.4]}, index=DIAS)
    saltos = pd.DataFrame({"security": ["L1RC34"], "date": [DIAS[2]]})
    r = bf.retornos_corrigidos(precos, saltos)
    assert r.loc[DIAS[2], "L1RC34"] == 0.0
    assert r.loc[DIAS[3], "L1RC34"] == pytest.approx(10.3 / 10.1 - 1)


def test_a_print_error_undone_the_next_session_is_zeroed_and_a_real_crash_is_kept():
    precos = pd.DataFrame({"AVGO34": [100.0, 101.0, 60.0, 100.5, 101.0, 102.0],
                           "F1RC34": [100.0, 99.0, 55.0, 54.0, 50.0, 49.0]}, index=DIAS)
    r = bf.retornos_corrigidos(precos, pd.DataFrame(columns=["security", "date"]))
    assert r.loc[DIAS[2], "AVGO34"] == 0.0 and r.loc[DIAS[3], "AVGO34"] == 0.0
    assert r.loc[DIAS[2], "F1RC34"] == pytest.approx(55.0 / 99.0 - 1)


def test_a_session_without_a_trade_has_no_return_and_the_next_trade_carries_the_move():
    precos = pd.DataFrame({"AAPL34": [10.0, np.nan, 11.0, 11.0, 11.0, 11.0]}, index=DIAS)
    r = bf.retornos_corrigidos(precos, pd.DataFrame(columns=["security", "date"]))
    assert np.isnan(r.loc[DIAS[1], "AAPL34"])
    assert r.loc[DIAS[2], "AAPL34"] == pytest.approx(0.1)


def test_monthly_returns_compound_the_month_and_a_month_without_a_trade_is_empty():
    dias = pd.bdate_range("2024-01-01", "2024-03-29")
    r = pd.DataFrame({"AAPL34": 0.0}, index=dias)
    r.loc["2024-01-02", "AAPL34"] = 0.10
    r.loc["2024-01-03", "AAPL34"] = 0.10
    r.loc["2024-02-01":"2024-02-29", "AAPL34"] = np.nan
    mensal = bf.retornos_mensais(r)
    assert list(mensal.index) == [pd.Timestamp("2024-01-01"), pd.Timestamp("2024-02-01"), pd.Timestamp("2024-03-01")]
    assert mensal.loc["2024-01-01", "AAPL34"] == pytest.approx(0.21)
    assert np.isnan(mensal.loc["2024-02-01", "AAPL34"])


def test_a_renamed_bdr_is_one_series_the_old_code_then_the_new():
    dias = pd.bdate_range("2022-06-16", periods=6)
    tabela = pd.DataFrame({"FBOK34": [0.01, 0.02, 0.03, np.nan, np.nan, np.nan],
                           "M1TA34": [np.nan, np.nan, np.nan, np.nan, 0.04, 0.05]}, index=dias)
    junta = bf.juntar(tabela, "FBOK34", "M1TA34")
    assert list(junta.fillna(-1)) == [0.01, 0.02, 0.03, -1, 0.04, 0.05]
