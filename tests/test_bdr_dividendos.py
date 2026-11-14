"""BDR dividends from the US underlying (spec 2026-11-14-bdr-setores, addendum 2)."""

from __future__ import annotations

import pandas as pd
import pytest

from core import bdr_dividendos as bd


def _eua(datas: list[str], fechamentos: list[float], ajustados: list[float]) -> pd.DataFrame:
    return pd.DataFrame({"t": datas, "c": fechamentos, "a": ajustados})


def test_a_dividend_is_the_jump_of_adjusted_over_close_on_the_ex_date():
    # adj/close goes 0.98 -> 0.99 on 2022-02-14: a dividend of 0.99 / 0.98 - 1
    diario = _eua(["2022-02-10", "2022-02-11", "2022-02-14", "2022-02-15"], [100, 100, 99, 99], [98, 98, 98.01, 98.01])
    eventos = bd.eventos(diario, _eua([], [], []))
    assert list(eventos.index) == [pd.Timestamp("2022-02-14")]
    assert eventos.iloc[0] == pytest.approx(0.99 / 0.98 - 1)


def test_rounding_noise_of_the_adjusted_series_cancels_instead_of_adding_up():
    # a stock without dividends whose adjusted price is rounded differently from its close
    diario = _eua(["2022-03-01", "2022-03-02", "2022-03-03"], [100.0, 100.0, 100.0], [100.0, 100.01, 100.0])
    eventos = bd.eventos(diario, _eua([], [], []))
    assert (1 + eventos).prod() == pytest.approx(1.0)


def test_monthly_events_count_only_before_the_daily_file_and_sit_on_the_months_last_day():
    mensal = _eua(["2021-01-04", "2021-02-01", "2021-11-01", "2021-12-01"], [10, 10, 10, 10], [9.0, 9.1, 9.1, 9.2])
    diario = _eua(["2021-11-01", "2021-11-02"], [10, 10], [9.1, 9.1])
    eventos = bd.eventos(diario, mensal)
    assert list(eventos.index) == [pd.Timestamp("2021-02-28")]      # December's step is the daily file's period


def test_the_bdr_gets_70_percent_from_the_first_b3_session_on_or_after_the_us_date():
    sessoes = pd.DatetimeIndex(["2022-02-11", "2022-02-15", "2022-02-16"])   # no B3 session on the 14th
    tr = pd.Series([50.0, 50.0, 50.0], index=sessoes)
    eventos = pd.Series({pd.Timestamp("2022-02-14"): 0.01})
    ajustado = bd.aplicar(tr, eventos, corte=None)
    assert list(ajustado) == pytest.approx([50.0, 50.0 * 1.007, 50.0 * 1.007])


def test_a_jump_above_20_percent_is_a_spin_off_applied_in_full():
    sessoes = pd.DatetimeIndex(["2021-11-01", "2021-11-03"])
    tr = pd.Series([24.7, 12.2], index=sessoes)
    ajustado = bd.aplicar(tr, pd.Series({pd.Timestamp("2021-11-02"): 1.0}), corte=None)
    assert ajustado.iloc[1] == pytest.approx(24.4)


def test_events_from_the_first_b3_cash_event_on_are_left_to_b3():
    sessoes = pd.DatetimeIndex(["2025-08-01", "2025-11-03", "2025-11-20"])
    tr = pd.Series([10.0, 10.0, 10.0], index=sessoes)
    eventos = pd.Series({pd.Timestamp("2025-08-01"): 0.01, pd.Timestamp("2025-11-20"): 0.01})
    ajustado = bd.aplicar(tr, eventos, corte=pd.Timestamp("2025-11-18"))
    assert list(ajustado) == pytest.approx([10.07, 10.07, 10.07])
