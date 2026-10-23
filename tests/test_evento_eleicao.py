"""Pieces of ``scripts/evento_eleicao.py`` (the election-window volatility study, pre-registered)."""
import numpy as np
import pandas as pd
import pytest


def test_day_zero_is_the_first_session_after_the_sunday_vote_even_after_a_monday_holiday():
    from scripts.evento_eleicao import dia_zero
    sessoes = pd.DatetimeIndex(["2022-09-30", "2022-10-03", "2022-10-04", "2022-10-28", "2022-11-01"])
    assert dia_zero(sessoes, pd.Timestamp("2022-10-02")) == 1
    assert dia_zero(sessoes, pd.Timestamp("2022-10-30")) == 4  # 31/10 missing from the sessions: the next one


def test_the_market_is_the_mean_return_of_the_months_members_that_traded_that_day_and_the_day_before():
    from scripts.evento_eleicao import retorno_de_mercado
    precos = pd.DataFrame({"AAAA3": [10.0, 11.0, 11.0, 12.1], "BBBB4": [20.0, 19.0, np.nan, 19.0],
                           "CCCC3": [5.0, 10.0, 20.0, 40.0]},
                          index=pd.to_datetime(["2022-09-29", "2022-09-30", "2022-10-03", "2022-10-04"]))
    membros = pd.DataFrame({"month": pd.to_datetime(["2022-09-01", "2022-09-01", "2022-10-01", "2022-10-01"]),
                            "security": ["AAAA3", "BBBB4", "AAAA3", "BBBB4"]})  # CCCC3 never a member
    mercado = retorno_de_mercado(precos, membros)
    assert list(mercado.index) == list(pd.to_datetime(["2022-09-30", "2022-10-03", "2022-10-04"]))
    assert mercado.to_numpy() == pytest.approx([(0.10 - 0.05) / 2, 0.0, 0.10])  # BBBB4 has no return on 3 or 4/10


def test_the_measures_compare_sessions_0_to_4_with_sessions_minus_60_to_minus_11():
    from scripts.evento_eleicao import medidas_do_evento
    r = np.full(100, 0.01)
    r[70:75] = 0.02  # sessions 0..+4 of day 0 = 70 at twice the usual size
    r[65:70] = 0.03  # sessions -5..-1
    r[60:65] = 0.50  # sessions -10..-6: outside every window
    m = medidas_do_evento(r, 70)
    assert m.principal == pytest.approx(np.log(2.0))
    assert m.antecipacao == pytest.approx(np.log(3.0))
    assert m.dia0 == pytest.approx(2.0)
    with pytest.raises(ValueError, match="window"):
        medidas_do_evento(r, 59)  # sessions -60.. would start before the data
    with pytest.raises(ValueError, match="window"):
        medidas_do_evento(r, 96)  # sessions ..+4 would end after the data


def test_placebo_days_open_a_week_sit_60_sessions_from_every_round_and_have_full_windows():
    from scripts.evento_eleicao import dias_placebo
    sessoes = pd.bdate_range("2021-01-04", periods=400)  # Mondays at 0, 5, 10, ...
    sessoes = sessoes.delete(100)  # a Monday holiday: the Tuesday, now at 100, opens that week; Mondays then at 4 mod 5
    placebo = dias_placebo(sessoes, eleicoes=[200])
    assert 100 in placebo and 264 in placebo
    assert 154 not in placebo and 59 not in placebo  # 46 sessions from the round; window before the data
    assert all(p >= 60 and p + 5 <= len(sessoes) and abs(p - 200) >= 60 for p in placebo)
    assert all(sessoes[p].isocalendar().week != sessoes[p - 1].isocalendar().week for p in placebo)
    assert len(placebo) == len(set(placebo))


def test_the_p_value_counts_draws_of_ten_placebo_means_at_or_above_the_observed_mean_plus_one():
    from scripts.evento_eleicao import p_placebo
    placebo = np.arange(30, dtype=float)  # ten distinct values: means from 4.5 to 24.5
    assert p_placebo(25.0, placebo, n_eventos=10, sorteios=999, semente=1) == pytest.approx(1 / 1000)
    assert p_placebo(4.5, placebo, n_eventos=10, sorteios=999, semente=1) == pytest.approx(1.0)
    middle = p_placebo(14.5, placebo, n_eventos=10, sorteios=4999, semente=1)
    assert 0.4 < middle < 0.6
    assert middle == p_placebo(14.5, placebo, n_eventos=10, sorteios=4999, semente=1)  # the seed fixes it
