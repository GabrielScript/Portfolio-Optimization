"""The forecast's reading (design, section 3): monthly Spearman IC, Newey-West t, the gate."""

import numpy as np
import pandas as pd
import pytest

from core.qml_ic import IC_MINIMO, T_MINIMO, ic_do_alvo, passa_no_ic, t_newey_west


def test_without_lags_the_t_is_the_plain_one_and_positive_autocorrelation_lowers_it():
    gerador = np.random.default_rng(0)
    independente = pd.Series(gerador.normal(0.02, 0.1, 3_000))
    simples = independente.mean() / (independente.std(ddof=0) / np.sqrt(len(independente)))
    assert t_newey_west(independente, 0) == pytest.approx(simples)
    autocorrelada = pd.Series(np.convolve(gerador.normal(0.02, 0.1, 3_002), np.ones(3) / 3, mode="valid"))
    assert t_newey_west(autocorrelada, 2) < t_newey_west(autocorrelada, 0) * 0.8


def test_the_gate_needs_both_the_size_and_the_t():
    assert (IC_MINIMO, T_MINIMO) == (0.02, 2.0)
    assert passa_no_ic(0.03, 2.5)
    assert not passa_no_ic(0.01, 5.0)
    assert not passa_no_ic(0.05, 1.5)


def test_the_ic_is_a_spearman_per_month_and_ignores_months_without_a_closed_target():
    tabela = pd.DataFrame({"month": pd.to_datetime(["2020-01-01"] * 3 + ["2020-02-01"] * 3 + ["2020-03-01"] * 3),
                           "previsao": [1.0, 2.0, 3.0, 3.0, 2.0, 1.0, 1.0, 2.0, 3.0],
                           "y": [0.1, 0.5, 9.0, 0.1, 0.5, 9.0, np.nan, np.nan, np.nan]})
    ics = ic_do_alvo(tabela, "previsao", "y")
    assert list(ics.index) == [pd.Timestamp("2020-01-01"), pd.Timestamp("2020-02-01")]
    assert ics.iloc[0] == pytest.approx(1.0) and ics.iloc[1] == pytest.approx(-1.0)
