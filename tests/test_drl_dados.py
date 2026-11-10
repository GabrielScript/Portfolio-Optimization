from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from drl import dados


def _painel(sessoes: int = 300) -> pd.DataFrame:
    """Two names, ``sessoes`` business days: A trades every session, B stops after session 100."""
    datas = pd.bdate_range("2020-01-02", periods=sessoes)
    a = np.linspace(10.0, 20.0, sessoes)
    b = np.linspace(30.0, 31.0, sessoes)
    b[100:] = np.nan
    c = np.full(sessoes, np.nan)
    return pd.DataFrame({"A": a, "B": b, "C": c}, index=datas)


def test_the_panel_keeps_the_names_that_die_and_drops_the_empty_ones(tmp_path):
    arquivo = tmp_path / "precos.parquet"
    _painel().to_parquet(arquivo)
    p = dados.painel(arquivo)
    assert list(p.columns) == ["A", "B"]          # C never traded: no column
    assert p.index.is_monotonic_increasing


def test_the_panel_cuts_the_sessions_and_never_the_columns(tmp_path):
    arquivo = tmp_path / "precos.parquet"
    _painel().to_parquet(arquivo)
    p = dados.painel(arquivo, inicio="2020-06-01", fim="2020-12-31")
    assert p.index.min() >= pd.Timestamp("2020-06-01")
    assert p.index.max() <= pd.Timestamp("2020-12-31")
    assert "B" in p.columns                          # B died inside the window and stays


def test_a_return_after_a_gap_is_not_invented():
    p = _painel()
    r = dados.retornos(p)
    assert np.isnan(r["A"].iloc[0])                  # the first session has no previous close
    assert r["B"].iloc[100] is np.nan or np.isnan(r["B"].iloc[100])
    assert r["A"].iloc[1] == pytest.approx(p["A"].iloc[1] / p["A"].iloc[0] - 1.0)


def test_a_name_becomes_eligible_only_after_252_valid_closes():
    p = _painel()
    e = dados.elegiveis(p)
    assert not e["A"].iloc[250]
    assert e["A"].iloc[251]                          # the 252nd valid close is the session it becomes tradeable on
    assert not e["B"].iloc[251]                      # B stopped at session 100


def test_a_name_that_stops_trading_is_not_offered_again():
    p = _painel()
    e = dados.elegiveis(p)
    assert not e["B"].iloc[101:].any()


def test_the_return_window_has_one_row_per_name_and_zero_fills_gaps():
    p = _painel()
    j = dados.janela_retornos(dados.retornos(p), ate=260, tamanho=60)
    assert j.shape == (p.shape[1], 60)
    assert np.isfinite(j).all()                      # a dead name's gap is 0, not a NaN that poisons the policy
    assert j[0, -1] == pytest.approx(np.log(1 + (p["A"].iloc[260] / p["A"].iloc[259] - 1)))
