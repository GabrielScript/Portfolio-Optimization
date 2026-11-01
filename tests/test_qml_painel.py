"""The feature panel of the hybrid forecast (docs/superpowers/specs/2026-11-01-qpo-hibrido-design.md, section 3)."""

import numpy as np
import pandas as pd
import pytest

from core.qml_painel import alvo_3m, angulos, baixa_volatilidade, linhas_de_treino, sem_vazamento


def _retornos(meses: int = 48, nomes: int = 4, semente: int = 0) -> pd.DataFrame:
    gerador = np.random.default_rng(semente)
    return pd.DataFrame(gerador.normal(0.01, 0.05, (meses, nomes)), columns=[f"P{i}" for i in range(nomes)],
                        index=pd.date_range("2015-01-01", periods=meses, freq="MS"))


def test_an_angle_is_the_percentile_among_those_with_a_value_and_missing_is_neutral():
    nota = pd.Series({"A": 3.0, "B": 1.0, "C": np.nan, "D": 2.0})
    resultado = angulos(nota)
    assert resultado["B"] == pytest.approx(np.pi * (1 / 6 - 0.5))
    assert resultado["D"] == pytest.approx(0.0)
    assert resultado["A"] == pytest.approx(np.pi * (5 / 6 - 0.5))
    assert resultado["C"] == 0.0
    assert resultado.abs().max() < np.pi / 2


def test_low_volatility_reads_only_months_before_the_decision_and_needs_twelve():
    retornos = _retornos()
    mes = retornos.index[20]
    alterados = retornos.copy()
    alterados.loc[mes:, "P0"] = 9.0                                    # a huge return from the decision month on
    assert baixa_volatilidade(alterados, mes)["P0"] == pytest.approx(baixa_volatilidade(retornos, mes)["P0"])
    assert baixa_volatilidade(retornos, mes)["P0"] == pytest.approx(-retornos["P0"].iloc[:20].std(ddof=1))
    assert np.isnan(baixa_volatilidade(retornos, retornos.index[11])["P0"])         # 11 months only
    assert not np.isnan(baixa_volatilidade(retornos, retornos.index[12])["P0"])


def test_low_volatility_stops_at_36_months():
    retornos = _retornos(meses=60)
    mes = retornos.index[50]
    assert baixa_volatilidade(retornos, mes)["P1"] == pytest.approx(-retornos["P1"].iloc[14:50].std(ddof=1))


def test_the_target_is_the_normal_score_of_the_three_months_from_the_decision_month():
    retornos = pd.DataFrame({"A": [0.1, 0.1, 0.1, 0.5], "B": [0.0, 0.0, 0.0, 0.5], "C": [-0.1, -0.1, -0.1, 0.5]},
                            index=pd.date_range("2020-01-01", periods=4, freq="MS"))
    alvo = alvo_3m(retornos, retornos.index[0], ["A", "B", "C"])
    assert alvo["A"] > alvo["B"] > alvo["C"]
    assert alvo.sum() == pytest.approx(0.0)


def test_a_name_that_dies_keeps_its_loss_in_the_target_instead_of_leaving():
    # dropping it would teach the model on survivors only
    retornos = pd.DataFrame({"A": [0.1, 0.1, 0.1], "B": [0.0, 0.0, 0.0], "MORTA": [-0.9, np.nan, np.nan]},
                            index=pd.date_range("2020-01-01", periods=3, freq="MS"))
    alvo = alvo_3m(retornos, retornos.index[0], ["A", "B", "MORTA"])
    assert alvo["MORTA"] < alvo["B"] < alvo["A"]


def test_the_target_does_not_exist_until_its_three_months_have_closed():
    retornos = _retornos(meses=6)
    assert alvo_3m(retornos, retornos.index[3], list(retornos.columns)).notna().all()
    assert alvo_3m(retornos, retornos.index[4], list(retornos.columns)).isna().all()


def test_training_for_a_month_stops_where_the_last_label_has_closed():
    # the purge: s <= t - 3, so the label of s ends at t - 1, before the decision
    painel = pd.DataFrame({"month": pd.date_range("2019-01-01", periods=24, freq="MS").repeat(2), "y": 0.0})
    mes = pd.Timestamp("2020-06-01")
    usados = painel.loc[linhas_de_treino(painel["month"], mes), "month"]
    assert usados.max() == pd.Timestamp("2020-03-01")
    assert sem_vazamento(usados, mes)
    assert not sem_vazamento(pd.Series([pd.Timestamp("2020-04-01")]), mes)


def test_training_keeps_the_last_120_months_and_at_most_20000_rows_the_same_for_every_model():
    meses = pd.Series(pd.date_range("2005-01-01", periods=180, freq="MS").repeat(200))
    mes = pd.Timestamp("2020-01-01")
    primeira = linhas_de_treino(meses, mes)
    assert len(primeira) == 20_000 and np.array_equal(primeira, linhas_de_treino(meses, mes))
    assert meses.iloc[primeira].min() >= pd.Timestamp("2009-10-01")
    assert meses.iloc[primeira].max() == pd.Timestamp("2019-10-01")
    assert not np.array_equal(primeira, linhas_de_treino(meses, pd.Timestamp("2020-02-01")))
