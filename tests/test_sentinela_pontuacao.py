import math

import numpy as np
import pandas as pd
import pytest

from core.sentinela import pontuacao


def test_brier_and_log_scores():
    assert pontuacao.brier(0.8, True) == pytest.approx(0.04)
    assert pontuacao.brier(0.8, False) == pytest.approx(0.64)
    assert pontuacao.log_score(0.8, True) == pytest.approx(-math.log(0.8))
    assert pontuacao.log_score(0.8, False) == pytest.approx(-math.log(0.2))
    assert math.isfinite(pontuacao.log_score(1.0, False))


def test_the_probability_of_a_fall_is_the_confidence_for_a_sale_and_its_complement_for_manter():
    assert pontuacao.p_de_queda("vender", 0.8) == 0.8
    assert pontuacao.p_de_queda("reduzir_metade", 0.7) == 0.7
    assert pontuacao.p_de_queda("manter", 0.9) == pytest.approx(0.1)


def _precos(queda: float) -> pd.DataFrame:
    gerador = np.random.default_rng(1)
    dias = pd.bdate_range("2026-06-01", periods=120)
    retornos = gerador.normal(0, 0.01, len(dias))
    retornos[90:110] = queda / 20
    return pd.DataFrame({"PETR4": 30 * np.cumprod(1 + retornos)}, index=dias)


def test_outcome_after_5_and_20_sessions_and_the_fall_event_beyond_1_5_sigma():
    precos = _precos(-0.30)
    dia = precos.index[89]
    resultado = pontuacao.resultado(precos["PETR4"], dia)
    assert resultado.ret_20 == pytest.approx(precos["PETR4"].iloc[109] / precos["PETR4"].iloc[89] - 1)
    assert resultado.ret_5 == pytest.approx(precos["PETR4"].iloc[94] / precos["PETR4"].iloc[89] - 1)
    assert resultado.queda and resultado.movimento
    calmo = pontuacao.resultado(_precos(0.0)["PETR4"], dia)
    assert not calmo.queda


def test_no_outcome_before_20_sessions_have_passed():
    precos = _precos(0.0)
    assert pontuacao.resultado(precos["PETR4"], precos.index[110]) is None


def test_scores_table_fills_outcomes_and_scores_of_decisions_and_judges():
    precos = _precos(-0.30)
    dia = precos.index[89]
    registro = pd.DataFrame([
        {"hora": dia + pd.Timedelta(hours=10), "ticker": "PETR4", "etapa": "decisao", "acao": "vender",
         "confianca": 0.8, "p": None},
        {"hora": dia + pd.Timedelta(hours=10), "ticker": "PETR4", "etapa": "juiz:d1.1", "acao": None,
         "confianca": None, "p": 0.9}])
    tabela = pontuacao.pontuar(registro, precos)
    assert list(tabela["p_evento"]) == [pytest.approx(0.8), pytest.approx(0.9)]
    assert list(tabela["evento"]) == [True, True]
    assert tabela["brier"].iloc[0] == pytest.approx(0.04)
    assert tabela["log"].iloc[1] == pytest.approx(-math.log(0.9))


def test_autonomy_is_kept_only_if_70_percent_calls_hit_55_percent_and_brier_beats_the_base_rate():
    def tabela(acertos: list[bool], ps: list[float]):
        return pd.DataFrame({"etapa": "decisao", "acao": "vender", "confianca": ps, "p_evento": ps,
                             "evento": acertos, "brier": [pontuacao.brier(p, y) for p, y in zip(ps, acertos)]})
    boa = pontuacao.autonomia(tabela([True] * 5 + [True] * 3 + [False] * 2, [0.9] * 5 + [0.7] * 5))
    assert boa.mantem and boa.acerto == pytest.approx(0.8) and boa.brier < boa.brier_base
    assert not pontuacao.autonomia(tabela([True] * 5 + [False] * 5, [0.8] * 10)).mantem      # 50% right
    pior_que_a_base = pontuacao.autonomia(tabela([True] * 6 + [False] * 4, [0.8] * 10))      # 60% right, overconfident
    assert pior_que_a_base.brier > pior_que_a_base.brier_base and not pior_que_a_base.mantem
    assert pontuacao.autonomia(tabela([], [])).mantem is None
