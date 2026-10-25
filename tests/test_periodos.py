"""Tests of core/periodos.py: the Conservadora's frozen results by period and calendar year (screen, A2)."""

import numpy as np
import pandas as pd
import pytest

from core.periodos import desde_inicio_comum, medidas_por_periodo, retornos_por_ano


def _curvas() -> pd.DataFrame:
    datas = pd.to_datetime(["2023-12-28", "2023-12-29", "2024-06-28", "2024-12-30", "2025-03-31", "2025-06-30"])
    return pd.DataFrame({"Conservadora": [100.0, 110.0, 99.0, 121.0, 130.0, 133.1],
                         "BOVA11": [100.0, 100.0, 90.0, 80.0, 84.0, 88.0]}, index=datas)


def test_retorno_do_ano_vai_do_ultimo_pregao_do_ano_anterior_ao_ultimo_do_ano():
    anos = retornos_por_ano(_curvas())
    assert anos.loc[2024, "Conservadora"] == pytest.approx(121.0 / 110.0 - 1)
    assert anos.loc[2024, "BOVA11"] == pytest.approx(80.0 / 100.0 - 1)
    assert anos.loc[2025, "Conservadora"] == pytest.approx(133.1 / 121.0 - 1)


def test_primeiro_ano_parte_do_primeiro_valor_da_curva():
    anos = retornos_por_ano(_curvas())
    assert anos.loc[2023, "Conservadora"] == pytest.approx(0.10)
    assert anos.loc[2023, "BOVA11"] == pytest.approx(0.0)


def test_ultimos_anos_comecam_no_primeiro_pregao_depois_de_fim_menos_n_anos():
    datas = pd.bdate_range("2019-01-01", "2026-10-16")
    valores = pd.Series(np.linspace(100.0, 300.0, len(datas)), index=datas)
    curvas = pd.DataFrame({"Conservadora": valores, "BOVA11": valores * 0 + 100.0})
    tabela = medidas_por_periodo(curvas, anos=(3,))
    linha = tabela.loc["Últimos 3 anos"]
    assert linha[("Início", "")] == pd.Timestamp("2023-10-17")
    trecho = valores.loc["2023-10-17":]
    esperado = (trecho.iloc[-1] / trecho.iloc[0]) ** (252 / (len(trecho) - 1)) - 1
    assert linha[("Conservadora", "CAGR")] == pytest.approx(esperado)
    assert linha[("BOVA11", "Vol anual")] == pytest.approx(0.0)
    assert tabela.loc["Período inteiro", ("Início", "")] == datas[0]


def test_the_saved_curves_are_one_row_per_phase_and_session_with_both_names():
    from scripts.curvas_conservadora import quadro_das_curvas
    datas = pd.to_datetime(["2026-10-15", "2026-10-16"])
    capturadas = [(0, {"Conservadora (15 nomes, pesos iguais, com regras)": pd.Series([1.0, 1.1], index=datas),
                       "BOVA11": pd.Series([1.0, 0.9], index=datas), "IVVB11": pd.Series([1.0, 1.0], index=datas)}),
                  (21, {"Conservadora (15 nomes, pesos iguais, com regras)": pd.Series([2.0, 2.2], index=datas),
                        "BOVA11": pd.Series([2.0, 1.8], index=datas)})]
    quadro = quadro_das_curvas(capturadas)
    assert list(quadro.columns) == ["fase", "date", "Conservadora", "BOVA11"]
    assert quadro["fase"].tolist() == [0, 0, 21, 21]
    assert quadro.loc[quadro["fase"] == 21, "Conservadora"].tolist() == [2.0, 2.2]


def test_desde_inicio_comum_comeca_quando_todos_tem_preco_e_vale_1():
    datas = pd.bdate_range("2021-08-02", periods=4)
    precos = pd.DataFrame({"BOVA11": [100.0, 110.0, 121.0, 110.0], "HASH11": [None, 50.0, None, 60.0]}, index=datas)
    curvas = desde_inicio_comum(precos)
    assert list(curvas.index) == list(datas[1:])
    assert curvas.iloc[0].tolist() == [1.0, 1.0]
    assert curvas["HASH11"].tolist() == [1.0, 1.0, 1.2]          # a missing day keeps the last price
    assert curvas["BOVA11"].iloc[-1] == 1.0
