"""Tests of core/periodos.py: the Conservadora's frozen results by period and calendar year (screen, A2)."""

import numpy as np
import pandas as pd
import pytest

from core.periodos import desde_inicio_comum, medidas_por_ano, medidas_por_periodo, retornos_por_ano


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


def _ano() -> tuple[pd.DataFrame, pd.Series]:
    """X doubles BOVA11's daily moves; two months of 2024 after the 2023 close; CDI 0.1% a session."""
    datas = pd.to_datetime(["2023-12-29", "2024-01-15", "2024-01-31", "2024-02-15", "2024-02-29"])
    bova = [100.0, 105.0, 99.75, 104.7375, 99.500625]                     # +5%, -5%, +5%, -5%
    x = [100.0, 110.0, 99.0, 118.8, 95.04]                                  # +10%, -10%, +20%, -20%
    curvas = pd.DataFrame({"X": x, "BOVA11": bova}, index=datas)
    curvas.loc[datas[3:], "X"] = [108.9, 98.01]                             # +10%, -10% instead: beta 2 throughout
    return curvas, pd.Series(0.001, index=datas)


def test_year_measures_return_excess_drawdown_calmar_worst_month_and_beta():
    curvas, cdi = _ano()
    medidas = medidas_por_ano(curvas, cdi).loc[(2024, "X")]
    assert medidas["Retorno"] == pytest.approx(98.01 / 100 - 1)
    assert medidas["Excesso sobre o CDI"] == pytest.approx(98.01 / 100 - 1 - (1.001 ** 4 - 1))
    assert medidas["Queda máx."] == pytest.approx(98.01 / 110 - 1)
    assert medidas["Calmar"] == pytest.approx((98.01 / 100 - 1) / abs(98.01 / 110 - 1))
    assert medidas["Pior mês"] == pytest.approx(min(99.0 / 100 - 1, 98.01 / 99.0 - 1))
    assert medidas["Beta (BOVA11)"] == pytest.approx(2.0)
    assert medidas["Sessões"] == 4


def test_year_vol_sharpe_and_sortino_use_daily_returns_over_the_cdi_annualized_by_252():
    curvas, cdi = _ano()
    medidas = medidas_por_ano(curvas, cdi).loc[(2024, "X")]
    diarios = np.array([0.10, -0.10, 0.10, -0.10])
    excesso = diarios - 0.001
    assert medidas["Vol"] == pytest.approx(diarios.std(ddof=1) * np.sqrt(252))
    assert medidas["Sharpe"] == pytest.approx(excesso.mean() * 252 / (diarios.std(ddof=1) * np.sqrt(252)))
    queda = np.sqrt(np.mean(np.minimum(excesso, 0.0) ** 2)) * np.sqrt(252)
    assert medidas["Sortino"] == pytest.approx(excesso.mean() * 252 / queda)


def test_a_column_that_starts_mid_year_is_measured_on_its_own_sessions_only():
    curvas, cdi = _ano()
    curvas["NOVO"] = [np.nan, np.nan, 50.0, 55.0, 60.5]
    medidas = medidas_por_ano(curvas, cdi).loc[(2024, "NOVO")]
    assert medidas["Retorno"] == pytest.approx(60.5 / 50.0 - 1)
    assert medidas["Sessões"] == 2


def _dois_anos() -> tuple[pd.DataFrame, pd.Series]:
    """X: +21% in 2024 and -10% in 2025 (beta 2 to BOVA11 throughout); BOVA11 +10% and -5%; NOVO lists in 2025."""
    rng = np.random.default_rng(7)
    datas = pd.bdate_range("2023-12-29", "2025-12-31")
    bova = pd.Series(rng.normal(0.0003, 0.01, len(datas)), index=datas)
    bova.iloc[0] = 0.0
    curvas = pd.DataFrame({"X": (1 + 2 * bova).cumprod() * 100, "BOVA11": (1 + bova).cumprod() * 100})
    curvas["NOVO"] = curvas["BOVA11"].where(curvas.index >= "2025-03-03")
    return curvas, pd.Series(0.0002, index=datas)


def test_period_summary_has_the_aggressive_investors_risk_measures_from_each_columns_own_start():
    from core.metrics import cagr, comparar_benchmark, cvar_historico, drawdown
    from core.periodos import resumo_do_periodo
    curvas, cdi = _dois_anos()
    resumo = resumo_do_periodo(curvas, cdi)
    x = curvas["X"]
    benchmark = comparar_benchmark(x, curvas["BOVA11"], cdi)
    assert resumo.loc["X", "CAGR"] == pytest.approx(cagr(x))
    assert resumo.loc["X", "Queda máx."] == pytest.approx(drawdown(x)[1])
    assert resumo.loc["X", "Calmar"] == pytest.approx(cagr(x) / abs(drawdown(x)[1]))
    assert resumo.loc["X", "Beta (BOVA11)"] == pytest.approx(2.0)
    assert resumo.loc["X", "Alfa a.a. (BOVA11)"] == pytest.approx(benchmark.alfa_anual)
    assert resumo.loc["X", "t do alfa"] == pytest.approx(benchmark.t_stat_alfa)
    assert resumo.loc["X", "Information ratio"] == pytest.approx(benchmark.information_ratio)
    assert resumo.loc["X", "CVaR 95% diário"] == pytest.approx(cvar_historico(x.pct_change().dropna()))
    anos = medidas_por_ano(curvas, cdi)["Retorno"].unstack()
    assert resumo.loc["X", "Melhor ano"] == pytest.approx(anos["X"].max())
    assert resumo.loc["X", "Pior ano"] == pytest.approx(anos["X"].min())
    assert list(anos.index) == [2024, 2025]                                # 2023 has one session: no year
    assert resumo.loc["X", "Anos acima do BOVA11"] == f"{int((anos['X'] > anos['BOVA11']).sum())} de 2"
    assert resumo.loc["NOVO", "Anos acima do BOVA11"] == f"{int((anos['NOVO'] > anos['BOVA11']).sum())} de 1"
    assert resumo.loc["NOVO", "Desde"] == pd.Timestamp("2025-03-03")
    assert resumo.loc["NOVO", "CAGR"] == pytest.approx(cagr(curvas["NOVO"].dropna()))
    assert np.isnan(resumo.loc["BOVA11", "t do alfa"]) or abs(resumo.loc["BOVA11", "Alfa a.a. (BOVA11)"]) < 1e-12
