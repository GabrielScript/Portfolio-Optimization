"""The three named funds of the user's request of 2026-10-30 (core/tres_fundos.py, DECISOES 189): the 7 combinations,
the invest-in-the-funds returns, the copy of their public CDA holdings and the CDA rows the copy reads."""

import numpy as np
import pandas as pd
import pytest

A, B, C = "11.111.111/0001-11", "22.222.222/0001-22", "33.333.333/0001-33"


def test_the_combinations_are_the_three_together_the_three_pairs_and_each_alone():
    from core.tres_fundos import combinacoes
    assert combinacoes(["x", "y", "z"]) == [("x", "y", "z"), ("x", "y"), ("x", "z"), ("y", "z"), ("x",), ("y",),
                                            ("z",)]


def test_a_combination_invests_equal_parts_each_month_from_the_first_month_every_member_has_a_return():
    from core.tres_fundos import retorno_da_combinacao
    meses = pd.date_range("2024-01-01", periods=4, freq="MS")
    retornos = pd.DataFrame({A: [0.10, 0.02, -0.04, 0.01], B: [np.nan, 0.04, 0.00, np.nan]}, index=meses)
    serie = retorno_da_combinacao(retornos, (A, B))
    assert list(serie.index) == list(meses[1:3])                       # from 02/2024; 04/2024 has no B: stops
    assert serie.iloc[0] == pytest.approx(0.03) and serie.iloc[1] == pytest.approx(-0.02)


def _posicoes() -> pd.DataFrame:
    return pd.DataFrame({"cnpj": [A, A, A, B],
                         "month": pd.to_datetime(["2024-01-31", "2024-01-31", "2024-02-29", "2024-01-31"]),
                         "ticker": ["PETR4", "VALE3", "PETR4", "AURA33"], "value": [30.0, 10.0, 50.0, 95.0],
                         "confid_until": pd.NaT})


def test_the_copy_uses_each_funds_latest_public_book_in_equal_fund_parts():
    # the 31/01 book is public from 03/2024 (t+2), the 29/02 one from 04/2024
    from core.tres_fundos import pesos_copiados
    pesos = pesos_copiados(_posicoes(), pd.Timestamp("2024-03-01"), (A, B))
    assert pesos.to_dict() == pytest.approx({"PETR4": 0.375, "VALE3": 0.125, "AURA33": 0.5})
    assert pesos_copiados(_posicoes(), pd.Timestamp("2024-04-01"), (A,)).to_dict() == {"PETR4": 1.0}


def test_a_member_with_no_public_book_yet_is_left_out_and_none_gives_an_empty_copy():
    from core.tres_fundos import pesos_copiados
    assert pesos_copiados(_posicoes(), pd.Timestamp("2024-03-01"), (A, C)).to_dict() == {"PETR4": 0.75,
                                                                                        "VALE3": 0.25}
    assert pesos_copiados(_posicoes(), pd.Timestamp("2024-02-01"), (A,)).empty


def test_the_copys_month_return_is_the_weighted_total_return_over_the_names_that_have_one():
    from core.tres_fundos import retorno_da_copia
    pesos = pd.Series({"PETR4": 0.5, "VALE3": 0.25, "SUMIU3": 0.25})
    retornos = pd.Series({"PETR4": 0.04, "VALE3": -0.02})
    assert retorno_da_copia(pesos, retornos) == pytest.approx((0.5 * 0.04 - 0.25 * 0.02) / 0.75)
    assert np.isnan(retorno_da_copia(pd.Series(dtype=float), retornos))


SEPARADOR = ";"
CABECALHO = ("TP_FUNDO_CLASSE;CNPJ_FUNDO_CLASSE;DENOM_SOCIAL;DT_COMPTC;TP_APLIC;TP_ATIVO;EMISSOR_LIGADO;TP_NEGOC;"
             "QT_VENDA_NEGOC;VL_VENDA_NEGOC;QT_AQUIS_NEGOC;VL_AQUIS_NEGOC;QT_POS_FINAL;VL_MERC_POS_FINAL;"
             "VL_CUSTO_POS_FINAL;DT_CONFID_APLIC;CD_ATIVO;DS_ATIVO;DT_INI_VIGENCIA;DT_FIM_VIGENCIA;CD_ISIN")


def test_the_cda_reader_keeps_the_stock_and_bdr_rows_of_the_named_funds():
    from core.tres_fundos import linhas_do_blc4
    linhas = [CABECALHO,
              f"FI;{B};AMI;2024-12-31;Brazilian Depository Receipt - BDR;BDR nível III;;;;;;;4922869;123120953.69;;"
              "2025-03-31;AURA33;;;;BRAURABDR001",
              f"FI;{A};X;2024-12-31;Ações;Ação ordinária;;;;;;;100;3000.5;;;PETR4;;;;BRPETRACNPR6",
              f"FI;{A};X;2024-12-31;Ações e outros TVM cedidos em empréstimo;Ação ordinária;;;;;;;10;300;;;VALE3;;;;x",
              f"FI;{C};Y;2024-12-31;Ações;Ação ordinária;;;;;;;1;1;;;ITUB4;;;;x"]
    tabela = linhas_do_blc4(linhas, {A, B})
    assert tabela.to_dict("records") == [
        {"cnpj": B, "month": pd.Timestamp("2024-12-31"), "ticker": "AURA33", "value": 123120953.69,
         "confid_until": pd.Timestamp("2025-03-31")},
        {"cnpj": A, "month": pd.Timestamp("2024-12-31"), "ticker": "PETR4", "value": 3000.5, "confid_until": pd.NaT},
        {"cnpj": A, "month": pd.Timestamp("2024-12-31"), "ticker": "VALE3", "value": 300.0, "confid_until": pd.NaT}]


def test_monthly_measures_annualize_by_12_and_regress_on_bova11s_excess_over_the_cdi():
    from scipy import stats
    from core.tres_fundos import medidas_mensais
    rng = np.random.default_rng(3)
    meses = pd.date_range("2022-01-01", periods=36, freq="MS")
    bova = pd.Series(rng.normal(0.01, 0.05, 36), index=meses)
    cdi = pd.Series(0.008, index=meses)
    serie = 0.004 + cdi + 1.5 * (bova - cdi) + pd.Series(rng.normal(0, 0.003, 36), index=meses)
    medidas = medidas_mensais(serie, bova, cdi)
    riqueza = (1 + serie).cumprod()
    regressao = stats.linregress(bova - cdi, serie - cdi)
    assert medidas["Meses"] == 36
    assert medidas["Retorno total"] == pytest.approx(riqueza.iloc[-1] - 1)
    assert medidas["CAGR"] == pytest.approx(riqueza.iloc[-1] ** (12 / 36) - 1)
    assert medidas["Vol"] == pytest.approx(serie.std(ddof=1) * np.sqrt(12))
    assert medidas["Sharpe"] == pytest.approx((serie - cdi).mean() * 12 / (serie.std(ddof=1) * np.sqrt(12)))
    assert medidas["Queda máx."] == pytest.approx(float((riqueza / riqueza.cummax().clip(lower=1.0) - 1).min()))
    assert medidas["Beta (BOVA11)"] == pytest.approx(regressao.slope)
    assert medidas["Alfa a.a. (BOVA11)"] == pytest.approx(regressao.intercept * 12)
    assert medidas["t do alfa"] > 2                                      # 0.4% a month with little noise


def test_the_cda_reader_also_reads_the_pre_cvm_175_header():
    from core.tres_fundos import linhas_do_blc4
    antigo = CABECALHO.replace("TP_FUNDO_CLASSE;CNPJ_FUNDO_CLASSE", "TP_FUNDO;CNPJ_FUNDO")
    tabela = linhas_do_blc4([antigo, f"FI;{A};X;2022-05-31;Ações;Ação ordinária;;;;;;;100;10;;;PETR4;;;;x"], {A})
    assert tabela[["cnpj", "ticker", "value"]].to_dict("records") == [{"cnpj": A, "ticker": "PETR4", "value": 10.0}]
