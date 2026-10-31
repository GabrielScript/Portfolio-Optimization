"""The US stocks' indicators for the screen (``core.us_indicadores``): risk from monthly returns, the app's multiples, macro."""

from datetime import date

import numpy as np
import pandas as pd
import pytest

from core import us_indicadores as ui

MESES = pd.date_range("2023-10-01", periods=36, freq="MS")


def test_the_12_month_return_compounds_the_last_12_and_needs_all_of_them():
    retornos = pd.DataFrame({"AAA_": 0.01, "NOVA": np.nan}, index=MESES)
    retornos.iloc[-5:, 1] = 0.02
    mercado = pd.Series(0.01, index=MESES)
    risco = ui.risco_mensal(retornos, mercado)
    assert risco.loc["AAA_", "Retorno 12m"] == pytest.approx(1.01 ** 12 - 1)
    assert np.isnan(risco.loc["NOVA", "Retorno 12m"])


def test_the_volatility_is_annual_and_the_beta_reads_the_common_months():
    gerador = np.random.default_rng(3)
    mercado = pd.Series(gerador.normal(0.01, 0.04, 36), index=MESES)
    retornos = pd.DataFrame({"AAA_": 1.5 * mercado, "NOVA": 2.0 * mercado})
    retornos.iloc[:6, 1] = np.nan
    risco = ui.risco_mensal(retornos, mercado)
    assert risco.loc["AAA_", "Beta"] == pytest.approx(1.5)
    assert risco.loc["NOVA", "Beta"] == pytest.approx(2.0)
    assert risco.loc["AAA_", "Vol anual"] == pytest.approx(retornos["AAA_"].iloc[-12:].std() * 12 ** 0.5)


def test_a_short_history_has_no_beta_and_the_drawdown_reads_three_years():
    retornos = pd.DataFrame({"AAA_": 0.0, "NOVA": np.nan}, index=MESES)
    retornos.iloc[10, 0] = -0.5
    retornos.iloc[-6:, 1] = 0.01
    risco = ui.risco_mensal(retornos, pd.Series(np.linspace(-0.02, 0.02, 36), index=MESES))
    assert risco.loc["AAA_", "Queda máx. (3 anos)"] == pytest.approx(-0.5)
    assert np.isnan(risco.loc["NOVA", "Beta"])


def _gold():
    retornos = pd.DataFrame({"AAA_": 0.01, "BNK_": 0.02, "FORA": 0.03}, index=MESES)
    papeis = pd.DataFrame({"security": ["AAA_", "BNK_", "FORA"], "symbol": ["AAA", "BNK", "FORA"], "cik": [1, 2, 3],
                           "name": ["Aaa Inc", "Bnk Corp", "Fora Ltd"], "sector": ["BusEq", "Financeiro", "Shops"],
                           "source": "stockanalysis"})
    fundamentos = pd.DataFrame({
        "issuer": ["AAA_"] * 3 + ["BNK_"] * 2, "account": ["net_income", "equity", "minority_interest", "net_income", "equity"],
        "dt_refer": pd.Timestamp("2026-06-30"), "value": [10e6, 60e6, 10e6, 5e6, 50e6],
        "available_from": pd.Timestamp("2026-08-01"), "sector": ["BusEq"] * 3 + ["Financeiro"] * 2})
    mercado = pd.DataFrame({"issuer": ["AAA_", "BNK_"], "month": pd.Timestamp("2026-10-01"), "market_cap": [200e6, 100e6]})
    proventos = pd.DataFrame({"security": ["AAA_"], "ex_date": pd.Timestamp("2026-03-02"), "yield": [0.03]})
    return retornos, papeis, fundamentos, mercado, proventos


def test_the_table_holds_the_universe_with_name_cap_risk_and_the_apps_multiples():
    retornos, papeis, fundamentos, mercado, proventos = _gold()
    tabela = ui.tabela_de_indicadores(retornos, pd.Series(0.01, index=MESES), ["BNK_", "AAA_"], papeis, fundamentos, mercado,
                                      proventos, date(2026, 10, 31))
    assert tabela["Ticker"].tolist() == ["AAA", "BNK"]                  # the largest first; out of the universe stays out
    linha = tabela.set_index("Ticker").loc["AAA"]
    assert linha["Empresa"] == "Aaa Inc" and linha["Setor"] == "BusEq"
    assert linha["Valor de mercado"] == pytest.approx(200e6)
    assert linha["P/L"] == pytest.approx(20.0)
    assert linha["P/VP"] == pytest.approx(4.0)                          # equity 60 less 10 of minorities
    assert linha["DY 12m"] == pytest.approx(0.03)
    assert linha["Balanço de"] == pd.Timestamp("2026-06-30")
    banco = tabela.set_index("Ticker").loc["BNK"]
    assert banco["P/L"] == pytest.approx(20.0)                          # EDGAR's equity is a bank's equity, unlike the CVM line
    assert np.isnan(banco["Dív. líq./EBITDA"])                          # a bank has no EBITDA to speak of


def test_a_name_without_statements_stays_in_the_table_with_empty_multiples():
    retornos, papeis, fundamentos, mercado, proventos = _gold()
    tabela = ui.tabela_de_indicadores(retornos, pd.Series(0.01, index=MESES), ["FORA", "AAA_"], papeis, fundamentos, mercado,
                                      proventos, date(2026, 10, 31)).set_index("Ticker")
    assert np.isnan(tabela.loc["FORA", "P/L"]) and np.isnan(tabela.loc["FORA", "Valor de mercado"])
    assert tabela.loc["FORA", "Retorno 12m"] == pytest.approx(1.03 ** 12 - 1)


def test_the_macro_reads_each_series_last_known_month_and_the_inflation_over_12_months():
    meses = pd.date_range("2025-08-01", periods=15, freq="MS")
    macro = pd.DataFrame({"month": meses, "CPIAUCSL": np.r_[np.full(14, 100.0), np.nan], "DGS3MO": 4.0, "DGS10": 5.0,
                          "BAA10Y": 1.5, "VIXCLS": 17.0, "UNRATE": np.r_[np.full(14, 4.0), np.nan], "NFCI": -0.5,
                          "USREC": np.r_[np.zeros(14), np.nan]})
    macro.loc[13, "CPIAUCSL"] = 103.0
    fatores = pd.DataFrame({"month": meses, "mkt_rf": 0.01, "smb": 0.0, "hml": 0.0, "rmw": 0.0, "cma": 0.0, "rf": 0.002,
                            "mom": 0.0})
    tabela = ui.tabela_macro(macro, fatores).set_index("Indicador")
    assert tabela.loc["Inflação em 12 meses (CPI)", "Valor"] == pytest.approx(3.0)
    assert tabela.loc["Inflação em 12 meses (CPI)", "Mês"] == pd.Timestamp("2026-09-01")
    assert tabela.loc["Juro de 10 anos menos 3 meses", "Valor"] == pytest.approx(1.0)
    assert tabela.loc["Juro de 3 meses", "Mês"] == pd.Timestamp("2026-10-01")
    assert tabela.loc["Mercado em 12 meses (French)", "Valor"] == pytest.approx((1.012 ** 12 - 1) * 100)
    assert tabela["Unidade"].notna().all()
