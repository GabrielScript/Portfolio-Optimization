"""The screen's per-stock table (``core.indicadores``): valuation from the lake's CVM statements and risk from the
prices. Information only."""

import numpy as np
import pandas as pd
import pytest

from core.indicadores import multiplos


def test_the_multiples_come_from_the_last_known_statement_and_the_market_value():
    fundamentos = pd.DataFrame({"net_income_parent": [10.0, -5.0], "equity": [50.0, 40.0], "revenue": [100.0, 80.0],
                                "minority_interest": [0.0, 0.0], "ebit": [20.0, 5.0], "depreciation": [-5.0, -1.0],
                                "short_term_debt": [10.0, 0.0], "long_term_debt": [30.0, 0.0], "cash": [15.0, 2.0]},
                               index=["PETR", "MGLU"])
    tabela = multiplos(fundamentos, pd.Series({"PETR": 100.0, "MGLU": 20.0}))
    assert tabela.loc["PETR", "P/L"] == pytest.approx(10.0)
    assert tabela.loc["PETR", "P/VP"] == pytest.approx(2.0)
    assert tabela.loc["PETR", "ROE"] == pytest.approx(0.2)
    assert tabela.loc["PETR", "Margem líquida"] == pytest.approx(0.1)
    assert tabela.loc["PETR", "Dív. líq./EBITDA"] == pytest.approx((40.0 - 15.0) / 25.0)
    assert np.isnan(tabela.loc["MGLU", "P/L"])            # loss: no P/L, the ROE shows it
    assert tabela.loc["MGLU", "ROE"] == pytest.approx(-5.0 / 40.0)


def test_the_risk_measures_are_last_year_return_volatility_beta_and_the_windows_drawdown():
    from core.indicadores import risco_por_acao
    datas = pd.bdate_range("2024-01-01", periods=600)
    gerador = np.random.default_rng(3)
    mercado = gerador.normal(0.0005, 0.01, len(datas))
    retornos = pd.DataFrame({"BOVA11.SA": mercado, "PETR4.SA": 2.0 * mercado}, index=datas)
    precos = 10 * (1 + retornos).cumprod()
    risco = risco_por_acao(precos, "BOVA11.SA")
    ultimo_ano = precos.iloc[-253:]
    assert risco.loc["PETR4.SA", "Retorno 12m"] == pytest.approx(ultimo_ano["PETR4.SA"].iloc[-1] /
                                                                  ultimo_ano["PETR4.SA"].iloc[0] - 1)
    assert risco.loc["PETR4.SA", "Beta"] == pytest.approx(2.0)
    assert risco.loc["PETR4.SA", "Vol anual"] == pytest.approx(retornos["PETR4.SA"].iloc[-252:].std() * np.sqrt(252))
    assert risco.loc["PETR4.SA", "Queda máx."] == pytest.approx((precos["PETR4.SA"] / precos["PETR4.SA"].cummax()
                                                                 - 1).min())


def test_the_fundamentals_table_uses_statements_received_by_today_the_last_market_value_and_12m_payouts():
    from datetime import date
    from core.indicadores import tabela_de_fundamentos
    fundamentos = pd.DataFrame({
        "issuer": ["PETR"] * 4, "account": ["net_income_parent", "equity", "net_income_parent", "equity"],
        "dt_refer": pd.to_datetime(["2026-03-31", "2026-03-31", "2026-06-30", "2026-06-30"]),
        "value": [8.0, 50.0, 10.0, 50.0],
        "available_from": pd.to_datetime(["2026-05-10", "2026-05-10", "2026-08-06", "2026-08-06"])})
    mercado = pd.DataFrame({"issuer": ["PETR", "PETR"], "month": pd.to_datetime(["2026-09-01", "2026-10-01"]),
                            "market_cap": [90.0, 100.0]})
    proventos = pd.DataFrame({"security": ["PETR4", "PETR4", "PETR4"],
                              "ex_date": pd.to_datetime(["2025-09-01", "2026-03-02", "2026-08-21"]),
                              "yield": [0.05, 0.04, 0.03]})
    tabela = tabela_de_fundamentos(fundamentos, mercado, proventos, ["PETR4.SA", "ITUB4.SA"], date(2026, 10, 23))
    assert tabela.loc["PETR4", "P/L"] == pytest.approx(10.0)              # June statement, October market value
    assert tabela.loc["PETR4", "DY 12m"] == pytest.approx(0.07)
    assert tabela.loc["PETR4", "Balanço de"] == pd.Timestamp("2026-06-30")
    assert "ITUB4" not in tabela.index                                     # no statement in the data given


def test_banks_and_insurers_get_no_multiples_because_their_equity_line_is_not_the_equity():
    from datetime import date
    from core.indicadores import tabela_de_fundamentos
    fundamentos = pd.DataFrame({"issuer": ["ITUB", "ITUB"], "account": ["net_income_parent", "equity"],
                                "dt_refer": pd.to_datetime(["2026-06-30"] * 2), "value": [40.0, 2000.0],
                                "available_from": pd.to_datetime(["2026-08-06"] * 2),
                                "sector": ["Financeiro", "Financeiro"]})
    mercado = pd.DataFrame({"issuer": ["ITUB"], "month": pd.to_datetime(["2026-10-01"]), "market_cap": [400.0]})
    proventos = pd.DataFrame({"security": ["ITUB4"], "ex_date": pd.to_datetime(["2026-08-01"]), "yield": [0.07]})
    linha = tabela_de_fundamentos(fundamentos, mercado, proventos, ["ITUB4.SA"], date(2026, 10, 23)).loc["ITUB4"]
    assert np.isnan(linha["P/VP"]) and np.isnan(linha["ROE"]) and np.isnan(linha["P/L"])
    assert linha["DY 12m"] == pytest.approx(0.07)
