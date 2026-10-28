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


def test_a_stock_listed_inside_the_last_year_has_no_12m_return_and_its_beta_uses_the_common_sessions():
    # review finding 6
    from core.indicadores import risco_por_acao
    datas = pd.bdate_range("2024-01-01", periods=600)
    mercado = np.random.default_rng(5).normal(0.0005, 0.01, len(datas))
    retornos = pd.DataFrame({"BOVA11.SA": mercado, "NOVA3.SA": 1.5 * mercado}, index=datas)
    precos = 10 * (1 + retornos).cumprod()
    precos.iloc[:-60, 1] = np.nan                                         # listed 60 sessions ago
    risco = risco_por_acao(precos, "BOVA11.SA")
    assert np.isnan(risco.loc["NOVA3.SA", "Retorno 12m"])
    assert risco.loc["NOVA3.SA", "Beta"] == pytest.approx(1.5)


def test_a_thinly_traded_names_beta_uses_only_sessions_where_both_traded_and_under_two_it_is_empty_silently():
    # since 2026-10-27 the panel has ~1,545 tickers, many BDRs trading on few sessions: one pass, no warnings
    import warnings
    from core.indicadores import risco_por_acao
    datas = pd.bdate_range("2024-01-01", periods=300)
    gerador = np.random.default_rng(11)
    mercado = gerador.normal(0.0, 0.01, len(datas))
    retornos = pd.DataFrame({"BOVA11.SA": mercado, "RARO34.SA": 0.8 * mercado + gerador.normal(0.0, 0.01, len(datas)),
                             "UMDIA34.SA": mercado}, index=datas)
    precos = 10 * (1 + retornos).cumprod()
    precos.iloc[::3, 1] = np.nan                                          # trades 2 sessions in 3
    precos.iloc[:-2, 2] = np.nan                                          # two prices: one return only
    precos.iloc[100, 0] = np.nan                                          # a gap in the reference too
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        risco = risco_por_acao(precos, "BOVA11.SA")
    juntos = precos.pct_change(fill_method=None).iloc[-252:][["RARO34.SA", "BOVA11.SA"]].dropna()
    assert risco.loc["RARO34.SA", "Beta"] == pytest.approx(juntos.iloc[:, 0].cov(juntos.iloc[:, 1])
                                                           / juntos.iloc[:, 1].var())
    assert np.isnan(risco.loc["UMDIA34.SA", "Beta"])


def test_the_per_stock_table_joins_name_sector_last_close_risk_and_fundamentals_by_ticker_without_the_suffix():
    from core.indicadores import tabela_por_acao
    datas = pd.bdate_range("2024-01-01", periods=300)
    gerador = np.random.default_rng(5)
    precos = pd.DataFrame(10 * (1 + gerador.normal(0, 0.01, (300, 3))).cumprod(axis=0), index=datas,
                          columns=["BOVA11.SA", "PETR4.SA", "VALE3.SA"])
    precos.iloc[-1, 2] = np.nan                                   # VALE3 did not trade on the last session
    fundamentos = pd.DataFrame({"P/L": [4.8], "Balanço de": [pd.Timestamp("2026-06-30")]}, index=["PETR4"])
    tabela = tabela_por_acao(precos, {"PETR4.SA": {"nome": "Petrobras", "setor": "Petróleo"}}, fundamentos,
                             "BOVA11.SA")
    assert list(tabela.index) == ["PETR4", "VALE3"]
    assert list(tabela.columns[:4]) == ["Empresa", "Setor", "Preço (R$)", "Fechamento de"]
    assert tabela.loc["PETR4", "Empresa"] == "Petrobras" and tabela.loc["VALE3", "Empresa"] == ""
    assert tabela.loc["VALE3", "Preço (R$)"] == pytest.approx(precos["VALE3.SA"].iloc[-2])
    assert tabela.loc["VALE3", "Fechamento de"] == datas[-2]
    assert tabela.loc["PETR4", "P/L"] == pytest.approx(4.8) and np.isnan(tabela.loc["VALE3", "P/L"])
    assert "Queda máx. (3 anos)" in tabela.columns and "Balanço de" in tabela.columns
