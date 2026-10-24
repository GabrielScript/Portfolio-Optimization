"""Classic value strategies for the comparison with the Conservadora: Carlisle's Acquirer's Multiple and Piotroski's
F-score, on the lake's point-in-time CVM statements."""

import pandas as pd
import pytest

from core.classicas import (balancos_em_tres_anos, f_score, multiplo_do_adquirente, multiplo_do_adquirente_no_mes,
                            piotroski_no_mes)
from core.valor import Balancos


def _fundamentos(*linhas):
    return pd.DataFrame(linhas, columns=["issuer", "account", "dt_refer", "available_from", "value"]).astype(
        {"dt_refer": "datetime64[ns]", "available_from": "datetime64[ns]"})


def _balanco(**contas_por_emissor):
    """Issuer x account, zero where an example leaves an EV account out."""
    contas = ["ebit", "cash", "short_term_investments", "short_term_debt", "long_term_debt", "minority_interest"]
    balanco = pd.DataFrame(contas_por_emissor).T
    return balanco.reindex(columns=list(dict.fromkeys(contas + balanco.columns.tolist()))).fillna(0.0)


def test_multiplo_do_adquirente_ranks_the_lowest_ev_over_ebit_first():
    # market value 100 each; EV = 100 + debt - cash
    #        EBIT   EV    EV/EBIT
    #   A     10   150     15      (debt 50)
    #   B     10    80      8      (cash 20): the cheapest
    #   C     20   100      5      cheaper still
    #   D     -5   100      -      operating loss: last
    #   E     10    -10     -      EV <= 0: out (the multiple means nothing)
    #   F     50   100      2      a bank: out (other chart of accounts)
    balanco = _balanco(A={"ebit": 10.0, "long_term_debt": 50.0}, B={"ebit": 10.0, "cash": 20.0}, C={"ebit": 20.0},
                       D={"ebit": -5.0}, E={"ebit": 10.0, "cash": 110.0}, F={"ebit": 50.0})
    setores = {**dict.fromkeys("ABCDE", "Bens Industriais"), "F": "Financeiro"}
    nota = multiplo_do_adquirente(balanco, pd.Series(100.0, index=list("ABCDEF")), setores)
    assert nota.sort_values(ascending=False).index.tolist() == ["C", "B", "A", "D"]


def test_three_years_of_balances_take_the_same_quarter_and_only_documents_already_received():
    fundamentos = _fundamentos(
        ("ACME", "total_assets", "2023-06-30", "2023-08-10", 80.0),
        ("ACME", "total_assets", "2024-06-30", "2024-08-10", 90.0),
        ("ACME", "total_assets", "2024-12-31", "2025-03-20", 95.0),
        ("ACME", "total_assets", "2025-06-30", "2025-08-10", 100.0),
        # BETA's year-earlier quarter was re-filed after the decision: not known then, so BETA has no prior year
        ("BETA", "total_assets", "2024-06-30", "2025-10-01", 50.0),
        ("BETA", "total_assets", "2025-06-30", "2025-08-12", 60.0))
    atual, um_ano, dois_anos = balancos_em_tres_anos(fundamentos, pd.Timestamp("2025-09-01"))
    assert (atual.loc["ACME", "total_assets"], um_ano.loc["ACME", "total_assets"],
            dois_anos.loc["ACME", "total_assets"]) == (100.0, 90.0, 80.0)
    assert atual.loc["BETA", "total_assets"] == 60.0 and "BETA" not in um_ano.index
    # in August the June quarter has not arrived: the year-end 2024 is the latest, and 2023's year-end is missing
    atual, um_ano, _ = balancos_em_tres_anos(fundamentos, pd.Timestamp("2025-08-01"))
    assert atual.loc["ACME", "total_assets"] == 95.0 and "ACME" not in um_ano.index


# A firm with all nine signals: ROA 10/95 > 8/90, CFO 15 > NI 10, leverage 20/100 < 22/95, current ratio 50/25 >
# 45/25, gross margin 48/120 > 42/110, asset turnover 120/95 > 110/90, no new shares.
BASE_ATUAL = {"net_income": 10.0, "operating_cash_flow": 15.0, "total_assets": 100.0, "long_term_debt": 20.0,
              "current_assets": 50.0, "current_liabilities": 25.0, "revenue": 120.0, "gross_profit": 48.0}
BASE_UM_ANO = {"net_income": 8.0, "operating_cash_flow": 12.0, "total_assets": 95.0, "long_term_debt": 22.0,
               "current_assets": 45.0, "current_liabilities": 25.0, "revenue": 110.0, "gross_profit": 42.0}
BASE_DOIS_ANOS = {"total_assets": 90.0}


def test_f_score_adds_the_nine_binary_signals_with_ratios_over_lagged_assets():
    firmas = {  # issuer: (changes now, changes a year earlier, changes two years earlier, shares now)
        "BASE": ({}, {}, {}, 100.0),
        "NEGL": ({"net_income": -1.0}, {}, {}, 100.0),          # loss: NI and ROA change off (CFO > NI holds)
        "CAIX": ({"operating_cash_flow": -5.0}, {}, {}, 100.0),  # cash burn: CFO and accrual off
        "ROAD": ({}, {"net_income": 12.0}, {}, 100.0),           # ROA 10/95 < 12/90
        "ACCR": ({"operating_cash_flow": 9.0}, {}, {}, 100.0),   # CFO < NI
        "LEVE": ({"long_term_debt": 30.0}, {}, {}, 100.0),       # leverage 0.30 > 0.23
        "LIQU": ({"current_liabilities": 30.0}, {}, {}, 100.0),  # current ratio 1.67 < 1.80
        "EMIS": ({}, {}, {}, 110.0),                             # issued shares
        "MINI": ({}, {}, {}, 100.05),                            # rounding of split-adjusted counts: no issue
        "MARG": ({"gross_profit": 45.0}, {}, {}, 100.0),         # gross margin 0.375 < 0.382
        "GIRO": ({}, {"revenue": 115.0, "gross_profit": 43.9}, {}, 100.0),  # turnover 120/95 < 115/90
        # lagged assets matter: 8/70 > 10/95 and 110/70 > 120/95 (with current assets both would hold)
        "DEFA": ({}, {}, {"total_assets": 70.0}, 100.0),
    }
    atual = pd.DataFrame({e: {**BASE_ATUAL, **f[0]} for e, f in firmas.items()}).T
    um_ano = pd.DataFrame({e: {**BASE_UM_ANO, **f[1]} for e, f in firmas.items()}).T
    dois_anos = pd.DataFrame({e: {**BASE_DOIS_ANOS, **f[2]} for e, f in firmas.items()}).T
    acoes = pd.Series({e: f[3] for e, f in firmas.items()})
    nota = f_score(atual, um_ano, dois_anos, acoes, pd.Series(100.0, index=acoes.index))
    assert nota.to_dict() == {"BASE": 9, "NEGL": 7, "CAIX": 7, "ROAD": 8, "ACCR": 8, "LEVE": 8, "LIQU": 8,
                              "EMIS": 8, "MINI": 9, "MARG": 8, "GIRO": 8, "DEFA": 7}


def test_f_score_leaves_out_a_firm_with_a_missing_input_instead_of_scoring_it_zero():
    # no gross result (a bank-like statement), no year-before-last, no share count: unknown, not weak
    atual = pd.DataFrame({"BASE": BASE_ATUAL, "SEMB": {k: v for k, v in BASE_ATUAL.items() if k != "gross_profit"},
                          "SEMD": BASE_ATUAL, "SEMA": BASE_ATUAL}).T
    um_ano = pd.DataFrame(dict.fromkeys(atual.index, BASE_UM_ANO)).T
    dois_anos = pd.DataFrame({"BASE": BASE_DOIS_ANOS, "SEMB": BASE_DOIS_ANOS, "SEMA": BASE_DOIS_ANOS}).T
    acoes = pd.Series({"BASE": 100.0, "SEMB": 100.0, "SEMD": 100.0})
    assert f_score(atual, um_ano, dois_anos, acoes, acoes).to_dict() == {"BASE": 9}


def _documento(emissor: str, contas: dict, dt_refer: str, available_from: str) -> list[tuple]:
    return [(emissor, conta, dt_refer, available_from, valor) for conta, valor in contas.items()]


def test_piotroski_picks_the_highest_f_score_and_breaks_ties_by_book_to_market():
    # market value 100 each. AAAA and BBBB score 9, CCCC 8 (cash flow below the profit) with the highest
    # book-to-market; the bank and the negative-equity firm also score 9 but are out
    firmas = {"AAAA": ({"equity": 50.0}, "Bens Industriais"), "BBBB": ({"equity": 80.0}, "Bens Industriais"),
              "CCCC": ({"equity": 200.0, "operating_cash_flow": 9.0}, "Consumo Cíclico"),
              "BANC": ({"equity": 90.0}, "Financeiro"), "NEGA": ({"equity": -10.0}, "Saúde")}
    linhas = []
    for emissor, (mudancas, _) in firmas.items():
        linhas += _documento(emissor, {**BASE_ATUAL, **mudancas}, "2024-12-31", "2025-03-20")
        linhas += _documento(emissor, BASE_UM_ANO, "2023-12-31", "2024-03-20")
        linhas += _documento(emissor, BASE_DOIS_ANOS, "2022-12-31", "2023-03-20")
    mes, ano_antes = pd.Timestamp("2025-06-01"), pd.Timestamp("2024-06-01")
    valor_mercado = pd.DataFrame([{"issuer": e, "month": m, "market_cap": 100.0, "shares_today_units": 100.0}
                                  for e in firmas for m in (mes, ano_antes)])
    balancos = Balancos(_fundamentos(*linhas), valor_mercado, {e: setor for e, (_, setor) in firmas.items()})
    nota = piotroski_no_mes(balancos, mes, ["AAAA3", "BBBB3", "CCCC3", "BANC4", "NEGA3"])
    assert nota.sort_values(ascending=False).index.tolist() == ["BBBB3", "AAAA3", "CCCC3"]


def test_acquirers_multiple_of_a_month_uses_documents_received_before_it_on_each_class_of_the_issuer():
    # ACME's cheap quarter arrives on 10/05: in May it still has the year-end EBIT of 5 and BETA (EBIT 8) leads
    linhas = (_documento("ACME", {"ebit": 5.0}, "2024-12-31", "2025-03-20")
              + _documento("ACME", {"ebit": 30.0}, "2025-03-31", "2025-05-10")
              + _documento("BETA", {"ebit": 8.0}, "2024-12-31", "2025-03-20"))
    valor_mercado = pd.DataFrame([{"issuer": e, "month": pd.Timestamp(m), "market_cap": 100.0}
                                  for e in ("ACME", "BETA") for m in ("2025-05-01", "2025-06-01")])
    balancos = Balancos(_fundamentos(*linhas), valor_mercado, dict.fromkeys(["ACME", "BETA"], "Bens Industriais"))
    candidatos = ["ACME3", "ACME4", "BETA3", "SEMN3"]
    maio = multiplo_do_adquirente_no_mes(balancos, pd.Timestamp("2025-05-01"), candidatos)
    junho = multiplo_do_adquirente_no_mes(balancos, pd.Timestamp("2025-06-01"), candidatos)
    assert maio.idxmax() == "BETA3" and junho.idxmax() in ("ACME3", "ACME4")
    assert maio["ACME3"] == maio["ACME4"] and "SEMN3" not in maio.index
