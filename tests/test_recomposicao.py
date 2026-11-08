import pandas as pd
import pytest

from core import recomposicao as rc

ALIASES = {"CCRO3": "MOTV3", "MOTV3": "MOTV3", "MRFG3": "MBRF3", "MBRF3": "MBRF3", "PETR4": "PETR4"}


def test_tickers_become_the_lake_s_security_and_rights_and_receipts_are_dropped():
    # spec 2026-11-08, data item 1: renames mapped with the lake's securities; rights (1, 2, 12-15) and receipts
    # (9, 10) are not index members; units (11) are
    assert rc.normalizar(["CCRO3", "PETR4", "RENT1", "CASH1", "ITSA2", "GOLL14", "XPTO9", "BPAC11"], ALIASES) == {
        "MOTV3", "PETR4", "BPAC11"}


def _posicoes(linhas):
    return pd.DataFrame(linhas, columns=["cnpj", "month", "ticker"]).assign(month=lambda d: pd.to_datetime(d["month"]))


def test_a_month_takes_the_main_fund_of_its_period_else_names_two_other_index_funds_hold():
    posicoes = _posicoes([
        ("BOVA", "2018-08-31", "PETR4"), ("BOVA", "2018-08-31", "VALE3"), ("X", "2018-08-31", "ITUB4"),
        ("X", "2019-03-31", "PETR4"), ("X", "2019-03-31", "VALE3"), ("Y", "2019-03-31", "PETR4"),
        ("Y", "2019-03-31", "ABEV3"), ("Z", "2019-03-31", "PETR4"), ("Z", "2019-03-31", "VALE3")])
    principais = [("BOVA", "2008-11-30", "2018-08-31")]
    meses = rc.composicoes(posicoes, principais, ["X", "Y", "Z"], ALIASES)
    assert meses[pd.Timestamp("2018-08-31")] == {"PETR4", "VALE3"}             # the main fund, not X's ITUB4
    assert meses[pd.Timestamp("2019-03-31")] == {"PETR4", "VALE3"}             # ABEV3: one fund only


def test_an_entrant_is_absent_before_the_rebalance_and_present_in_both_month_ends_after_it():
    meses = {pd.Timestamp("2025-03-31"): {"PETR4", "AMOB3", "LWSA3"},
             pd.Timestamp("2025-04-30"): {"PETR4", "AMOB3", "LWSA3"},
             pd.Timestamp("2025-05-31"): {"PETR4", "DIRR3", "SMFT3", "LWSA3"},
             pd.Timestamp("2025-06-30"): {"PETR4", "DIRR3", "LWSA3"},           # SMFT3 missing: a partial month
             pd.Timestamp("2025-07-31"): {"PETR4", "DIRR3", "SMFT3", "LWSA3"}}   # not a rebalance month
    eventos = rc.eventos(meses)
    assert eventos.to_dict("records") == [
        {"rebalanceamento": pd.Timestamp("2025-05-01"), "papel": "AMOB3", "tipo": "saida"},
        {"rebalanceamento": pd.Timestamp("2025-05-01"), "papel": "DIRR3", "tipo": "entrada"}]


def test_previews_keep_ibovespa_pages_by_effective_month_and_number_with_tickers_as_securities():
    paginas = [
        {"url": "a", "json": {"e_previa_do_ibovespa": True, "numero_da_previa": 1, "mes_de_vigencia": "2025-05",
                              "data_de_divulgacao": "2025-04-01", "entradas": ["DIRR3"], "saidas": ["AMOB3"]}},
        {"url": "b", "json": {"e_previa_do_ibovespa": True, "numero_da_previa": 1, "mes_de_vigencia": "2025-05",
                              "data_de_divulgacao": "2025-04-01", "entradas": ["DIRR3"], "saidas": ["AMOB3"]}},
        {"url": "c", "json": {"e_previa_do_ibovespa": False, "numero_da_previa": 1, "mes_de_vigencia": "2025-05",
                              "entradas": ["XXXX3"], "saidas": []}},                          # another index
        {"url": "d", "json": {"e_previa_do_ibovespa": True, "numero_da_previa": 3, "mes_de_vigencia": "2025-05",
                              "data_de_divulgacao": "2025-04-29", "entradas": ["DIRR3", "SMFT3", "CCRO3"],
                              "saidas": []}},
        {"url": "e", "json": {"erro": 500}}]
    tabela = rc.previas(paginas, ALIASES)
    assert tabela.to_dict("records") == [
        {"rebalanceamento": pd.Timestamp("2025-05-01"), "previa": 1, "divulgacao": pd.Timestamp("2025-04-01"),
         "entradas": ("DIRR3",), "saidas": ("AMOB3",), "fonte": "a"},
        {"rebalanceamento": pd.Timestamp("2025-05-01"), "previa": 3, "divulgacao": pd.Timestamp("2025-04-29"),
         "entradas": ("DIRR3", "MOTV3", "SMFT3"), "saidas": (), "fonte": "d"}]


def test_agreement_counts_entrants_both_sources_have_and_those_only_one_has():
    previas = pd.DataFrame([
        {"rebalanceamento": pd.Timestamp("2025-05-01"), "previa": 3, "divulgacao": pd.Timestamp("2025-04-29"),
         "entradas": ("DIRR3", "SMFT3"), "saidas": ("AMOB3",), "fonte": "d"},
        {"rebalanceamento": pd.Timestamp("2025-05-01"), "previa": 1, "divulgacao": pd.Timestamp("2025-04-01"),
         "entradas": ("DIRR3",), "saidas": (), "fonte": "a"}])
    eventos = pd.DataFrame([{"rebalanceamento": pd.Timestamp("2025-05-01"), "papel": "DIRR3", "tipo": "entrada"},
                            {"rebalanceamento": pd.Timestamp("2025-05-01"), "papel": "MOTV3", "tipo": "entrada"}])
    tabela = rc.concordancia(previas, eventos, previa=3)
    assert tabela.to_dict("records") == [{"rebalanceamento": pd.Timestamp("2025-05-01"), "ambas": ("DIRR3",),
                                          "so_previa": ("SMFT3",), "so_cvm": ("MOTV3",)}]


def test_an_entrant_the_funds_bought_on_the_old_portfolio_s_last_session_still_counts():
    # 2025-09: the new portfolio started on Monday 01/09, the funds bought CEAB3 and CURY3 at the close of 29/08, so
    # the August month-end already holds them; the composition before is the month-end two months earlier (July)
    meses = {pd.Timestamp("2025-07-31"): {"PETR4", "SMTO3"},
             pd.Timestamp("2025-08-31"): {"PETR4", "CEAB3", "CURY3"},
             pd.Timestamp("2025-09-30"): {"PETR4", "CEAB3", "CURY3"},
             pd.Timestamp("2025-10-31"): {"PETR4", "CEAB3", "CURY3"}}
    eventos = rc.eventos(meses)
    assert eventos.to_dict("records") == [
        {"rebalanceamento": pd.Timestamp("2025-09-01"), "papel": "CEAB3", "tipo": "entrada"},
        {"rebalanceamento": pd.Timestamp("2025-09-01"), "papel": "CURY3", "tipo": "entrada"},
        {"rebalanceamento": pd.Timestamp("2025-09-01"), "papel": "SMTO3", "tipo": "saida"}]


def test_the_new_portfolio_starts_on_the_first_session_from_the_first_monday_and_the_funds_trade_the_day_before():
    pregoes = pd.bdate_range("2023-12-20", "2024-01-10").drop(pd.Timestamp("2024-01-01"))   # 01/01 is a holiday
    assert rc.inicio_da_carteira(pd.Timestamp("2024-01-01"), pregoes) == pd.Timestamp("2024-01-02")
    assert rc.ultimo_pregao_antigo(pd.Timestamp("2024-01-01"), pregoes) == pd.Timestamp("2023-12-29")
    setembro = pd.bdate_range("2025-08-20", "2025-09-10")
    assert rc.inicio_da_carteira(pd.Timestamp("2025-09-01"), setembro) == pd.Timestamp("2025-09-01")   # a Monday
    assert rc.ultimo_pregao_antigo(pd.Timestamp("2025-09-01"), setembro) == pd.Timestamp("2025-08-29")
    assert rc.data_da_terceira_previa(pd.Timestamp("2025-09-01"), setembro) == pd.Timestamp("2025-08-28")


def test_an_event_s_abnormal_return_is_net_of_both_sides_costs_against_bova11_over_the_same_window():
    dias = pd.bdate_range("2025-08-26", "2025-08-29")
    papel = pd.Series([10.0, 10.0, 10.5, 11.0], index=dias)
    bova = pd.Series([100.0, 100.0, 101.0, 102.0], index=dias)
    evento = rc.resultado_do_evento(papel, bova, dias[1], dias[3], custo_por_lado=0.001)
    assert evento["retorno"] == pytest.approx(1.1 * 0.999 / 1.001 - 1)
    assert evento["retorno_bova"] == pytest.approx(0.02)
    assert evento["anormal"] == pytest.approx(evento["retorno"] - 0.02)
    assert rc.resultado_do_evento(papel.iloc[2:], bova, dias[1], dias[3], 0.001) is None   # no price on the entry day


def test_the_summary_clusters_by_rebalance_date_and_reads_the_gate_and_the_three_blocks():
    eventos = pd.DataFrame({"rebalanceamento": pd.to_datetime(["2010-01-01", "2010-01-01", "2016-05-01",
                                                               "2022-09-01", "2023-01-01"]),
                            "anormal": [0.02, 0.04, -0.01, 0.03, 0.01]})
    resumo = rc.resumo(eventos)
    por_data = [0.03, -0.01, 0.03, 0.01]                                        # equal weights inside a date
    media = sum(por_data) / 4
    desvio = pd.Series(por_data).std(ddof=1)
    assert resumo["datas"] == 4 and resumo["eventos"] == 5
    assert resumo["media"] == pytest.approx(media) and resumo["t"] == pytest.approx(media / (desvio / 2))
    assert resumo["blocos"] == {"2009-2014": pytest.approx(0.03), "2015-2020": pytest.approx(-0.01),
                                "2021-2026": pytest.approx(0.02)}
    assert resumo["passa"] is False                                             # t = 1.57 < 2.24


def test_the_entry_is_b3_s_rule_date_or_a_later_found_publication_never_an_earlier_or_impossible_one():
    pregoes = pd.bdate_range("2025-07-25", "2025-09-10")
    mes = pd.Timestamp("2025-09-01")
    assert rc.data_da_primeira_previa(mes, pregoes) == pd.Timestamp("2025-08-01")
    regra = rc.data_da_terceira_previa(mes, pregoes)                                      # 28/08
    saida = rc.ultimo_pregao_antigo(mes, pregoes)                                         # 29/08
    assert rc.data_de_entrada(regra, pd.NaT, saida, pregoes) == regra
    assert rc.data_de_entrada(regra, pd.Timestamp("2025-08-20"), saida, pregoes) == regra       # earlier: the rule
    assert rc.data_de_entrada(regra, pd.Timestamp("2025-09-02"), saida, pregoes) == regra       # after the exit
    primeira = rc.data_da_primeira_previa(mes, pregoes)
    assert rc.data_de_entrada(primeira, pd.Timestamp("2025-08-03"), saida, pregoes) == pd.Timestamp("2025-08-04")
