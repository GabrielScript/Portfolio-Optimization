import pandas as pd

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
    meses = {pd.Timestamp("2025-04-30"): {"PETR4", "AMOB3", "LWSA3"},
             pd.Timestamp("2025-05-31"): {"PETR4", "DIRR3", "SMFT3", "LWSA3"},
             pd.Timestamp("2025-06-30"): {"PETR4", "DIRR3", "LWSA3"},           # SMFT3 missing: a partial month
             pd.Timestamp("2025-07-31"): {"PETR4", "DIRR3", "SMFT3", "LWSA3"}}   # not a rebalance month
    eventos = rc.eventos(meses)
    assert eventos.to_dict("records") == [
        {"rebalanceamento": pd.Timestamp("2025-05-01"), "papel": "AMOB3", "tipo": "saida"},
        {"rebalanceamento": pd.Timestamp("2025-05-01"), "papel": "DIRR3", "tipo": "entrada"}]
