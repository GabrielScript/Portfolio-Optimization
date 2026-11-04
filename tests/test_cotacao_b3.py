"""The plan's quotes from B3's own quote service (the user's UI list of 2026-11-03: no Yahoo)."""

import json

import pandas as pd

from core import market_data
from core.cotacao_b3 import FONTE_B3, cotacoes_b3, ler_historico_do_dia

RESPOSTA = json.dumps({"BizSts": {"cd": "OK"}, "TradgFlr": {"date": "2026-11-03", "scty": {"lstQtn": [
    {"closPric": 50.62, "dtTm": "17:54:00"}, {"closPric": 50.71, "dtTm": "18:05:00"}]}}})
DIARIOS = pd.DataFrame({"PETR4.SA": [52.2, 52.71], "VALE3.SA": [60.0, 61.5]},
                       index=pd.to_datetime(["2026-10-30", "2026-10-31"]))


def test_the_days_history_gives_the_last_trade_and_its_session_time():
    # PETR4 on 03/11/2026: the closing auction's print (18:05) is the COTAHIST close, 50.71
    assert ler_historico_do_dia(RESPOSTA) == (50.71, pd.Timestamp("2026-11-03 18:05"))


def test_an_answer_without_trades_or_with_an_error_status_reads_as_none():
    assert ler_historico_do_dia(json.dumps({"BizSts": {"cd": "NOK"}})) is None
    assert ler_historico_do_dia(json.dumps({"BizSts": {"cd": "OK"}, "TradgFlr": {"date": "2026-11-03",
                                                                                  "scty": {"lstQtn": []}}})) is None
    assert ler_historico_do_dia("<html>erro</html>") is None


def test_each_ticker_gets_b3s_last_trade_or_else_the_lakes_last_close():
    def baixar(ticker: str) -> str:
        if ticker == "PETR4":
            return RESPOSTA
        raise OSError("timeout")

    cotacoes = cotacoes_b3(["PETR4.SA", "VALE3.SA", "XPTO3.SA"], DIARIOS, baixar)
    assert cotacoes.loc["PETR4.SA"].tolist() == [50.71, pd.Timestamp("2026-11-03 18:05"), FONTE_B3]
    assert cotacoes.loc["VALE3.SA"].tolist() == [61.5, pd.Timestamp("2026-10-31"), market_data.FONTE_FECHAMENTO]
    assert pd.isna(cotacoes.loc["XPTO3.SA", "preco"])
