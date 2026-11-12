"""The adapter between the MT5 executor and the terminal (spec 2026-11-12-mt5-demo, plan step 2), on a fake terminal."""

from __future__ import annotations

from types import SimpleNamespace as NS

import pandas as pd
import pytest

from core.mt5_book import Order
from core.mt5_corretora import MAGIC, Corretora, TerminalError

T0 = 1_794_497_019          # the terminal's tick time: Brasília wall clock as epoch seconds (2026-11-12 15:23:39)


class FakeTerminal:
    ACCOUNT_TRADE_MODE_DEMO, ACCOUNT_TRADE_MODE_REAL = 0, 2
    TRADE_ACTION_PENDING = 5
    ORDER_TYPE_BUY_LIMIT, ORDER_TYPE_SELL_LIMIT = 2, 3
    ORDER_TIME_DAY, ORDER_FILLING_RETURN = 1, 2
    TRADE_RETCODE_PLACED, TRADE_RETCODE_DONE, TRADE_RETCODE_REJECT = 10008, 10009, 10006
    DEAL_TYPE_BUY, DEAL_TYPE_SELL = 0, 1
    POSITION_TYPE_BUY, POSITION_TYPE_SELL = 0, 1

    def __init__(self, ok: bool = True, mode: int = 0, server: str = "XPMT5-DEMO", retcode: int = 10008) -> None:
        self.ok, self.mode, self.server, self.retcode = ok, mode, server, retcode
        self.sent: list[dict] = []
        self.selected: list[str] = []
        self.closed = False
        self.positions = [NS(symbol="PETR4", volume=200.0, type=0, magic=MAGIC),
                          NS(symbol="PETR4F", volume=37.0, type=0, magic=MAGIC),
                          NS(symbol="VALE3", volume=100.0, type=0, magic=0),           # the user's own, not the book's
                          NS(symbol="ITSA4F", volume=5.0, type=1, magic=MAGIC)]
        self.deals = [NS(symbol="PETR4F", type=0, volume=37.0, price=44.7, commission=-0.5, time=T0, order=11, magic=MAGIC),
                      NS(symbol="BBSE3", type=1, volume=100.0, price=37.9, commission=0.0, time=T0, order=12, magic=MAGIC),
                      NS(symbol="VALE3", type=0, volume=100.0, price=60.0, commission=0.0, time=T0, order=13, magic=0)]

    def initialize(self, path: str, timeout: int) -> bool:
        self.path = path
        return self.ok

    def last_error(self):
        return (-6, "Terminal: Authorization failed")

    def shutdown(self) -> None:
        self.closed = True

    def account_info(self):
        return NS(trade_mode=self.mode, server=self.server, login=59803758, balance=1_000_000.0)

    def symbol_select(self, symbol: str, enable: bool) -> bool:
        self.selected.append(symbol)
        return True

    def symbol_info_tick(self, symbol: str):
        return NS(bid=44.68, ask=44.70, last=44.69, time=T0) if symbol != "NADA3" else None

    def positions_get(self):
        return tuple(self.positions)

    def order_send(self, request: dict):
        self.sent.append(request)
        return NS(retcode=self.retcode, order=99, comment="Request placed")

    def history_deals_get(self, date_from, date_to):
        self.window = (date_from, date_to)
        return tuple(self.deals)


def test_connecting_raises_with_the_terminals_error_and_always_shuts_down() -> None:
    falso = FakeTerminal(ok=False)
    with pytest.raises(TerminalError, match="Authorization failed"), Corretora(falso).conectar():
        pass
    terminal = FakeTerminal()
    with pytest.raises(RuntimeError), Corretora(terminal).conectar():
        raise RuntimeError("boom")
    assert terminal.closed


def test_the_account_is_read_as_demo_or_not() -> None:
    with Corretora(FakeTerminal()).conectar() as c:
        conta = c.conta()
    assert conta.is_demo and conta.server == "XPMT5-DEMO" and conta.balance == 1_000_000.0
    with Corretora(FakeTerminal(mode=2)).conectar() as c:
        assert not c.conta().is_demo


def test_a_quote_selects_the_symbol_and_keeps_the_wall_clock() -> None:
    terminal = FakeTerminal()
    with Corretora(terminal).conectar() as c:
        cotacao = c.cotacao("PETR4F")
        assert c.cotacao("NADA3") is None
    assert terminal.selected[0] == "PETR4F"
    assert (cotacao.bid, cotacao.ask, cotacao.last) == (44.68, 44.70, 44.69)
    assert cotacao.time == pd.Timestamp("2026-11-12 15:23:39")


def test_the_books_shares_are_its_own_positions_summed_by_name() -> None:
    with Corretora(FakeTerminal()).conectar() as c:
        acoes = c.acoes_da_carteira()
    assert acoes.to_dict() == {"PETR4": 237, "ITSA4": -5}       # VALE3 is not the book's; a sell position is negative


def test_an_order_is_a_day_limit_with_the_books_magic() -> None:
    terminal = FakeTerminal()
    with Corretora(terminal).conectar() as c:
        envio = c.enviar(Order("PETR4", "PETR4F", "buy", 37), limite=45.59, rotulo="aleph 2026-11-12")
    pedido = terminal.sent[0]
    assert envio.ok and envio.order == 99
    assert pedido["action"] == 5 and pedido["type"] == 2 and pedido["symbol"] == "PETR4F"
    assert pedido["volume"] == 37.0 and pedido["price"] == 45.59
    assert pedido["type_time"] == 1 and pedido["type_filling"] == 2 and pedido["magic"] == MAGIC
    assert pedido["comment"] == "aleph 2026-11-12"


def test_a_rejected_order_is_reported_not_raised() -> None:
    with Corretora(FakeTerminal(retcode=10006)).conectar() as c:
        envio = c.enviar(Order("PETR4", "PETR4", "sell", 100), limite=43.8, rotulo="x")
    assert not envio.ok and envio.retcode == 10006


def test_no_order_ever_reaches_an_account_that_is_not_xp_demo() -> None:
    for terminal in (FakeTerminal(mode=2), FakeTerminal(server="XPMT5-PRD")):
        with Corretora(terminal).conectar() as c, pytest.raises(TerminalError, match="not"):
            c.enviar(Order("PETR4", "PETR4", "buy", 100), limite=45.0, rotulo="x")
        assert terminal.sent == []


def test_fills_are_the_books_deals_only() -> None:
    terminal = FakeTerminal()
    with Corretora(terminal).conectar() as c:
        fills = c.execucoes(pd.Timestamp("2026-11-12"), pd.Timestamp("2026-11-13"))
    assert [(f.name, f.symbol, f.side, f.volume, f.price, f.commission) for f in fills] == [
        ("PETR4", "PETR4F", "buy", 37, 44.7, 0.5), ("BBSE3", "BBSE3", "sell", 100, 37.9, 0.0)]
    assert fills[0].time == pd.Timestamp("2026-11-12 15:23:39")
