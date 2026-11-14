"""The MT5 executor's run (spec 2026-11-12-mt5-demo, plan step 3), on a fake terminal and temporary files."""

from __future__ import annotations

from datetime import datetime
from types import SimpleNamespace as NS

import pandas as pd
import pytest

from core.mt5_corretora import MAGIC
from scripts import mt5_executor as ex

DIA = datetime(2026, 11, 13)
NA_CHAMADA = DIA.replace(hour=17, minute=56)
PRECOS = {"AAAA3": 10.0, "BBBB3": 20.0, "CCCC3": 5.0, "DDDD3": 8.0, "EEEE3": 40.0}


class Terminal:
    ACCOUNT_TRADE_MODE_DEMO = 0
    TRADE_ACTION_PENDING, ORDER_TYPE_BUY_LIMIT, ORDER_TYPE_SELL_LIMIT = 5, 2, 3
    ORDER_TIME_DAY, ORDER_FILLING_RETURN = 1, 2
    TRADE_RETCODE_PLACED, TRADE_RETCODE_DONE = 10008, 10009
    DEAL_TYPE_BUY, DEAL_TYPE_SELL, POSITION_TYPE_BUY = 0, 1, 0

    def __init__(self, agora: datetime, mode: int = 0, positions=(), deals=(), zero: str = "") -> None:
        self.agora, self.mode, self.zero = agora, mode, zero
        self.positions, self.deals, self.sent = list(positions), list(deals), []

    def initialize(self, path, timeout):
        return True

    def last_error(self):
        return (0, "")

    def shutdown(self):
        pass

    def account_info(self):
        return NS(trade_mode=self.mode, server="XPMT5-DEMO", login=1, balance=1e6)

    def symbol_select(self, symbol, enable):
        return True

    def symbol_info_tick(self, symbol):
        nome = symbol.removesuffix("F")
        preco = 0.0 if nome == self.zero else PRECOS[nome]
        return NS(bid=preco, ask=preco, last=preco, time=int(pd.Timestamp(self.agora).timestamp()))

    def positions_get(self):
        return tuple(self.positions)

    def order_send(self, request):
        self.sent.append(request)
        return NS(retcode=10008, order=len(self.sent), comment="placed")

    def history_deals_get(self, date_from, date_to):
        return tuple(self.deals)


@pytest.fixture
def pastas(tmp_path):
    registro = tmp_path / "composto.csv"
    linhas = [("2026-10-27", "AAAA3", 0.6, 9.0), ("2026-10-27", "BBBB3", 0.4, 19.0),
              ("2026-11-12", "AAAA3", 0.2, 10.0), ("2026-11-12", "BBBB3", 0.2, 20.0), ("2026-11-12", "CCCC3", 0.2, 5.0),
              ("2026-11-12", "DDDD3", 0.2, 8.0), ("2026-11-12", "EEEE3", 0.2, 40.0)]
    pd.DataFrame(linhas, columns=["data", "ticker", "peso", "preco"]).to_csv(registro, index=False)
    return ex.Pastas(registro=registro, livro=tmp_path / "paper-mt5", runs=tmp_path / "runs.csv")


def rodar(pastas, terminal, agora=NA_CHAMADA, simular=False, esperas=None):
    return ex.rodar(terminal, pastas, lambda: agora, (esperas if esperas is not None else []).append, simular=simular)


def test_a_new_record_is_bought_in_the_closing_call_with_the_books_100k(pastas) -> None:
    terminal = Terminal(NA_CHAMADA)
    r = rodar(pastas, terminal)
    assert r.desfecho == "enviado" and r.ordens == len(terminal.sent) > 0
    comprado = sum(p["volume"] * PRECOS[p["symbol"].removesuffix("F")] for p in terminal.sent)
    assert 99_000 < comprado <= 100_000
    assert {p["symbol"].removesuffix("F") for p in terminal.sent} == set(PRECOS)
    assert all(p["magic"] == MAGIC and p["comment"] == "aleph 2026-11-12" for p in terminal.sent)
    assert all(p["price"] == pytest.approx(PRECOS[p["symbol"].removesuffix("F")] * 1.005, abs=0.01) for p in terminal.sent)
    alvos = pd.read_csv(pastas.livro / "alvos.csv")
    assert set(alvos["registro"]) == {"2026-11-12"} and len(alvos) == 5


def test_the_same_record_is_not_traded_twice_the_same_day(pastas) -> None:
    rodar(pastas, Terminal(NA_CHAMADA))
    terminal = Terminal(NA_CHAMADA)
    assert rodar(pastas, terminal).desfecho == "nada" and terminal.sent == []


def test_before_the_call_it_waits_and_after_it_sends_nothing(pastas) -> None:
    tarde = Terminal(DIA.replace(hour=18, minute=5))
    r = rodar(pastas, tarde, agora=DIA.replace(hour=18, minute=5))
    assert r.desfecho == "fora da chamada" and tarde.sent == []
    esperas: list = []
    rodar(pastas, Terminal(DIA.replace(hour=17, minute=50)), agora=DIA.replace(hour=17, minute=50), esperas=esperas)
    assert esperas == [DIA.replace(hour=17, minute=55, second=15)]


def test_the_stop_file_and_a_real_account_send_nothing(pastas) -> None:
    pastas.livro.mkdir(parents=True)
    (pastas.livro / "PARAR").touch()
    terminal = Terminal(NA_CHAMADA)
    assert rodar(pastas, terminal).desfecho == "parado" and terminal.sent == []
    (pastas.livro / "PARAR").unlink()
    real = Terminal(NA_CHAMADA, mode=2)
    assert rodar(pastas, real).desfecho == "recusado" and real.sent == []


def test_a_record_name_without_a_quote_refuses_the_run_and_it_is_due_again(pastas) -> None:
    terminal = Terminal(NA_CHAMADA, zero="CCCC3")
    r = rodar(pastas, terminal)
    assert r.desfecho == "recusado" and "CCCC3" in r.nota and terminal.sent == []
    assert rodar(pastas, Terminal(NA_CHAMADA)).desfecho == "enviado"


def test_the_next_session_resends_only_what_did_not_fill_and_never_a_third_time(pastas) -> None:
    rodar(pastas, Terminal(NA_CHAMADA))
    alvos = pd.read_csv(pastas.livro / "alvos.csv").set_index("nome")["acoes"]
    cheio = [NS(symbol=n, volume=float(q), type=0, magic=MAGIC) for n, q in alvos.items() if n != "EEEE3"]
    deals = [NS(symbol=n, type=0, volume=float(q), price=PRECOS[n], commission=0.0,
                time=int(pd.Timestamp(NA_CHAMADA).timestamp()), order=1, magic=MAGIC) for n, q in alvos.items()
             if n != "EEEE3"]
    amanha = NA_CHAMADA.replace(day=16)
    terminal = Terminal(amanha, positions=cheio, deals=deals)
    r = rodar(pastas, terminal, agora=amanha)
    assert r.desfecho == "enviado"
    assert {p["symbol"].removesuffix("F") for p in terminal.sent} == {"EEEE3"}
    depois = NA_CHAMADA.replace(day=17)
    assert rodar(pastas, Terminal(depois, positions=cheio, deals=deals), agora=depois).desfecho == "nada"


def test_a_dry_run_prints_the_orders_and_writes_nothing(pastas) -> None:
    terminal = Terminal(DIA.replace(hour=11))
    r = rodar(pastas, terminal, agora=DIA.replace(hour=11), simular=True)
    assert r.desfecho == "simulado" and r.ordens > 0 and terminal.sent == []
    assert not (pastas.livro / "alvos.csv").exists() and not (pastas.livro / "execucoes.csv").exists()


def test_the_books_cash_comes_from_its_fills() -> None:
    fills = pd.DataFrame({"side": ["buy", "sell"], "volume": [100, 50], "price": [10.0, 12.0], "commission": [1.0, 0.5]})
    assert ex.caixa(fills) == pytest.approx(100_000 - 1_000 - 1.0 + 600 - 0.5)


def test_options_ignore_the_launchers_own_arguments(monkeypatch) -> None:
    monkeypatch.setattr("sys.argv", ["scripts/rodar_paper.py", "mt5_executor"])     # how the Windows task calls it
    opcoes = ex.ler_opcoes(None)
    assert not opcoes.simular and not opcoes.leitura
    assert ex.ler_opcoes(["--simular"]).simular
