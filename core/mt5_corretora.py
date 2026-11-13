"""
mt5_corretora.py - The thin adapter between the MT5 executor and the MetaTrader 5 terminal
(docs/superpowers/specs/2026-11-12-mt5-demo-design.md, plan step 2).

The terminal object is the official ``MetaTrader5`` module (Windows only; it talks to the running terminal, not to
the internet), or a fake with the same names in the tests. Every order carries the book's ``MAGIC`` number, so the
book's positions and fills are told apart from anything else in the account. ``enviar`` checks the account itself
before every order: nothing reaches an account that is not XP's demo, whatever the caller did.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Protocol

import pandas as pd

from core.mt5_book import FRACTIONAL, Order, Side, check_account

TERMINAL_PATH = "C:/Program Files/MetaTrader 5 Terminal/terminal64.exe"    # forward slashes: backslashes broke the IPC
TIMEOUT_MS = 60_000
MAGIC = 51_2026          # the book's mark on its orders (attempt 51's Composite, 2026)
PRIMEIRO_TICK_TENTATIVAS, PRIMEIRO_TICK_PAUSA_S = 10, 0.2    # up to 2 s for a newly selected symbol's first tick


class Terminal(Protocol):
    """The part of the ``MetaTrader5`` module the executor uses (constants are read from it too)."""

    def initialize(self, path: str, timeout: int) -> bool: ...
    def last_error(self) -> Any: ...
    def shutdown(self) -> None: ...
    def account_info(self) -> Any: ...
    def symbol_select(self, symbol: str, enable: bool) -> bool: ...
    def symbol_info_tick(self, symbol: str) -> Any: ...
    def positions_get(self) -> Any: ...
    def order_send(self, request: dict) -> Any: ...
    def history_deals_get(self, date_from: Any, date_to: Any) -> Any: ...


class TerminalError(RuntimeError):
    """The terminal refused, or a guardrail of the adapter stopped an order."""


@dataclass(frozen=True)
class Conta:
    is_demo: bool
    server: str
    login: int
    balance: float


@dataclass(frozen=True)
class Cotacao:
    bid: float
    ask: float
    last: float
    time: pd.Timestamp          # the terminal's wall clock (Brasília), naive


@dataclass(frozen=True)
class Envio:
    ok: bool
    retcode: int
    order: int
    comment: str


@dataclass(frozen=True)
class Fill:
    name: str
    symbol: str
    side: Side
    volume: int
    price: float
    commission: float           # charged, >= 0
    time: pd.Timestamp
    order: int


def name_of(symbol: str) -> str:
    """PETR4F -> PETR4; PETR4 -> PETR4 (B3 equity tickers end in a digit; the fractional symbol adds an F)."""
    return symbol[:-1] if symbol.endswith(FRACTIONAL) else symbol


def _wall_clock(seconds: int) -> pd.Timestamp:
    return pd.Timestamp(int(seconds), unit="s")


class Corretora:
    def __init__(self, terminal: Terminal, path: str = TERMINAL_PATH,
                 esperar: Callable[[float], None] = time.sleep) -> None:
        self.t, self.path, self.esperar = terminal, path, esperar

    @contextmanager
    def conectar(self) -> Iterator[Corretora]:
        if not self.t.initialize(path=self.path, timeout=TIMEOUT_MS):
            raise TerminalError(f"terminal refused: {self.t.last_error()}")
        try:
            yield self
        finally:
            self.t.shutdown()

    def conta(self) -> Conta:
        a = self.t.account_info()
        if a is None:
            raise TerminalError(f"no account: {self.t.last_error()}")
        return Conta(a.trade_mode == self.t.ACCOUNT_TRADE_MODE_DEMO, a.server, int(a.login), float(a.balance))

    def cotacao(self, symbol: str) -> Cotacao | None:
        self.t.symbol_select(symbol, True)
        tick = self.t.symbol_info_tick(symbol)
        for _ in range(PRIMEIRO_TICK_TENTATIVAS):    # a symbol just added to the Market Watch has no tick yet (time 0)
            if tick is None or tick.time:
                break
            self.esperar(PRIMEIRO_TICK_PAUSA_S)
            tick = self.t.symbol_info_tick(symbol)
        if tick is None:
            return None
        return Cotacao(float(tick.bid), float(tick.ask), float(tick.last), _wall_clock(tick.time))

    def acoes_da_carteira(self) -> pd.Series:
        """The book's shares by name (its positions only, standard and fractional summed; a sell position is
        negative)."""
        acoes: dict[str, int] = {}
        for p in self.t.positions_get() or ():
            if p.magic != MAGIC:
                continue
            sinal = 1 if p.type == self.t.POSITION_TYPE_BUY else -1
            nome = name_of(p.symbol)
            acoes[nome] = acoes.get(nome, 0) + sinal * round(p.volume)
        return pd.Series(acoes, dtype=int)

    def enviar(self, ordem: Order, limite: float, rotulo: str) -> Envio:
        conta = self.conta()
        motivo = check_account(conta.is_demo, conta.server)
        if motivo:
            raise TerminalError(f"order not sent, {motivo}")
        tipo = self.t.ORDER_TYPE_BUY_LIMIT if ordem.side == "buy" else self.t.ORDER_TYPE_SELL_LIMIT
        pedido = {"action": self.t.TRADE_ACTION_PENDING, "symbol": ordem.symbol, "volume": float(ordem.volume),
                  "type": tipo, "price": float(limite), "type_time": self.t.ORDER_TIME_DAY,
                  "type_filling": self.t.ORDER_FILLING_RETURN, "magic": MAGIC, "comment": rotulo[:31]}
        r = self.t.order_send(pedido)
        if r is None:
            return Envio(False, -1, 0, str(self.t.last_error()))
        ok = r.retcode in (self.t.TRADE_RETCODE_PLACED, self.t.TRADE_RETCODE_DONE)
        return Envio(ok, int(r.retcode), int(r.order), str(r.comment))

    def execucoes(self, desde: pd.Timestamp, ate: pd.Timestamp) -> list[Fill]:
        """The book's fills between two wall-clock times."""
        fills = []
        for d in self.t.history_deals_get(desde.to_pydatetime(), ate.to_pydatetime()) or ():
            if d.magic != MAGIC or d.type not in (self.t.DEAL_TYPE_BUY, self.t.DEAL_TYPE_SELL):
                continue
            lado: Side = "buy" if d.type == self.t.DEAL_TYPE_BUY else "sell"
            fills.append(Fill(name_of(d.symbol), d.symbol, lado, round(d.volume), float(d.price),
                              abs(float(d.commission)), _wall_clock(d.time), int(d.order)))
        return fills
