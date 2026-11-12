"""
mt5_book.py - The pure part of the MT5 executor (docs/superpowers/specs/2026-11-12-mt5-demo-design.md): the book's
target in whole shares, the orders that take it there, the guardrails that refuse a run, and the book's cash. No
``MetaTrader5`` import here: the terminal is reached only through ``core.mt5_corretora``.
"""

from __future__ import annotations

import math
from collections.abc import Collection, Iterable, Sequence
from dataclasses import dataclass
from typing import Literal

import pandas as pd

from core import simulador
from core.ordens import RegrasCompra, plano_de_compra
from core.risco import LIMITES

Side = Literal["buy", "sell"]
LOT = 100                       # B3's standard lot; the rest goes to the fractional market
FRACTIONAL = "F"                # suffix of the fractional symbol
BAND = 0.02                     # limit price band around the last price in the closing call
TICK = 0.01
SERVER = "XPMT5-DEMO"
MAX_QUOTE_AGE = pd.Timedelta(minutes=5)
SMALL_BOOK, FEE_PER_ORDER = 1_000.0, 4.90   # the user's real size and XP's per-order fee (to be confirmed)


@dataclass(frozen=True)
class Order:
    name: str                   # the stock (e.g. PETR4)
    symbol: str                 # where it is sent (PETR4 or PETR4F)
    side: Side
    volume: int


def target_shares(weights: pd.Series, quotes: pd.Series, book_value: float) -> pd.Series:
    """Whole shares of each name for a book of ``book_value``, by the house plan (``core.ordens.plano_de_compra``:
    the cost reserve, at most 20% a name)."""
    regras = RegrasCompra(orcamento=book_value, custo_bps=simulador.CUSTO_BPS, peso_maximo=LIMITES.peso_maximo_acao,
                          quantidade_minima=1)
    plano = plano_de_compra(weights, quotes.reindex(weights.index), regras)
    return plano.itens["quantidade"].astype(int)


def _split(name: str, side: Side, volume: int) -> list[Order]:
    lotes, resto = divmod(volume, LOT)
    return ([Order(name, name, side, lotes * LOT)] if lotes else []) + \
           ([Order(name, name + FRACTIONAL, side, resto)] if resto else [])


def orders_for(target: pd.Series, held: pd.Series) -> list[Order]:
    """The orders from ``held`` to ``target`` (shares by name, a name absent = 0): sells first, each quantity split
    into lots of 100 on the standard symbol and the rest on the fractional one."""
    nomes = sorted(set(target.index) | set(held.index))
    diferenca = target.reindex(nomes, fill_value=0).astype(int) - held.reindex(nomes, fill_value=0).astype(int)
    vendas = [o for nome, d in diferenca.items() if d < 0 for o in _split(nome, "sell", -int(d))]
    compras = [o for nome, d in diferenca.items() if d > 0 for o in _split(nome, "buy", int(d))]
    return vendas + compras


def limit_price(side: Side, last: float, band: float = BAND, tick: float = TICK) -> float:
    """The closing-call limit: ``band`` above the last price for a buy, below for a sell, on the tick (rounded away
    from the last price so the band is never narrower than asked)."""
    if not last > 0:
        raise ValueError(f"last price must be > 0, got {last}")
    bruto = last * (1 + band) if side == "buy" else last * (1 - band)
    passos = bruto / tick
    passos = math.ceil(passos - 1e-9) if side == "buy" else math.floor(passos + 1e-9)
    return round(passos * tick, 2)


def check_account(is_demo: bool, server: str) -> str | None:
    """None when the run may trade this account; else why not."""
    if not is_demo:
        return "the account is not DEMO: the executor trades the demo account only"
    if server != SERVER:
        return f"server {server!r} is not {SERVER!r}"
    return None


def check_names(orders: Iterable[Order], allowed: Collection[str]) -> str | None:
    fora = sorted({o.name for o in orders} - set(allowed))
    return f"orders on names outside the book: {fora}" if fora else None


def check_cash(orders: Sequence[Order], quotes: pd.Series, cash: float) -> str | None:
    """The buys at the quotes must fit in the book's cash plus its sells (all fill in the same closing call)."""
    valor = {lado: sum(o.volume * float(quotes[o.name]) for o in orders if o.side == lado) for lado in ("buy", "sell")}
    if valor["buy"] > cash + valor["sell"] + 1e-6:
        return f"buys {valor['buy']:.2f} exceed the book's cash {cash:.2f} plus sells {valor['sell']:.2f}"
    return None


def check_quote(symbol: str, bid: float, ask: float, quote_time: pd.Timestamp, now: pd.Timestamp,
                max_age: pd.Timedelta = MAX_QUOTE_AGE) -> str | None:
    if not (bid > 0 and ask > 0):
        return f"{symbol}: zero quote (bid {bid}, ask {ask})"
    if now - quote_time > max_age:
        return f"{symbol}: quote too old ({quote_time:%H:%M:%S} at {now:%H:%M:%S})"
    return None


def cash_after(cash: float, fills: Iterable[tuple[Side, int, float, float]]) -> float:
    """The book's cash after its fills (side, volume, price, commission charged)."""
    for lado, volume, preco, comissao in fills:
        cash += (volume * preco if lado == "sell" else -volume * preco) - comissao
    return cash


def cost_at_small_book_bps(order_value: float, book_value: float, small_book: float = SMALL_BOOK,
                           fee: float = FEE_PER_ORDER) -> float:
    """What the per-order fee would cost, in bps of the order, if the same order were placed on a book of
    ``small_book`` (the user's real size)."""
    return fee / (order_value * small_book / book_value) * 1e4
