"""The MT5 executor's pure part (docs/superpowers/specs/2026-11-12-mt5-demo-design.md, plan step 1)."""

from __future__ import annotations

import pandas as pd
import pytest

from core import mt5_book as mb
from core.mt5_book import Order


def test_target_shares_follow_the_house_plan_in_whole_shares() -> None:
    nomes = ["AAAA3", "BBBB3", "CCCC3", "DDDD3", "EEEE3"]
    pesos = pd.Series(0.2, index=nomes)
    cotacoes = pd.Series([10.0, 33.0, 7.0, 51.0, 2.0], index=nomes)
    alvo = mb.target_shares(pesos, cotacoes, 10_000.0)
    valores = alvo * cotacoes
    assert alvo.dtype.kind == "i" and list(alvo.index) == nomes
    assert valores.sum() <= 10_000.0 / 1.001                      # the house's 10 bps reserve, never more than the book
    assert valores.sum() >= 10_000.0 / 1.001 - cotacoes.max()     # and close to all of it
    assert (valores <= 2_000.0 + 1e-9).all()                       # 20% a name


def test_target_shares_respect_the_20_percent_cap() -> None:
    pesos = pd.Series({"AAAA3": 0.9, "BBBB3": 0.1})
    alvo = mb.target_shares(pesos, pd.Series({"AAAA3": 1.0, "BBBB3": 1.0}), 1_000.0)
    assert alvo["AAAA3"] <= 200


def test_orders_sell_first_and_split_round_and_fractional_lots() -> None:
    alvo = pd.Series({"AAAA3": 250, "BBBB3": 0, "CCCC3": 40})
    detidas = pd.Series({"BBBB3": 130, "CCCC3": 40, "DDDD3": 7})
    ordens = mb.orders_for(alvo, detidas)
    assert ordens == [Order("BBBB3", "BBBB3", "sell", 100), Order("BBBB3", "BBBB3F", "sell", 30),
                      Order("DDDD3", "DDDD3F", "sell", 7),
                      Order("AAAA3", "AAAA3", "buy", 200), Order("AAAA3", "AAAA3F", "buy", 50)]


def test_no_order_when_the_book_already_holds_the_target() -> None:
    assert mb.orders_for(pd.Series({"AAAA3": 10}), pd.Series({"AAAA3": 10})) == []


def test_limit_prices_are_a_half_percent_band_around_the_last_price_on_the_tick() -> None:
    # the user, 2026-11-13: 0.5%, not 2% (XP's demo fills a limit order at its limit at once: the first buy paid 2%)
    assert mb.limit_price("buy", 10.00) == pytest.approx(10.05)
    assert mb.limit_price("sell", 10.00) == pytest.approx(9.95)
    assert mb.limit_price("buy", 2.09) == pytest.approx(2.11)      # 2.10045 rounded up to the cent
    assert mb.limit_price("sell", 2.09) == pytest.approx(2.07)     # 2.07955 rounded down
    with pytest.raises(ValueError):
        mb.limit_price("buy", 0.0)


def test_the_account_must_be_xp_demo() -> None:
    assert mb.check_account(is_demo=True, server="XPMT5-DEMO") is None
    assert "DEMO" in mb.check_account(is_demo=False, server="XPMT5-DEMO")
    assert "server" in mb.check_account(is_demo=True, server="XPMT5-PRD")


def test_orders_only_on_allowed_names() -> None:
    ordens = [Order("AAAA3", "AAAA3", "buy", 100), Order("ZZZZ3", "ZZZZ3F", "sell", 5)]
    assert mb.check_names(ordens, {"AAAA3", "ZZZZ3"}) is None
    assert "ZZZZ3" in mb.check_names(ordens, {"AAAA3"})


def test_buys_never_exceed_the_book_cash_plus_its_sells() -> None:
    cotacoes = pd.Series({"AAAA3": 10.0, "BBBB3": 20.0})
    ordens = [Order("BBBB3", "BBBB3", "sell", 100), Order("AAAA3", "AAAA3", "buy", 300)]
    assert mb.check_cash(ordens, cotacoes, cash=1_000.0) is None    # 3,000 <= 1,000 + 2,000
    assert "cash" in mb.check_cash(ordens, cotacoes, cash=999.0)


def test_a_quote_must_be_live() -> None:
    agora = pd.Timestamp("2026-11-13 17:55")
    assert mb.check_quote("PETR4", 44.6, 44.7, agora - pd.Timedelta(seconds=30), agora) is None
    assert "zero" in mb.check_quote("PETR4", 0.0, 44.7, agora, agora)
    assert "old" in mb.check_quote("PETR4", 44.6, 44.7, agora - pd.Timedelta(minutes=6), agora)


def test_cash_moves_with_the_fills_and_their_fees() -> None:
    fills = [("sell", 100, 20.0, 1.0), ("buy", 300, 10.0, 2.5)]   # side, volume, price, commission
    assert mb.cash_after(1_000.0, fills) == pytest.approx(1_000.0 + 2_000.0 - 1.0 - 3_000.0 - 2.5)


def test_the_cost_of_an_order_at_the_users_real_size() -> None:
    # an order of 6,700 in a 100,000 book is 67 at R$ 1,000; R$ 4.90 on 67 is 731 bps
    assert mb.cost_at_small_book_bps(6_700.0, 100_000.0) == pytest.approx(4.90 / 67.0 * 1e4)


def test_a_name_without_a_fractional_symbol_is_sent_whole_on_its_standard_symbol() -> None:
    # the Composite 80/20 (spec 2026-11-13): BITH11 has volume step 1 and no BITH11F on the terminal
    ordens = mb.orders_for(pd.Series({"BITH11": 222, "AAAA3": 222}), pd.Series(dtype=int), unit_lot={"BITH11"})
    assert ordens == [Order("AAAA3", "AAAA3", "buy", 200), Order("AAAA3", "AAAA3F", "buy", 22),
                      Order("BITH11", "BITH11", "buy", 222)]


def test_the_80_20_target_holds_20_percent_in_the_etf_under_the_name_cap() -> None:
    pesos = pd.Series({"AAAA3": 0.4, "BBBB3": 0.4, "BITH11": 0.2})
    alvo = mb.target_shares(pesos, pd.Series({"AAAA3": 10.0, "BBBB3": 20.0, "BITH11": 90.0}), 100_000.0)
    assert 0.19 < alvo["BITH11"] * 90.0 / 100_000.0 <= 0.20
