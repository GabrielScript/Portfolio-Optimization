"""Aleph Bot, stage 1: the two event studies (docs/superpowers/specs/2026-11-12-aleph-bot-design.md and amendments)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from core.bot import eventos as ev


def _dia(dia: str, precos: list[float], inicio: str = "10:00", spread: int = 1) -> pd.DataFrame:
    """Flat bars (open = close = the price of the slot) starting at ``inicio``, 15 minutes apart."""
    horas = pd.date_range(f"{dia} {inicio}", periods=len(precos), freq="15min")
    return pd.DataFrame({"time": horas, "open": precos, "close": precos, "spread": spread})


def test_regular_bars_drop_the_closing_call() -> None:
    barras = pd.concat([_dia("2024-03-04", [10.0] * 29),             # 10:00 .. 17:00: the 17:00 bar is the day's last
                        _dia("2024-03-05", [10.0] * 33)])            # 10:00 .. 18:00
    regulares = ev.regular_bars(barras)
    ultimas = regulares.groupby(regulares["time"].dt.normalize())["time"].max().dt.strftime("%H:%M").tolist()
    assert ultimas == ["16:45", "17:45"]


def test_cost_is_half_the_spread_at_least_half_a_tick_plus_fees() -> None:
    assert ev.side_cost(price=10.0, spread_points=4) == pytest.approx(0.02 / 10.0 + 2.5e-4)
    assert ev.side_cost(price=10.0, spread_points=0) == pytest.approx(0.005 / 10.0 + 2.5e-4)
    assert ev.spread_bps(price=20.0, spread_points=2) == pytest.approx(10.0)


def test_the_clustered_t_counts_days_not_trades() -> None:
    um_por_dia = pd.DataFrame({"day": pd.date_range("2024-01-01", periods=100), "net": [0.01, -0.005] * 50})
    repetido = pd.concat([um_por_dia] * 10)                          # ten identical trades a day: no new information
    assert ev.clustered_t(repetido)[0] == pytest.approx(ev.clustered_t(um_por_dia)[0])
    media, t, n = ev.clustered_t(um_por_dia)[1], ev.clustered_t(um_por_dia)[0], ev.clustered_t(um_por_dia)[2]
    assert media == pytest.approx(0.0025) and n == 100 and t == pytest.approx(0.0025 / (0.0075 / np.sqrt(100)), rel=0.02)


QUEDA = pd.Timestamp("2024-02-06")


def _historia(queda: float, dias: int = 25) -> tuple[pd.DataFrame, pd.DataFrame]:
    """25 quiet days (the name wiggles by 0.1% around 20, the market is flat), then a day with a 30-minute drop."""
    datas = pd.bdate_range("2024-01-02", periods=dias + 1).strftime("%Y-%m-%d")
    rng = np.random.default_rng(0)
    nome = [_dia(d, list(20.0 * (1 + 0.001 * rng.standard_normal(28)))) for d in datas[:-1]]
    ultimo = [20.0] * 28
    ultimo[8:] = [20.0 * (1 + queda)] * 4 + [20.0] * 16              # the 12:00 bar opens lower; back by 13:00
    nome.append(_dia(datas[-1], ultimo))
    mercado = pd.concat([_dia(d, [100.0] * 28) for d in datas])
    return pd.concat(nome, ignore_index=True), mercado


def test_overreaction_enters_after_the_drop_and_leaves_an_hour_later() -> None:
    nome, mercado = _historia(queda=-0.03)
    trades = ev.overreaction(nome, mercado, beta=pd.Series(1.0, index=pd.to_datetime(nome["time"].dt.normalize().unique())))
    trades = trades[trades["day"] == QUEDA]          # the random quiet days may hold a 3-sigma wiggle of their own
    assert len(trades) == 1
    t = trades.iloc[0]
    assert t["entry_time"] == pd.Timestamp("2024-02-06 12:15") and t["exit_time"] == pd.Timestamp("2024-02-06 13:15")
    assert t["gross"] == pytest.approx(20.0 / 19.4 - 1)
    assert t["net"] == pytest.approx(t["gross"] - ev.side_cost(19.4, 1) - ev.side_cost(20.0, 1))
    calmo = ev.overreaction(*_historia(queda=-0.001), beta=pd.Series(dtype=float))
    assert calmo[calmo["day"] == QUEDA].empty                        # an ordinary wiggle: nothing


def test_overreaction_needs_twenty_sessions_of_history() -> None:
    nome, mercado = _historia(queda=-0.03, dias=10)
    assert ev.overreaction(nome, mercado, beta=pd.Series(dtype=float)).empty


def test_a_wide_spread_at_entry_keeps_the_bot_out() -> None:
    nome, mercado = _historia(queda=-0.03)
    nome["spread"] = 5                                               # R$ 0.05 on R$ 19.40 is 26 bps
    assert ev.overreaction(nome, mercado, beta=pd.Series(dtype=float)).empty      # too wide on every day


def test_opening_gap_buys_at_1015_and_sells_at_1130() -> None:
    datas = pd.bdate_range("2024-01-02", periods=62).strftime("%Y-%m-%d")
    rng = np.random.default_rng(1)
    niveis = 30.0 * np.cumprod(1 + 0.01 * rng.standard_normal(61))
    nome = [_dia(d, [float(n)] * 28) for d, n in zip(datas[:-1], niveis)]
    base = float(niveis[-1])
    ultimo = [base * 0.95, base * 0.96, base * 0.96, base * 0.96, base * 0.96, base * 0.96, base * 0.98] + [base * 0.98] * 21
    nome.append(_dia(datas[-1], ultimo))
    mercado = pd.concat([_dia(d, [100.0] * 28) for d in datas])
    trades = ev.opening_gap(pd.concat(nome, ignore_index=True), mercado, beta=pd.Series(dtype=float))
    ultimo_dia = trades[trades["day"] == pd.Timestamp(datas[-1])]
    assert len(ultimo_dia) == 1
    t = ultimo_dia.iloc[0]
    assert t["entry_time"] == pd.Timestamp(f"{datas[-1]} 10:15") and t["exit_time"] == pd.Timestamp(f"{datas[-1]} 11:30")
    assert t["gross"] == pytest.approx(0.98 / 0.96 - 1)


def test_excluded_days_never_trade() -> None:
    nome, mercado = _historia(queda=-0.03)
    fora = {pd.Timestamp("2024-02-06")}
    trades = ev.overreaction(nome, mercado, beta=pd.Series(dtype=float), excluded=fora)
    assert trades[trades["day"] == QUEDA].empty
