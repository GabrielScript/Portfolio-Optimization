"""
eventos.py - Aleph Bot, stage 1: the two pre-registered intraday event studies on 15-minute bars
(docs/superpowers/specs/2026-11-12-aleph-bot-design.md and its two amendments). Decision on a bar's close, fill at
the next bar's open; long only; out the same day; cost per side = half the bar's spread (at least half a tick) plus
exchange fees.
"""

from __future__ import annotations

from collections.abc import Collection

import numpy as np
import pandas as pd

TICK = 0.01
FEES = 2.5e-4                    # exchange fees per side
MAX_SPREAD_BPS = 10.0
Z_OVERREACTION, Z_GAP = -3.0, -2.0
SLOT_SESSIONS, DAILY_SESSIONS = 20, 60
HOLD_BARS = 4                    # 60 minutes
GAP_ENTRY, GAP_EXIT = "10:15", "11:30"
COLUMNS = ["day", "entry_time", "exit_time", "entry", "exit", "z", "gross", "net"]


def regular_bars(bars: pd.DataFrame) -> pd.DataFrame:
    """The day's bars without the closing call: no bar starting at 18:00, and no 17:00 bar when it is the day's
    last."""
    hora = bars["time"].dt.strftime("%H:%M")
    ultima = bars.groupby(bars["time"].dt.normalize())["time"].transform("max")
    return bars[(hora != "18:00") & ~((hora == "17:00") & (bars["time"] == ultima))].reset_index(drop=True)


def side_cost(price: float, spread_points: float, cost_multiple: float = 1.0) -> float:
    return cost_multiple * (max(spread_points * TICK / 2, TICK / 2) / price + FEES)


def spread_bps(price: float, spread_points: float) -> float:
    return spread_points * TICK / price * 1e4


def clustered_t(trades: pd.DataFrame) -> tuple[float, float, int]:
    """(t, mean, clusters) of the per-trade net return, standard error clustered by day."""
    if trades.empty:
        return float("nan"), float("nan"), 0
    media = float(trades["net"].mean())
    somas = (trades["net"] - media).groupby(trades["day"]).sum()
    dias = len(somas)
    if dias < 2:
        return float("nan"), media, dias
    erro = float(np.sqrt((somas**2).sum() * dias / (dias - 1)) / len(trades))
    return (media / erro if erro > 0 else float("nan")), media, dias


def _trade(dia: pd.Timestamp, entrada: pd.Series, saida: pd.Series, z: float, cost_multiple: float) -> dict:
    bruto = float(saida["open"]) / float(entrada["open"]) - 1.0
    custo = side_cost(float(entrada["open"]), float(entrada["spread"]), cost_multiple) + \
        side_cost(float(saida["open"]), float(saida["spread"]), cost_multiple)
    return {"day": dia, "entry_time": entrada["time"], "exit_time": saida["time"], "entry": float(entrada["open"]),
            "exit": float(saida["open"]), "z": float(z), "gross": bruto, "net": bruto - custo}


def _beta_of(beta: pd.Series, dia: pd.Timestamp) -> float:
    return float(beta.get(dia, 1.0)) if len(beta) else 1.0


def _frames(bars: pd.DataFrame, market: pd.DataFrame) -> pd.DataFrame:
    """The name's regular bars with the market's open and close of the same bar, the day and the slot."""
    nome = regular_bars(bars)
    mercado = regular_bars(market)[["time", "open", "close"]].rename(columns={"open": "m_open", "close": "m_close"})
    junto = nome.merge(mercado, on="time", how="left")
    junto["day"] = junto["time"].dt.normalize()
    junto["slot"] = junto["time"].dt.strftime("%H:%M")
    return junto


def overreaction(bars: pd.DataFrame, market: pd.DataFrame, beta: pd.Series,
                 excluded: Collection[pd.Timestamp] = (), cost_multiple: float = 1.0) -> pd.DataFrame:
    """Pattern (a): the name's abnormal 30-minute return (two bars, less beta x the market's), over the standard
    deviation of that same slot's abnormal return in the previous 20 sessions, at or below -3: long at the next
    bar's open, out at the open four bars later or at the open of the day's last regular bar."""
    q = _frames(bars, market)
    anterior = q.groupby("day")[["open", "m_open"]].shift(1)
    q["r30"] = q["close"] / anterior["open"] - 1.0
    q["m30"] = (q["m_close"] / anterior["m_open"] - 1.0).fillna(0.0)
    q["beta"] = q["day"].map(lambda dia: _beta_of(beta, dia))
    q["abn"] = q["r30"] - q["beta"] * q["m30"]
    porslot = q.pivot(index="day", columns="slot", values="abn")
    escala = porslot.shift(1).rolling(SLOT_SESSIONS, min_periods=SLOT_SESSIONS).std()
    q["scale"] = [escala.at[d, s] for d, s in zip(q["day"], q["slot"])]
    q["z"] = q["abn"] / q["scale"]
    trades = []
    for dia, d in q.groupby("day", sort=True):
        if dia in excluded or len(d) < HOLD_BARS + 3:
            continue
        d = d.reset_index(drop=True)
        ultima, livre = len(d) - 1, 0
        for i in np.flatnonzero((d["z"] <= Z_OVERREACTION).to_numpy()):
            entrada = i + 1
            if entrada < livre or entrada >= ultima:
                continue
            if spread_bps(float(d.at[entrada, "open"]), float(d.at[entrada, "spread"])) > MAX_SPREAD_BPS:
                continue
            saida = min(entrada + HOLD_BARS, ultima)
            trades.append(_trade(dia, d.loc[entrada], d.loc[saida], d.at[i, "z"], cost_multiple))
            livre = saida
    return pd.DataFrame(trades, columns=COLUMNS)


def daily_closes(bars: pd.DataFrame) -> pd.Series:
    """The last regular bar's close of each day."""
    r = regular_bars(bars)
    return r.groupby(r["time"].dt.normalize())["close"].last()


def rolling_beta(name_close: pd.Series, market_close: pd.Series, sessions: int = DAILY_SESSIONS) -> pd.Series:
    """The slope of the name's daily return on the market's over the previous ``sessions`` sessions (the day's own
    return not included), clipped to [0, 2]; 1 without enough history."""
    r = pd.concat([name_close.pct_change(), market_close.pct_change()], axis=1, keys=["n", "m"], sort=True).dropna()
    cov = r["n"].rolling(sessions).cov(r["m"]).shift(1)
    var = r["m"].rolling(sessions).var().shift(1)
    return (cov / var).clip(0.0, 2.0).fillna(1.0)


def opening_gap(bars: pd.DataFrame, market: pd.DataFrame, beta: pd.Series,
                excluded: Collection[pd.Timestamp] = (), cost_multiple: float = 1.0) -> pd.DataFrame:
    """Pattern (c): the overnight abnormal return (first open against the previous last regular close, less beta x
    the market's) at or below -2 standard deviations of the name's abnormal daily return over the previous 60
    sessions: long at the open of the 10:15 bar, out at the open of the 11:30 bar."""
    q = _frames(bars, market)
    dias = q.groupby("day")
    resumo = pd.DataFrame({"open": dias["open"].first(), "close": dias["close"].last(),
                           "m_open": dias["m_open"].first(), "m_close": dias["m_close"].last()})
    b = pd.Series([_beta_of(beta, dia) for dia in resumo.index], index=resumo.index)
    mercado_dia = resumo["m_close"].pct_change().fillna(0.0)
    abn_dia = resumo["close"].pct_change() - b * mercado_dia
    escala = abn_dia.shift(1).rolling(DAILY_SESSIONS, min_periods=DAILY_SESSIONS).std()
    gap = resumo["open"] / resumo["close"].shift(1) - 1.0
    gap_mercado = (resumo["m_open"] / resumo["m_close"].shift(1) - 1.0).fillna(0.0)
    z = (gap - b * gap_mercado) / escala
    trades = []
    for dia in z.index[z <= Z_GAP]:
        if dia in excluded:
            continue
        d = q[q["day"] == dia].set_index("slot")
        if GAP_ENTRY not in d.index or GAP_EXIT not in d.index:
            continue
        entrada, saida = d.loc[GAP_ENTRY], d.loc[GAP_EXIT]
        if spread_bps(float(entrada["open"]), float(entrada["spread"])) > MAX_SPREAD_BPS:
            continue
        trades.append(_trade(dia, entrada, saida, z[dia], cost_multiple))
    return pd.DataFrame(trades, columns=COLUMNS)
