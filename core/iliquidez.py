"""
iliquidez.py - Small-cap / illiquidity study (step b): which stocks a small account holds below the universe's
liquidity floor, their returns valued at the bid-ask midpoint, and the forced exit of a name that stops trading.

Inputs are the frozen files of ``scripts/export_iliquidez.py`` in the lake (``docs/dados/2026-10-20-iliquidez``):
``monthly`` (month, security, liquidity and distress columns), ``universe`` (month, security), ``daily`` (security,
date, close, best_bid, best_ask, factor_change, ret_total, return_break). Every decision uses months before its own.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from core.execucao import Liquidez

LOOKBACK_MONTHS = 12
MIN_TRADED_RATIO = 0.80  # mean over the 12 months, a month without trades counting as 0
BUCKET_SIZE = 150        # the most traded names below the universe: what a small account can still buy
MAX_SHARE_OF_VOLUME = 0.05  # a position above this share of the median daily volume (R$) cannot be bought in a day


def _lookback(monthly: pd.DataFrame, decision: pd.Timestamp) -> pd.DataFrame:
    """The LOOKBACK_MONTHS monthly rows before the decision's month (all known at the decision)."""
    month = decision.to_period("M")
    return monthly[(monthly["month"] >= (month - LOOKBACK_MONTHS).to_timestamp())
                   & (monthly["month"] < month.to_timestamp())]


def _month_before(window: pd.DataFrame, decision: pd.Timestamp) -> pd.DataFrame:
    return window[window["month"] == (decision.to_period("M") - 1).to_timestamp()]


def bucket(monthly: pd.DataFrame, universe: pd.DataFrame, decision: pd.Timestamp, size: int = BUCKET_SIZE) -> list[str]:
    """Securities outside the universe of the decision's month, traded on at least MIN_TRADED_RATIO of the sessions
    of the LOOKBACK_MONTHS before it, and not in distress (judicial recovery and the like) in the month before; the
    ``size`` with the highest median of the monthly median daily volume (R$) over those months, in name order."""
    window = _lookback(monthly, decision)
    inside = set(universe.loc[universe["month"] == decision.to_period("M").to_timestamp(), "security"])
    last = _month_before(window, decision)
    excluded = inside | set(last.loc[last["distress"].fillna("") != "", "security"])
    traded = window.groupby("security")["traded_ratio"].sum() / LOOKBACK_MONTHS
    eligible = sorted(set(traded.index[traded >= MIN_TRADED_RATIO]) - excluded)
    volume = window[window["security"].isin(eligible)].groupby("security")["median_dtv_brl"].median()
    return sorted(volume.sort_values(ascending=False, kind="stable").index[:size])


def picks(monthly: pd.DataFrame, candidates: list[str], decision: pd.Timestamp, n: int, capital: float) -> list[str]:
    """The ``n`` candidates with the highest median monthly Amihud (|return| per R$ traded) over the LOOKBACK_MONTHS
    before the decision, in name order: the least liquid of what is still tradable. A name whose position
    (``capital`` / ``n``) is above MAX_SHARE_OF_VOLUME of its median daily volume in the month before gives way to the
    next one."""
    window = _lookback(monthly, decision)
    amihud = window[window["security"].isin(candidates)].groupby("security")["amihud"].median()
    last = _month_before(window, decision).set_index("security")
    volume = last["median_dtv_brl"].reindex(amihud.index).fillna(0.0)
    fits = amihud[volume * MAX_SHARE_OF_VOLUME >= capital / n]
    return sorted(fits.sort_values(ascending=False, kind="stable").index[:n])


def mid_returns(daily: pd.DataFrame) -> pd.DataFrame:
    """Date x security total returns valued at the bid-ask midpoint: midpoint over the previous session's (that
    security's previous row) times ``factor_change`` (splits and payouts), minus 1. A close-to-close return on a thin
    stock alternates between bid and ask and adds that bounce as return; the midpoint does not. Without a valid quote
    (bid > 0 and ask > bid) on either day, the lake's close-to-close ``ret_total``. Where ``ret_total`` is null (first
    session, a return break, the first session after > 180 days without trading) the return is unknown."""
    ordered = daily.sort_values(["security", "date"])
    valid = (ordered["best_bid"] > 0.0) & (ordered["best_ask"] > ordered["best_bid"])
    mid = ((ordered["best_bid"] + ordered["best_ask"]) / 2.0).where(valid)
    previous = mid.groupby(ordered["security"]).shift(1)
    at_mid = mid / previous * ordered["factor_change"] - 1.0
    known = ordered["ret_total"].notna()
    returns = ordered.assign(ret=at_mid.where(at_mid.notna(), ordered["ret_total"]).where(known))
    return returns.pivot(index="date", columns="security", values="ret").rename_axis(index=None, columns=None)


def mask_suspect_days(returns: pd.DataFrame, suspect: pd.DataFrame) -> pd.DataFrame:
    """``returns`` with the (date, security) pairs of ``suspect`` set to unknown: rejected share events and
    unexplained overnight jumps, which the lake keeps as returns because some are real (README of the lake)."""
    masked = returns.copy()
    for security, date in suspect[["security", "date"]].itertuples(index=False):
        if security in masked.columns and date in masked.index:
            masked.loc[date, security] = float("nan")
    return masked


EXPORT_HOLD_MONTHS = 3  # the lake's scripts/export_iliquidez.py: rows of a month m only for a name outside the
                        # universe in some month of [m - 3, m]


def traded_sessions(daily: pd.DataFrame, universe: pd.DataFrame, sessions: pd.DatetimeIndex) -> pd.DataFrame:
    """Session x security: True where ``daily`` has a row (the lake has a row only for a session with trades), and in
    a month the export left out (the name in the ``universe`` in all of the EXPORT_HOLD_MONTHS + 1 months up to it,
    so it was liquid and its silence there is unknown, not observed)."""
    rows = daily.assign(traded=True).pivot(index="date", columns="security", values="traded")
    traded = rows.reindex(sessions).fillna(False).astype(bool).rename_axis(index=None, columns=None)
    inside = set(zip(universe["security"], universe["month"]))
    months = sessions.to_period("M")
    for security in traded.columns:
        left_out = [all((security, (month - k).to_timestamp()) in inside for k in range(EXPORT_HOLD_MONTHS + 1))
                    for month in months.unique()]
        dropped = months.unique()[left_out]
        traded.loc[months.isin(dropped), security] = True
    return traded


HAIRCUT = 0.50              # a name that stops trading is sold at its last midpoint less this (exit unknown)
HAIRCUT_IN_DISTRESS = 1.00  # ... or written off when its last trading month was in distress
MAX_SILENCE = 63            # sessions without a trade after which a held name is treated as gone


def apply_forced_exits(returns: pd.DataFrame, traded: pd.DataFrame, targets: pd.DataFrame,
                       distressed: set[tuple[str, pd.Timestamp]], max_silence: int = MAX_SILENCE,
                       haircut: float = HAIRCUT, haircut_in_distress: float = HAIRCUT_IN_DISTRESS,
                       ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    The lake stops a delisted name at its last close, so a backtest would keep it at that value: the one-sided bias
    that can make the whole result. Here a name held in a period (``targets``: a row per rebalance, valid until the
    next) that goes more than ``max_silence`` sessions without a trade (``traded``) loses ``haircut`` on the session
    after its last trade (or on the period's first session, if the silence began before it), ``haircut_in_distress``
    when (security, month of its last trade) is in ``distressed``, and earns nothing for the rest of the period: at the
    next rebalance what is left is sold. Returns the adjusted returns and one row per exit (date, security, haircut).
    """
    adjusted = returns.copy()
    exits = []
    sessions = returns.index
    starts = [sessions.get_loc(date) for date in targets.index] + [len(sessions)]
    for (period, weights), start, end in zip(targets.iterrows(), starts[:-1], starts[1:]):
        for security in weights.index[weights > 0.0]:
            trades = np.flatnonzero(traded[security].to_numpy())
            following = np.append(trades[1:], len(sessions))
            gone = (following - trades - 1 > max_silence) & (following > start) & (np.maximum(trades + 1, start) < end)
            if not gone.any():
                continue
            last = int(trades[gone][0])
            exit_at = max(last + 1, start)
            month = sessions[last].to_period("M").to_timestamp()
            cut = haircut_in_distress if (security, month) in distressed else haircut
            adjusted.iloc[exit_at, adjusted.columns.get_loc(security)] = -cut
            adjusted.iloc[exit_at + 1:end, adjusted.columns.get_loc(security)] = 0.0
            exits.append((sessions[exit_at], security, cut))
    return adjusted, pd.DataFrame(exits, columns=["date", "security", "haircut"])


def liquidity(monthly: pd.DataFrame) -> Liquidez:
    """Month x security inputs of ``core.execucao``'s cost per order: half spread = half the larger of the quoted
    spread (close bid/ask) and the Corwin-Schultz estimate (high/low), so a missing or narrow quote does not make a
    thin name cheap; the daily volatility and the median daily volume (R$) of the month."""
    frame = monthly.assign(half_spread=monthly[["quoted_spread", "corwin_schultz"]].max(axis=1) / 2.0)
    wide = frame.pivot(index="month", columns="security")
    return Liquidez(*(wide[column].rename_axis(index=None, columns=None)
                      for column in ("half_spread", "vol_daily", "median_dtv_brl")))


PICKS = 15  # equal weights: R$ 667 a name with R$ 10 thousand, one round lot or a few fractional shares


def targets(monthly: pd.DataFrame, universe: pd.DataFrame, decisions: list[pd.Timestamp], n: int = PICKS,
            capital: float = 10_000.0) -> pd.DataFrame:
    """Decision x security weights: equal over the ``picks`` from the ``bucket`` of each decision, 0 elsewhere."""
    rows = {}
    for decision in decisions:
        chosen = picks(monthly, bucket(monthly, universe, decision), decision, n, capital)
        if not chosen:
            raise ValueError(f"No name to hold on {decision:%Y-%m-%d}")
        rows[decision] = pd.Series(1.0 / len(chosen), index=chosen)
    return pd.DataFrame(rows).T.fillna(0.0).sort_index(axis=1)


REBALANCE_EVERY = 63  # sessions, as the engine: a quarter


def decision_dates(sessions: pd.DatetimeIndex, first: pd.Timestamp, every: int = REBALANCE_EVERY,
                   offset: int = 0) -> list[pd.Timestamp]:
    """Sessions from ``first`` moved ``offset`` sessions on (the phase), then every ``every`` sessions."""
    return list(sessions[sessions.get_loc(first) + offset::every])


def value_without_costs(weights: pd.DataFrame, returns: pd.DataFrame, capital: float) -> pd.Series:
    """Account value from the session before the first decision, no costs and no tax, each period compounded on its
    own: an anchor for ``core.execucao.refazer`` written without its loop (unknown returns count as 0)."""
    sessions = returns.index[returns.index.get_loc(weights.index[0]) - 1:]
    value, values = float(capital), pd.Series(float(capital), index=sessions)
    ends = list(weights.index[1:]) + [None]
    for (decision, w), end in zip(weights.iterrows(), ends):
        period = returns.loc[decision:end].iloc[:-1] if end is not None else returns.loc[decision:]
        growth = (1.0 + period[w.index].fillna(0.0)).cumprod()
        path = value * (growth * w).sum(axis=1)
        values.loc[path.index] = path
        value = float(path.iloc[-1])
    return values
