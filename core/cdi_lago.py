"""
cdi_lago.py - Daily CDI from the lake's copy (``data/b3lake/cdi_daily.parquet``, BCB SGS 12 refreshed every night by
the ``b3-datalake`` routine), so a backtest does not depend on the SGS answering every ten-year window.

Same contract as ``market_data.baixar_cdi_diario``: decimal per day, named ``cdi_diario``, covering ``years`` up to
``today``. Kept out of ``core/bcb.py`` and ``core/market_data.py`` on purpose: both are in the paper-v1 watched path.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

from core import bcb, market_data

logger = logging.getLogger(__name__)

LAKE_CDI_FILE = Path(__file__).resolve().parents[1] / "data" / "b3lake" / "cdi_daily.parquet"
_START_SLACK_DAYS = 10  # the window may start on a weekend or a holiday stretch; a lake starting later is truncated


def _window_start(years: int, today: date) -> date:
    """The first day ``bcb.baixar_cdi_diario`` covers: ``years`` back plus 30 days of margin."""
    return today - timedelta(days=int(years * 365.25) + 30)


def _bcb_tail(start: date, today: date) -> pd.Series:
    """SGS 12 from ``start`` to ``today`` in decimal per day (``core.bcb`` asks for the whole decade and cuts)."""
    return bcb.baixar_serie(bcb.SERIE_CDI, start, today) / 100.0


def daily_cdi(years: int, today: date, path: Path | None = None,
              fetch_tail: Callable[[date, date], pd.Series] = _bcb_tail,
              download_all: Callable[[int, date], pd.Series] = market_data.baixar_cdi_diario) -> pd.Series:
    """
    CDI in decimal per day from the lake file, in date order. The CDI of a day comes out on the next one, so the
    lake is current when it has the last weekday before ``today``; otherwise the BCB is asked for the days after it.
    Without a usable lake file (missing, torn, another schema, or starting after the window), the whole window from
    the BCB as before (``download_all``, which raises ``market_data.ErroDadosMercado``).
    """
    try:
        table = pd.read_parquet(path or LAKE_CDI_FILE)
        cdi = pd.Series(table["cdi_pct"].to_numpy() / 100.0, index=pd.to_datetime(table["date"]), name="cdi_diario")
    except FileNotFoundError:
        return download_all(years, today)
    except (OSError, ValueError, KeyError) as error:  # torn mid-copy, or another schema (ArrowInvalid: ValueError)
        logger.warning("Lake CDI file unusable (%s); downloading the window from the BCB.", error)
        return download_all(years, today)
    cdi = cdi.sort_index()
    start = pd.Timestamp(_window_start(years, today))
    if cdi.empty or cdi.index[0] > start + pd.Timedelta(days=_START_SLACK_DAYS):
        return download_all(years, today)
    last = cdi.index[-1]
    if last < pd.Timestamp(today) - pd.offsets.BDay(1):
        try:
            tail = fetch_tail((last + pd.Timedelta(days=1)).date(), today)
        except bcb.ErroBcb as error:
            logger.warning("CDI after %s unavailable (%s); the lake series ends there.", last.date(), error)
        else:
            cdi = pd.concat([cdi, tail[tail.index > last]]).rename("cdi_diario")
    return cdi.loc[start:pd.Timestamp(today)]
