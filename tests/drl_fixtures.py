"""Small tapes for the DRL tests."""

from __future__ import annotations

import numpy as np
import pandas as pd

from drl.dados import Panel, build_panel


def universe_of(prices: pd.DataFrame, names: list[str] | None = None) -> pd.DataFrame:
    meses = pd.date_range(prices.index[0].to_period("M").to_timestamp(),
                          prices.index[-1].to_period("M").to_timestamp() + pd.offsets.MonthBegin(1), freq="MS")
    return pd.DataFrame([(mes, nome) for mes in meses for nome in (names or list(prices.columns))],
                        columns=["month", "security"])


def random_prices(sessions: int = 400, names: int = 5, seed: int = 0, start: str = "2015-01-02") -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    datas = pd.bdate_range(start, periods=sessions)
    retornos = rng.normal(0.0005, 0.02, size=(sessions, names))
    return pd.DataFrame(100.0 * np.cumprod(1.0 + retornos, axis=0), index=datas,
                        columns=[f"N{i:02d}" for i in range(names)])


def panel_of(prices: pd.DataFrame, cdi: float = 0.0, history: int = 1, names: list[str] | None = None) -> Panel:
    return build_panel(prices, universe_of(prices, names), pd.Series(cdi, index=prices.index), history)
