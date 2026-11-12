"""
signals.py - The five Inovemetria families (spec "The five families"), each a ``NotaDoMes``: (month, candidate
tickers) -> value per ticker, from statements received before the month and patents published before it. Higher
is always "more innovative / more accelerated". An issuer's classes share its value (``core.risco.emissor``).

Builder's call, recorded with the reading: every statement family drops documents whose period ended more than
``core.valor.MESES_DE_IDADE_MAXIMA`` months before the month (a company that stopped filing), as the value score does.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd

from core.inovemetria.data import InnovationData, issuer_patents
from core.inovemetria.diffusion import Fit, fit_laws, forecast, stage
from core.risco import emissor
from core.valor import MESES_DE_IDADE_MAXIMA, SETOR_FORA, NotaDoMes, documentos_conhecidos
from core.views import escore_normal

QUARTERS_MAX = 40            # S: the last 40 quarter ends at most
PAIRS_MIN = 12               # S: consecutive quarter pairs needed
HORIZON = 4                  # S: forecast quarters
SHARE_QUARTERS = 12          # E: quarter ends of the share path
PEERS_MIN = 3                # E: issuers in the sector with the issuer itself; B: names for a sector median
PATENT_MONTHS = 60           # P1 window
HALF_WINDOW = 24             # P2: recent and before windows
SNAPSHOT_HOLD = pd.Timestamp("2026-01-01")   # BADEPI dispatches end 30/12/2025: P is held at this month after it
CONVEXITY_MONTHS, CONVEXITY_MIN = 36, 24     # X window and the fewest months with a return


def _per_ticker(values: pd.Series, candidates: Sequence[str]) -> pd.Series:
    """Issuer values on the candidate tickers that have one."""
    tickers = [t for t in candidates if emissor(t) in values.index]
    return pd.Series([values[emissor(t)] for t in tickers], index=tickers, dtype=float)


def _known(fundamentals: pd.DataFrame, month: pd.Timestamp, *accounts: str) -> pd.DataFrame:
    """Rows of ``accounts`` received before ``month`` (refilings resolved), period ends not older than the limit."""
    rows = documentos_conhecidos(fundamentals[fundamentals["account"].isin(accounts)], month)
    return rows[rows["dt_refer"] >= month - pd.DateOffset(months=MESES_DE_IDADE_MAXIMA)]


def _history(fundamentals: pd.DataFrame, month: pd.Timestamp, account: str) -> pd.DataFrame:
    """dt_refer x issuer of ``account`` received before ``month``, for issuers whose latest period end is recent."""
    rows = documentos_conhecidos(fundamentals[fundamentals["account"] == account], month)
    table = rows.pivot_table(index="dt_refer", columns="issuer", values="value", aggfunc="last").sort_index()
    fresh = table.apply(lambda c: c.last_valid_index()) >= month - pd.DateOffset(months=MESES_DE_IDADE_MAXIMA)
    return table.loc[:, fresh[fresh].index]


def revenue_quarters(fundamentals: pd.DataFrame, month: pd.Timestamp, ipca: pd.Series | None = None) -> pd.DataFrame:
    """dt_refer x issuer: trailing-twelve-month revenue known before ``month``, in reais of the last IPCA month
    when ``ipca`` is given (each quarter deflated by the index of its period end's month)."""
    table = _history(fundamentals, month, "revenue")
    if ipca is None or table.empty:
        return table
    level = ipca.reindex(table.index.to_period("M").to_timestamp()).ffill().to_numpy()
    return table.div(level, axis=0)


def _consecutive(series: pd.Series) -> tuple[np.ndarray, np.ndarray] | None:
    """(x_t, x_{t+1}) over consecutive quarter ends among the last ``QUARTERS_MAX`` values; None if any value is not
    positive or fewer than ``PAIRS_MIN`` pairs."""
    s = series.dropna().iloc[-QUARTERS_MAX:]
    if (s <= 0).any():
        return None
    quarters = s.index.to_period("Q")
    adjacent = (quarters[1:] - quarters[:-1]).map(lambda d: d.n) == 1
    if adjacent.sum() < PAIRS_MIN:
        return None
    values = s.to_numpy()
    return values[:-1][adjacent], values[1:][adjacent]


def _fit_of(series: pd.Series, cache: dict) -> Fit | None:
    pairs = _consecutive(series)
    if pairs is None:
        return None
    key = (series.name, series.dropna().iloc[-QUARTERS_MAX:].to_numpy().tobytes())
    if key not in cache:
        cache[key] = fit_laws(*pairs)
    return cache[key]


def diffusion_table(data: InnovationData, month: pd.Timestamp, cache: dict | None = None) -> pd.DataFrame:
    """issuer: law, stage, forecast (log real growth over ``HORIZON`` quarters), x_over_m (None for exponential)."""
    cache = {} if cache is None else cache
    revenue = revenue_quarters(data.fundamentals, month, data.ipca)
    rows = {}
    for issuer in revenue.columns:
        series = revenue[issuer].rename(issuer)
        fit = _fit_of(series, cache)
        if fit is None:
            continue
        last = float(series.dropna().iloc[-1])
        rows[issuer] = {"law": fit.law, "stage": stage(fit, last), "forecast": np.log(forecast(fit, last, HORIZON) / last),
                        "x_over_m": None if fit.law == "exponential" else last / fit.params[-1]}
    return pd.DataFrame.from_dict(rows, orient="index")


def diffusion_score(data: InnovationData) -> NotaDoMes:
    """S: the chosen law's forecast log growth of real revenue over the next four quarters."""
    cache: dict = {}

    def _nota(month: pd.Timestamp, candidates: Sequence[str]) -> pd.Series:
        wanted = {emissor(t) for t in candidates}
        revenue = revenue_quarters(data.fundamentals, month, data.ipca)
        values = {}
        for issuer in wanted & set(revenue.columns):
            series = revenue[issuer].rename(issuer)
            fit = _fit_of(series, cache)
            if fit is not None:
                last = float(series.dropna().iloc[-1])
                values[issuer] = np.log(forecast(fit, last, HORIZON) / last)
        return _per_ticker(pd.Series(values, dtype=float), candidates)
    return _nota


def encroachment_score(data: InnovationData) -> NotaDoMes:
    """E: the least-squares slope per year of logit(share of the activity sector's revenue) over the issuer's last
    12 quarter ends, peers = candidates of the same activity sector with revenue at all of them."""
    def _nota(month: pd.Timestamp, candidates: Sequence[str]) -> pd.Series:
        revenue = _history(data.fundamentals, month, "revenue")
        issuers = [i for i in dict.fromkeys(emissor(t) for t in candidates) if i in revenue.columns]
        sector = data.activity.reindex(issuers)
        values = {}
        for issuer in issuers:
            own = revenue[issuer].dropna().iloc[-SHARE_QUARTERS:]
            if len(own) < SHARE_QUARTERS or pd.isna(sector[issuer]):
                continue
            peers = [i for i in issuers if sector.get(i) == sector[issuer]]
            block = revenue.loc[own.index, peers].dropna(axis=1)
            if block.shape[1] < PEERS_MIN or issuer not in block.columns or (block <= 0).to_numpy().any():
                continue
            share = block[issuer] / block.sum(axis=1)
            years = (own.index - own.index[0]).days.to_numpy() / 365.25
            values[issuer] = np.polyfit(years, np.log(share / (1 - share)).to_numpy(), 1)[0]
        return _per_ticker(pd.Series(values, dtype=float), candidates)
    return _nota


def _yearly_change(table: pd.DataFrame, account: str) -> tuple[pd.Series, pd.Series]:
    """Per issuer: the latest period end t with ``account`` at t and t - 1 year; (index of t, index of t - 1y)."""
    has = table[account].dropna().reset_index()
    has["prev"] = has["dt_refer"] - pd.DateOffset(years=1)
    has = has[pd.MultiIndex.from_frame(has[["issuer", "prev"]]).isin(table[account].dropna().index)]
    latest = has.sort_values("dt_refer").drop_duplicates("issuer", keep="last").set_index("issuer")
    return latest["dt_refer"], latest["prev"]


def effort_components(data: InnovationData, month: pd.Timestamp) -> pd.DataFrame:
    """issuer x (B1, B2) before the sector adjustment, over total assets a year before."""
    rows = documentos_conhecidos(data.fundamentals[data.fundamentals["account"].isin(
        ["intangibles", "fixed_assets", "depreciation", "total_assets"])], month)
    rows = rows[rows["dt_refer"] >= month - pd.DateOffset(months=MESES_DE_IDADE_MAXIMA + 12)]
    table = rows.pivot_table(index=["issuer", "dt_refer"], columns="account", values="value", aggfunc="last")
    if "intangibles" not in table.columns or "total_assets" not in table.columns:
        return pd.DataFrame(columns=["B1", "B2"])
    t, prev = _yearly_change(table, "intangibles")
    t = t[t >= month - pd.DateOffset(months=MESES_DE_IDADE_MAXIMA)]
    prev = prev.reindex(t.index)
    now = table.reindex(pd.MultiIndex.from_arrays([t.index, t.to_numpy()])).set_axis(t.index)
    before = table.reindex(pd.MultiIndex.from_arrays([prev.index, prev.to_numpy()])).set_axis(prev.index)
    assets = before["total_assets"].where(before["total_assets"] > 0)
    out = pd.DataFrame(index=t.index)
    out["B1"] = (now["intangibles"] - before["intangibles"]) / assets
    if {"fixed_assets", "depreciation"} <= set(table.columns):
        out["B2"] = (now["fixed_assets"] + now["intangibles"] - before["fixed_assets"] - before["intangibles"]
                     + now["depreciation"].abs()) / assets
    else:
        out["B2"] = np.nan
    return out.dropna(how="all")


def effort_score(data: InnovationData, sectors: Mapping[str, str]) -> NotaDoMes:
    """B: the mean normal score of B1 and B2, each minus its median in the candidates' broad sector (the
    candidates' median when the sector has fewer than 3 names); the financial sector has no score."""
    def _nota(month: pd.Timestamp, candidates: Sequence[str]) -> pd.Series:
        tickers = [t for t in candidates if sectors.get(t) not in (None, SETOR_FORA)]
        parts = effort_components(data, month)
        table = pd.DataFrame({c: _per_ticker(parts[c].dropna(), tickers) for c in ("B1", "B2")})
        if table.empty:
            return pd.Series(dtype=float)
        sector = pd.Series({t: sectors[t] for t in table.index})
        size = sector.map(sector.value_counts())
        relative = pd.DataFrame({c: table[c] - table[c].groupby(sector).transform("median")
                                 .where(size >= PEERS_MIN, table[c].median()) for c in ("B1", "B2")})
        return _mean_score(relative)
    return _nota


def _mean_score(table: pd.DataFrame) -> pd.Series:
    """Mean of the columns' normal scores, with those that exist (``core.views.nota_media_dos_escores`` on frames)."""
    scores = pd.DataFrame({c: escore_normal(table[c].dropna()) for c in table.columns}).reindex(table.index)
    return scores.mean(axis=1).dropna()


def _assets_year_before(data: InnovationData, month: pd.Timestamp) -> pd.Series:
    """Issuer -> total assets a year before its latest known period end (positive only)."""
    table = documentos_conhecidos(data.fundamentals[data.fundamentals["account"] == "total_assets"], month)
    table = table.pivot_table(index=["issuer", "dt_refer"], columns="account", values="value", aggfunc="last")
    if table.empty:
        return pd.Series(dtype=float)
    t, prev = _yearly_change(table, "total_assets")
    t = t[t >= month - pd.DateOffset(months=MESES_DE_IDADE_MAXIMA)]
    values = table["total_assets"].reindex(pd.MultiIndex.from_arrays([t.index, prev.reindex(t.index).to_numpy()]))
    values = pd.Series(values.to_numpy(), index=t.index)
    return values[values > 0]


def patent_components(data: InnovationData, month: pd.Timestamp) -> tuple[pd.Series, pd.Series]:
    """(P1, P2) per issuer with positive assets a year before: applications published in the last 60 months per
    R$ billion of those assets, and (recent - before) / (1 + recent + before) over two 24-month windows. The
    patent windows end at ``min(month, SNAPSHOT_HOLD)``."""
    assets = _assets_year_before(data, month)
    end = min(month, SNAPSHOT_HOLD)
    held = issuer_patents(data, end)
    held = held[held["issuer"].isin(assets.index)]
    def _count(months_back: int, months_to: int = 0) -> pd.Series:
        window = (held["known_from"] >= end - pd.DateOffset(months=months_back)) &                  (held["known_from"] < end - pd.DateOffset(months=months_to))
        return held[window].groupby("issuer").size().reindex(assets.index, fill_value=0).astype(float)
    recent, before = _count(HALF_WINDOW), _count(2 * HALF_WINDOW, HALF_WINDOW)
    p1 = _count(PATENT_MONTHS) / (assets / 1e9)
    p2 = (recent - before) / (1.0 + recent + before)
    return p1, p2


def patent_score(data: InnovationData) -> NotaDoMes:
    """P: the mean normal score of P1 and P2 among the candidates with assets (no application: both zero)."""
    def _nota(month: pd.Timestamp, candidates: Sequence[str]) -> pd.Series:
        p1, p2 = patent_components(data, month)
        table = pd.DataFrame({"P1": _per_ticker(p1, candidates), "P2": _per_ticker(p2, candidates)})
        return _mean_score(table) if len(table) else pd.Series(dtype=float)
    return _nota


def _quadratic_t(path: np.ndarray) -> float:
    """t-statistic of c in the least-squares fit path ~ a + b t + c t^2."""
    n = len(path)
    t = np.arange(n, dtype=float)
    design = np.column_stack([np.ones(n), t, t ** 2])
    coef, *_ = np.linalg.lstsq(design, path, rcond=None)
    resid = path - design @ coef
    sigma2 = resid @ resid / (n - 3)
    cov = sigma2 * np.linalg.inv(design.T @ design)
    return float(coef[2] / np.sqrt(cov[2, 2])) if cov[2, 2] > 0 else np.nan


def convexity_score(monthly_returns: pd.DataFrame) -> NotaDoMes:
    """X: on the cumulative log total return over the 36 months before the month (at least 24 with a return),
    the t-statistic of the quadratic term."""
    def _nota(month: pd.Timestamp, candidates: Sequence[str]) -> pd.Series:
        window = monthly_returns.loc[monthly_returns.index < month].iloc[-CONVEXITY_MONTHS:]
        values = {}
        for ticker in candidates:
            if ticker not in window.columns:
                continue
            r = window[ticker].dropna()
            if len(r) >= CONVEXITY_MIN and (r > -1).all():
                values[ticker] = _quadratic_t(np.cumsum(np.log1p(r.to_numpy())))
        return pd.Series(values, dtype=float).dropna()
    return _nota


def families(data: InnovationData, sectors: Mapping[str, str], monthly_returns: pd.DataFrame) -> dict[str, NotaDoMes]:
    """The five families of the spec, by their letters."""
    return {"S": diffusion_score(data), "E": encroachment_score(data), "B": effort_score(data, sectors),
            "P": patent_score(data), "X": convexity_score(monthly_returns)}


WIPO_FIELDS = list(range(1, 36))


def field_counts(data: InnovationData, month: pd.Timestamp) -> pd.DataFrame:
    """issuer x the 35 WIPO fields: applications published in P1's 60-month window (ending at
    ``min(month, SNAPSHOT_HOLD)``), each adding 1 to each distinct field it has."""
    end = min(month, SNAPSHOT_HOLD)
    held = issuer_patents(data, end)
    held = held[held["known_from"] >= end - pd.DateOffset(months=PATENT_MONTHS)]
    exploded = held[["issuer", "fields"]].explode("fields").dropna(subset=["fields"])
    if exploded.empty:
        return pd.DataFrame(columns=WIPO_FIELDS, dtype=float)
    table = pd.crosstab(exploded["issuer"], exploded["fields"].astype(int))
    return table.reindex(columns=WIPO_FIELDS, fill_value=0)
