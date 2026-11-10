"""The tape the DRL work trades on, built from the lake's frozen exports: the same total-return prices, the same
point-in-time eligible list and the same CDI the house engine (``core.backtest``) uses, so an agent and the Composite
are measured on one universe.

Rewritten on 2026-11-10. The first version read ``lago/precos_diarios.parquet`` (1,544 columns, every BDR and ETF),
called a name eligible after 252 closes, and dropped the return across any session without a trade: a held name that
halted and reopened lower cost its holder nothing. Here:

* ``forward[t]`` is the return a holder earns from the close of session ``t`` to the close of ``t + 1``. A session
  without a trade earns 0 and the whole move lands on the session the name trades again; a delisted name stays at its
  last close. This is ``core.backtest.retornos_da_carteira``.
* ``eligible[t]`` is true for the names the lake lists for the month of session ``t + 1`` (the session the choice
  earns), that have a close on ``t`` and an unbroken run of ``history`` closes up to it.
* ``cdi[t]`` is the CDI accrued from ``t`` to ``t + 1``; cash earns it.

Everything is positional numpy after loading: the environment indexes rows, never dates.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

HISTORY = 252                    # unbroken closes a name needs before it may be bought
NOT_STOCKS = ("BOVA11", "SMAL11", "DIVO11", "IVVB11")    # the ETFs of the price export: references, never traded
PRICES, UNIVERSE, CDI = "universe_prices.parquet", "universe_monthly.parquet", "cdi_daily.parquet"


@dataclass(frozen=True)
class Panel:
    """Sessions x names. ``forward`` and ``cdi`` have one row less than ``sessions``: the last session earns nothing."""

    sessions: pd.DatetimeIndex
    names: tuple[str, ...]
    forward: np.ndarray          # (sessions - 1, names) holder's simple return from t to t + 1
    eligible: np.ndarray         # (sessions, names) bool
    cdi: np.ndarray              # (sessions - 1,) CDI from t to t + 1
    market: np.ndarray           # (sessions, 2) standardized vol20 and vol20 / vol60 of the eligible market, known at t

    def __post_init__(self) -> None:
        sessoes, nomes = len(self.sessions), len(self.names)
        if self.forward.shape != (sessoes - 1, nomes) or self.eligible.shape != (sessoes, nomes):
            raise ValueError(f"shapes disagree: forward {self.forward.shape}, eligible {self.eligible.shape}, "
                             f"sessions {sessoes}, names {nomes}")
        if self.cdi.shape != (sessoes - 1,) or self.market.shape != (sessoes, 2):
            raise ValueError(f"shapes disagree: cdi {self.cdi.shape}, market {self.market.shape}")
        if not (np.isfinite(self.forward).all() and np.isfinite(self.cdi).all() and np.isfinite(self.market).all()):
            raise ValueError("the panel must hold no NaN or inf")

    def columns(self, names: list[str]) -> np.ndarray:
        """Positions of ``names`` in the panel."""
        posicao = {nome: i for i, nome in enumerate(self.names)}
        return np.array([posicao[nome] for nome in names], dtype=int)

    def eligible_names(self, row: int) -> list[str]:
        return [nome for nome, sim in zip(self.names, self.eligible[row]) if sim]


def unbroken_run(valid: np.ndarray) -> np.ndarray:
    """For each cell, how many consecutive true values end at it (0 where false). ``valid``: (sessions, names) bool."""
    corrida = np.zeros(valid.shape, dtype=int)
    atual = np.zeros(valid.shape[1], dtype=int)
    for linha in range(valid.shape[0]):
        atual = np.where(valid[linha], atual + 1, 0)
        corrida[linha] = atual
    return corrida


def expanding_z(values: np.ndarray) -> np.ndarray:
    """Each value minus the mean and over the standard deviation of the values up to and including it; NaN and the
    first values (no dispersion yet) become 0. Causal by construction."""
    serie = pd.Series(values, dtype=float)
    media, desvio = serie.expanding(min_periods=2).mean(), serie.expanding(min_periods=2).std(ddof=0)
    return ((serie - media) / desvio.where(desvio > 0)).fillna(0.0).to_numpy()


def market_features(forward: np.ndarray, eligible: np.ndarray) -> np.ndarray:
    """
    P1's regime indicators on our tape: the 20-session volatility of the market's daily return and its ratio to the
    60-session one, each standardized with an expanding window. The market is the equal-weighted return of the names
    eligible the session before (there is no VIX for B3, so P1's third indicator has no counterpart). Row ``t`` uses
    returns realized up to the close of ``t``.
    """
    sessoes = eligible.shape[0]
    contagem = eligible[:-1].sum(axis=1)
    mercado = np.where(contagem > 0, (forward * eligible[:-1]).sum(axis=1) / np.maximum(contagem, 1), 0.0)
    realizado = pd.Series(np.r_[np.nan, mercado])                    # row t = the return realized at the close of t
    vol20, vol60 = realizado.rolling(20).std(ddof=1), realizado.rolling(60).std(ddof=1)
    razao = (vol20 / vol60.where(vol60 > 0)).to_numpy()
    saida = np.column_stack([expanding_z(vol20.to_numpy()), expanding_z(razao)])
    if saida.shape != (sessoes, 2):
        raise AssertionError(f"market features {saida.shape} for {sessoes} sessions")
    return saida


def build_panel(prices: pd.DataFrame, universe: pd.DataFrame, cdi: pd.Series, history: int = HISTORY) -> Panel:
    """
    ``prices``: total-return index, sessions x names, NaN where the name did not trade. ``universe``: columns ``month``
    and ``security``. ``cdi``: daily rate (a fraction, not percent) indexed by session; a session without a rate
    accrues 0.
    """
    prices = prices.sort_index()
    sessoes = pd.DatetimeIndex(prices.index)
    cheio = prices.ffill()
    forward = (cheio.shift(-1) / cheio - 1.0).iloc[:-1].fillna(0.0).to_numpy(dtype=float)
    validos = prices.notna().to_numpy()
    meses_da_lista = pd.to_datetime(universe["month"])
    listados = {pd.Timestamp(mes): set(nomes) for mes, nomes in universe["security"].groupby(meses_da_lista)}
    seguinte = sessoes[1:].append(pd.DatetimeIndex([sessoes[-1] + pd.offsets.BDay(1)]))
    meses = seguinte.to_period("M").to_timestamp()
    na_lista = np.zeros(validos.shape, dtype=bool)
    for mes in meses.unique():
        coluna = np.fromiter((nome in listados.get(mes, ()) for nome in prices.columns), dtype=bool,
                             count=prices.shape[1])
        na_lista[np.asarray(meses == mes)] = coluna
    eligible = na_lista & (unbroken_run(validos) >= history)
    taxa = cdi.reindex(sessoes[1:]).fillna(0.0).to_numpy(dtype=float)
    return Panel(sessoes, tuple(prices.columns), forward, eligible, taxa, market_features(forward, eligible))


def load_panel(folder: Path, last_session: str | pd.Timestamp | None = None, history: int = HISTORY) -> Panel:
    """The panel from a folder of lake exports (``universe_prices``, ``universe_monthly``, ``cdi_daily``)."""
    tabela = pd.read_parquet(folder / PRICES)
    prices = tabela.set_index(pd.DatetimeIndex(pd.to_datetime(tabela.pop("date")))).astype("float64")
    prices = prices.drop(columns=[nome for nome in NOT_STOCKS if nome in prices.columns])
    if last_session is not None:
        prices = prices.loc[:last_session]
    universe = pd.read_parquet(folder / UNIVERSE, columns=["month", "security"])
    bruto = pd.read_parquet(folder / CDI)
    cdi = pd.Series(bruto["cdi_pct"].to_numpy(dtype=float) / 100.0, index=pd.DatetimeIndex(pd.to_datetime(bruto["date"])))
    return build_panel(prices, universe, cdi[~cdi.index.duplicated(keep="last")].sort_index(), history)
