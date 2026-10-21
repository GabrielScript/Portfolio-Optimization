"""
Step (b): small caps below the universe's liquidity floor, held by a small account that pays no income tax on months
with stock sales up to R$ 20 thousand, against SMAL11 bought and held. Does the illiquidity premium survive what a
small account really pays? It does not change the paper trading (paper-v1) nor any candidate. Network only for the
CDI (descriptive), when the lake's copy is missing.

Usage: .venv/Scripts/python -X utf8 scripts/avaliar_iliquidez.py > docs/iliquidez-2026-10-20.txt

Pre-registration (2026-10-20, written and committed before running; after that commit this text does not change):
- Data: docs/dados/2026-10-20-iliquidez (lake commit bc21ed6, data through 2026-10-20; the lake's
  scripts/export_iliquidez.py, byte-for-byte reproducible), checked against its SHA256SUMS: daily rows (close, low,
  high, best bid and ask at the close, the day's factor change, the lake's daily total return) of the stocks and units
  traded on >= 80% of the sessions of some month since 2007, kept for the months they were outside the universe or up
  to 3 months after, plus SMAL11 and BOVA11; their monthly liquidity and distress; the suspect days; dividends; the
  point-in-time universe (rule D1). A first export (lake e1ee331) kept only names that reached R$ 50 thousand a day
  in some month, later ones included: a look-ahead in favour of the study, fixed before any run (lake 3e699ba).
- Strategy (`core.iliquidez`): decisions every 63 sessions from the first session of 2009 moved 0, 21 or 42 sessions
  (the 3 phases) to the end of the file. At each decision, from months before its own only: the bucket = names
  outside the universe of the decision's month, traded on >= 80% of the sessions of the 12 months before (mean of
  the monthly ratios, a month without trades = 0), not in distress (judicial recovery and the like) in the month
  before, the 150 with the highest median of the monthly median daily volume (R$) (measured before the commit: 95 to
  127 names, so the cap does not bind); from it, the 15 with the highest median monthly Amihud, a name whose position
  (capital / 15, the initial capital) is above 5% of its median daily volume of the month before giving way to the
  next one (measured: every pick trades >= R$ 13 thousand a day); equal weights, bought at the close of the session
  before the decision (as the engine).
- Returns: total return at the bid-ask midpoint (midpoint ratio x the lake's factor change, so payouts and splits
  count), the lake's close-to-close total return where a day lacks a valid quote (bid > 0, ask > bid); unknown (0 in
  the account) on a return break and on the suspect days (rejected share events, unexplained jumps).
- Forced exit: a held name more than 63 sessions without a trade after its last one loses 50% on the session after
  that trade (100% when its last trading month was in distress) and earns nothing until the next rebalance, when what
  is left is sold. A session of a month the export left out (the name in the universe in all 4 months up to it)
  counts as traded: its silence there is unknown.
- Execution (`core.execucao.refazer`, as in the final test of the Conservadora): emolumentos 0.030% per order; half
  spread = half the larger of the quoted spread (close bid/ask) and Corwin-Schultz, of the month before the order's
  close; impact 1 x daily vol x sqrt(order / median daily volume); a name without a measure in that month takes the
  month's median. Income tax 15% on the month's net gain, months with stock sales up to R$ 20 thousand exempt (Lei
  11.033/2004, art. 3, I), losses carried; dividends into the cost basis by the `yield` of dividends.parquet on the
  ex-date. Exit value = equity less the tax of selling everything at that close.
- Benchmark: SMAL11 bought at the same close and held: emolumentos only, no exemption (ETF), no dividends (the ETF
  reinvests). BOVA11 the same, descriptive.
- Gates, before any measure: every curve, redone by `refazer` with no costs and no tax, equals `value_without_costs`
  (relative deviation <= 1e-9); no held row has a close outside [low, high] by more than 2% (measured before the
  commit: 4 rows in the whole file outside by less, all on 2020-06-08, EALT3, ISAE3, RPAD6 and SULA3, up to 1.4%).
  Either fails: the script stops.
- Decision fixed now: the premium survives for a small account = P[exit value > SMAL11's] (`psr_da_diferenca`,
  arithmetic) >= 0.95 with R$ 10 thousand in all 3 phases. Survives: the small-cap sleeve becomes a research candidate
  for a later pre-registered test of its place next to the optimizer (no paper trading without the user's OK). Does
  not survive: closed; no other bucket, haircut, cost, size or period without a new pre-registration. N for the DSR:
  33 (docs/tentativas.csv, one new configuration); no DSR in this run (the decision is the PSR against SMAL11, as in
  the final test).
- Descriptive only (the decision does not change): BOVA11; the same strategy with no costs and no tax, and with costs
  but no tax, each against SMAL11 redone the same way (the premium apart from the exemption); without the masking;
  close-to-close returns (the bid-ask bounce in); haircut 0; without forced exits (the naive backtest); 2x the half
  spread; execution one session later (both curves); R$ 1 thousand and R$ 100 thousand (their own picks); P per
  subperiod (2009-2015, 2016-2026); forced exits; names the capacity rule passed over; names replaced per rebalance;
  CAGR of the CDI.
- Simplifications: no brokerage; the month's tax leaves the equity pro rata, not as a sale; the exemption counts only
  this account's sales; orders of R$ 667 trade in the odd-lot market, whose spread may be wider than the round lot's
  in the lake; fractional positions (no lot rounding); the exit does not pay the selling orders' cost; delisting cash
  (a tender offer's price) is not in the lake, so a name that leaves after a tender offer takes the 50% haircut too (a
  bias against the strategy); the universe as the lake has it.
- Expectation recorded: the costs eat the premium. Picks trade R$ 13-20 thousand a day with half spreads of 1-2%, so
  an order costs ~2% and replacing ~40% of the names a quarter costs 4% to 9% of equity a year. With no costs and no
  tax, P > 0.90 in 2 or 3 phases; net, P between 0.2 and 0.8 in all 3 (does not survive). Tax paid along the way
  below 3% of the final exit; 3 to 15 forced exits per phase; without forced exits the CAGR 1 to 3 points higher;
  close-to-close returns above midpoint ones.
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core import market_data  # noqa: E402
from core.backtest import ResultadoBacktest  # noqa: E402
from core.cdi_lago import daily_cdi  # noqa: E402
from core.config import DIAS_UTEIS_ANO  # noqa: E402
from core.execucao import SEM_CUSTOS, Custos, Liquidez, Refeita, refazer  # noqa: E402
from core.iliquidez import (PICKS, apply_forced_exits, bucket, decision_dates, liquidity,  # noqa: E402
                            mask_suspect_days, mid_returns, picks, targets, traded_sessions, value_without_costs)
from core.impostos import ISENCAO_DE_VENDAS  # noqa: E402
from core.metrics import cagr, drawdown  # noqa: E402
from core.robustez import psr_da_diferenca  # noqa: E402
from scripts.avaliar_execucao import decidir_prova, proventos_diarios  # noqa: E402
from scripts.avaliar_robustez import _tabela, conferir_hashes, contar_tentativas  # noqa: E402

FOLDER = Path(__file__).resolve().parent.parent / "docs" / "dados" / "2026-10-20-iliquidez"
FILES = ("daily", "dividends", "monthly", "suspect_days", "universe_monthly")
CAPITAL = 10_000.0
OTHER_CAPITALS = (1_000.0, 100_000.0)
COSTS = Custos(emolumentos=0.0003, impacto=1.0)
ETF_COSTS = Custos(emolumentos=0.0003, impacto=0.0)
STRATEGY, BENCHMARK, MARKET = "Small caps", "SMAL11", "BOVA11"
PHASES = (0, 21, 42)
FIRST_DECISION_FROM = pd.Timestamp("2009-01-01")
SUBPERIODS = {"2009-2015": ("2009-01-01", "2015-12-31"), "2016-2026": ("2016-01-01", "2026-12-31")}
GATE_TOLERANCE = 0.02
ANCHOR_TOLERANCE = 1e-9
CDI_YEARS = 20
IN_PERCENT = ("CAGR", "MDD", "Cost per year", "SMAL11 CAGR")
IN_REAIS = ("Final exit (R$)", "Tax paid (R$)", "Tax at exit (R$)")


def held_rows(daily: pd.DataFrame, weights: pd.DataFrame) -> pd.DataFrame:
    """The rows of ``daily`` of each name with a weight, from its decision to the session before the next one."""
    periods = []
    ends = list(weights.index[1:]) + [None]
    for (decision, row), end in zip(weights.iterrows(), ends):
        in_period = (daily["date"] >= decision) & ((daily["date"] < end) if end is not None else True)
        periods.append(daily[in_period & daily["security"].isin(row.index[row > 0.0])])
    return pd.concat(periods)


def close_far_outside_range(rows: pd.DataFrame, tolerance: float) -> pd.DataFrame:
    """Rows whose close is below the day's low or above its high by more than ``tolerance`` (relative)."""
    return rows[(rows["close"] < rows["low"] * (1.0 - tolerance)) | (rows["close"] > rows["high"] * (1.0 + tolerance))]


def one_session_later(weights: pd.DataFrame, sessions: pd.DatetimeIndex) -> pd.DataFrame:
    """The same weights, each decided one session later (bought at the decision's close instead of the one before)."""
    return weights.set_axis(sessions[sessions.get_indexer(weights.index) + 1])


def given_way(monthly: pd.DataFrame, universe: pd.DataFrame, decision: pd.Timestamp, n: int,
              capital: float) -> list[str]:
    """The picks the capacity rule passed over at ``decision``: chosen without it, not with it."""
    candidates = bucket(monthly, universe, decision)
    return sorted(set(picks(monthly, candidates, decision, n, 0.0)) - set(picks(monthly, candidates, decision, n,
                                                                                 capital)))


def measures(account: Refeita, benchmark: Refeita) -> dict[str, float]:
    """P[exit value > the benchmark's] arithmetic (the decision's) and in logs, CAGR and MDD of the exit value, the
    final exit value, the cost per year over the mean equity, the tax paid along the way and the tax the exit would
    still charge."""
    years = (len(account.valores) - 1) / DIAS_UTEIS_ANO
    return {"P arithmetic": psr_da_diferenca(account.saida, benchmark.saida),
            "P log": psr_da_diferenca(account.saida, benchmark.saida, log=True),
            "CAGR": cagr(account.saida), "MDD": drawdown(account.saida)[1],
            "Final exit (R$)": float(account.saida.iloc[-1]),
            "Cost per year": float(account.custos.sum() / account.valores.mean() / years),
            "Tax paid (R$)": float(account.impostos.sum()),
            "Tax at exit (R$)": float(account.valores.iloc[-1] - account.saida.iloc[-1])}


@dataclass(frozen=True)
class Inputs:
    daily: pd.DataFrame
    monthly: pd.DataFrame
    universe: pd.DataFrame
    suspect: pd.DataFrame
    dividends: pd.DataFrame
    sha256: dict[str, str]


def load() -> Inputs:
    sha256 = conferir_hashes(FOLDER, [f"{name}.parquet" for name in FILES])
    read = {name: pd.read_parquet(FOLDER / f"{name}.parquet") for name in FILES}
    dated = {name: read[name].assign(**{column: pd.to_datetime(read[name][column])})
             for name, column in (("daily", "date"), ("monthly", "month"), ("universe_monthly", "month"),
                                  ("suspect_days", "date"))}
    return Inputs(dated["daily"], dated["monthly"], dated["universe_monthly"], dated["suspect_days"],
                  read["dividends"], sha256)


@dataclass(frozen=True)
class Market:
    returns: pd.DataFrame       # midpoint, suspect days masked (the primary)
    unmasked: pd.DataFrame      # midpoint, suspect days kept
    close: pd.DataFrame         # close to close, suspect days masked
    traded: pd.DataFrame
    distressed: set[tuple[str, pd.Timestamp]]
    liquidity: Liquidez
    payouts: pd.DataFrame

    @property
    def sessions(self) -> pd.DatetimeIndex:
        return self.returns.index


def market(inputs: Inputs) -> Market:
    mid = mid_returns(inputs.daily).astype(float)
    close = inputs.daily.pivot(index="date", columns="security", values="ret_total").rename_axis(
        index=None, columns=None).reindex(index=mid.index, columns=mid.columns).astype(float)
    in_distress = inputs.monthly[inputs.monthly["distress"].fillna("") != ""]
    return Market(mask_suspect_days(mid, inputs.suspect), mid, mask_suspect_days(close, inputs.suspect),
                  traded_sessions(inputs.daily, inputs.universe, mid.index),
                  set(zip(in_distress["security"], in_distress["month"])), liquidity(inputs.monthly),
                  proventos_diarios(inputs.dividends, mid.index, mid.columns))


def curve(name: str, weights: pd.DataFrame, returns: pd.DataFrame, capital: float) -> ResultadoBacktest:
    """The weights as an engine curve (value with no costs and no tax), checked against `refazer` (the anchor)."""
    values = value_without_costs(weights, returns, capital)
    turnover = weights.diff().fillna(weights).abs().sum(axis=1)
    result = ResultadoBacktest(name, "walk-forward OOS", values, weights, turnover, 0.0)
    redone = refazer(result, returns, capital, SEM_CUSTOS, com_ir=False)
    deviation = float((redone.valores / values - 1.0).abs().max())
    if not deviation <= ANCHOR_TOLERANCE:
        raise SystemExit(f"{name}: redone with no costs and no tax, {deviation:.2e} off value_without_costs")
    return result


def accounts(weights: pd.DataFrame, returns: pd.DataFrame, mkt: Market, capital: float, costs: Custos = COSTS,
             half_spread_times: float = 1.0, with_tax: bool = True) -> dict[str, Refeita]:
    """The strategy (stocks: exemption, dividends into the basis) and SMAL11 and BOVA11 bought at the same close."""
    free = costs == SEM_CUSTOS
    liq = None if free else Liquidez(mkt.liquidity.meio_spread * half_spread_times, mkt.liquidity.vol_diaria,
                                     mkt.liquidity.volume_mediano)
    done = {STRATEGY: refazer(curve(STRATEGY, weights, returns, capital), returns, capital, costs, liq,
                              com_ir=with_tax, isencao=ISENCAO_DE_VENDAS, proventos=mkt.payouts)}
    for etf in (BENCHMARK, MARKET):
        held = curve(etf, pd.DataFrame({etf: [1.0]}, index=weights.index[:1]), returns, capital)
        done[etf] = refazer(held, returns, capital, SEM_CUSTOS if free else ETF_COSTS, com_ir=with_tax, isencao=0.0)
    return done


def gate(daily: pd.DataFrame, weights: pd.DataFrame) -> str:
    rows = held_rows(daily, weights)
    far = close_far_outside_range(rows, GATE_TOLERANCE)
    if not far.empty:
        raise SystemExit(f"Data gate: held closes outside the day's range by > 2%: "
                         f"{far[['security', 'date']].astype(str).values.tolist()}")
    near = close_far_outside_range(rows, 1e-6)
    crossed = int(((rows["best_ask"] > 0.0) & (rows["best_bid"] > rows["best_ask"])).sum())
    return f"{len(rows)} held rows; {len(near)} closes outside the day's range by <= 2%; {crossed} crossed quotes"


def variants(inputs: Inputs, mkt: Market, weights: pd.DataFrame) -> pd.DataFrame:
    """Descriptive: one change at a time from the primary, each against SMAL11 under the same change."""
    def exited(returns: pd.DataFrame, held: pd.DataFrame, **haircuts: float) -> pd.DataFrame:
        return apply_forced_exits(returns, mkt.traded, held, mkt.distressed, **haircuts)[0]

    primary = exited(mkt.returns, weights)
    later = one_session_later(weights, mkt.sessions)
    runs = {"No costs, no tax": accounts(weights, primary, mkt, CAPITAL, SEM_CUSTOS, with_tax=False),
            "Costs, no tax": accounts(weights, primary, mkt, CAPITAL, with_tax=False),
            "Not masked": accounts(weights, exited(mkt.unmasked, weights), mkt, CAPITAL),
            "Close to close": accounts(weights, exited(mkt.close, weights), mkt, CAPITAL),
            "Haircut 0": accounts(weights, exited(mkt.returns, weights, haircut=0.0, haircut_in_distress=0.0),
                                  mkt, CAPITAL),
            "No forced exits": accounts(weights, mkt.returns, mkt, CAPITAL),
            "2x half spread": accounts(weights, primary, mkt, CAPITAL, half_spread_times=2.0),
            "One session later": accounts(later, exited(mkt.returns, later), mkt, CAPITAL)}
    for capital in OTHER_CAPITALS:
        own = targets(inputs.monthly, inputs.universe, list(weights.index), capital=capital)
        runs[f"R$ {capital:,.0f}".replace(",", ".")] = accounts(own, exited(mkt.returns, own), mkt, capital)
    rows = {}
    for name, done in runs.items():
        row = measures(done[STRATEGY], done[BENCHMARK])
        rows[name] = {"P arithmetic": row["P arithmetic"], "CAGR": row["CAGR"],
                      "SMAL11 CAGR": cagr(done[BENCHMARK].saida), "Cost per year": row["Cost per year"]}
    return pd.DataFrame(rows).T


def describe(inputs: Inputs, done: dict[str, Refeita], weights: pd.DataFrame, exits: pd.DataFrame) -> None:
    for label, (start, end) in SUBPERIODS.items():
        strategy, benchmark = done[STRATEGY].saida.loc[start:end], done[BENCHMARK].saida.loc[start:end]
        print(f"{label}: P arithmetic {psr_da_diferenca(strategy, benchmark):.3f}; CAGR {cagr(strategy):.2%} "
              f"vs SMAL11 {cagr(benchmark):.2%}")
    print(f"Forced exits: {len(exits)} ({int((exits['haircut'] >= 1.0).sum())} written off)")
    for row in exits.itertuples(index=False):
        print(f"  {row.date:%Y-%m-%d} {row.security} -{row.haircut:.0%}")
    passed_over = [len(given_way(inputs.monthly, inputs.universe, decision, PICKS, CAPITAL))
                   for decision in weights.index]
    held = weights > 0.0
    replaced = (held & ~held.shift(fill_value=False)).sum(axis=1).iloc[1:]
    print(f"Capacity: {sum(passed_over)} picks passed over in {sum(n > 0 for n in passed_over)} of {len(weights)} "
          f"decisions; names replaced per rebalance: mean {replaced.mean():.1f} of {PICKS}")


def cdi_cagr(dates: pd.DatetimeIndex, today: date) -> str:
    try:
        cdi = daily_cdi(CDI_YEARS, today)
    except market_data.ErroDadosMercado as error:
        return f"unavailable ({error})"
    growth = np.cumprod(np.r_[1.0, 1.0 + cdi.reindex(dates[1:]).fillna(0.0).to_numpy()])
    return f"{cagr(pd.Series(growth)):.2%}"


def _format(table: pd.DataFrame) -> str:
    reais = {column: table[column].round().astype("int64") for column in IN_REAIS if column in table}
    return _tabela(table.assign(**reais), IN_PERCENT).replace(",", ".")


def run_phase(inputs: Inputs, mkt: Market, phase: int, today: date) -> float:
    """Gate, anchors, the primary table (the decision's P), the variants and the description of one phase."""
    decisions = decision_dates(mkt.sessions, mkt.sessions[mkt.sessions >= FIRST_DECISION_FROM][0], offset=phase)
    weights = targets(inputs.monthly, inputs.universe, decisions, capital=CAPITAL)
    checked = gate(inputs.daily, weights)
    returns, exits = apply_forced_exits(mkt.returns, mkt.traded, weights, mkt.distressed)
    done = accounts(weights, returns, mkt, CAPITAL)
    table = pd.DataFrame({name: measures(account, done[BENCHMARK]) for name, account in done.items()}).T
    table.loc[BENCHMARK, ["P arithmetic", "P log"]] = float("nan")
    print(f"\n=== Phase {phase}: {len(decisions)} decisions, first {decisions[0]:%Y-%m-%d}, "
          f"end {mkt.sessions[-1]:%Y-%m-%d}; anchors checked; {checked}; "
          f"CDI CAGR {cdi_cagr(done[STRATEGY].valores.index, today)} ===")
    print(f"\n--- R$ 10.000, primary ---\n{_format(table)}")
    print(f"\n--- Descriptive variants ---\n{_format(variants(inputs, mkt, weights))}\n")
    describe(inputs, done, weights, exits)
    return float(table.loc[STRATEGY, "P arithmetic"])


def main() -> int:
    started = time.perf_counter()
    today = date.fromisoformat(sys.argv[1]) if len(sys.argv) > 1 else date.today()
    inputs = load()
    print(f"Files (sha256): {inputs.sha256}")
    print(f"N = {contar_tentativas()} (no DSR); {COSTS}; ETFs {ETF_COSTS}; capital R$ 10.000; {PICKS} picks")
    mkt = market(inputs)
    p_by_phase = []
    for phase in PHASES:
        p_by_phase.append(run_phase(inputs, mkt, phase, today))
        print(f"(phase {phase} done in {time.perf_counter() - started:.0f} s)", flush=True)
    survives = decidir_prova(p_by_phase)
    print("\nSmall caps with R$ 10.000: P[exit > SMAL11 exit] arithmetic by phase "
          + ", ".join(f"{p:.3f}" for p in p_by_phase)
          + f"; the premium {'SURVIVES (research candidate)' if survives else 'DOES NOT SURVIVE (closed)'}")
    print(f"(total time {time.perf_counter() - started:.0f} s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
