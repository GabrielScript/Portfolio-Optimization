# Volatility target on the Composite, inside the realistic execution engine

Date: 2026-11-11. Branch `feat/vol-target-motor`, worktree `../po-vol`. Owed by DECISOES 336.
**Pre-registration: this file is committed before any number of this test exists. After that commit the rule, the
gate and the expectation below do not change.**

## Why

DECISOES 336 measured the overlay on the saved Composite curve: target 20%, 21 sessions, CAGR 20.30% against 19.91%,
maximum drawdown -21.4% against -41.5%. That screen assumed a free daily adjustment of the exposure, charged a flat
10 bps and ignored income tax. This test replays the rule with real orders: every exposure change sells or buys a
slice of every position, pays the liquidity cost of each order and the personal income tax on realized gains.

Nothing here changes what the app buys. No order path, paper log or screen reads this code.

## The rule (one headline arm: "Composite + 20% volatility target")

1. **Carrier.** Attempt 51's Composite, unchanged: `scripts/composto_crescimento_avaliar.py`'s arm on the frozen
   exports of `docs/dados/2026-10-22` (data to 16/10/2026), engine `walk_forward_dinamico` with `Congelamento()`,
   phases 0, 21 and 42, 10 bps in the engine.
2. **Signal.** Realized volatility of the engine curve's daily returns over the 21 sessions ending at the close where
   the order is sent, annualized by sqrt(252). Target share in stocks = min(1, 0.20 / volatility)
   (`core.vol_target.exposicao`, unchanged). Never above 1: no leverage.
3. **Orders.**
   - At each engine rebalance: positions go to target weights x target share; the rest is cash.
   - Between rebalances: when the target share is more than **0.10** away from the current share (stocks / net
     worth), every position is scaled by the same factor to the target share, at that close.
   - House freeze rule (`core.risco.pode_negociar`): no exposure order at the close of a session in which the net
     worth lost 5% or more; the order waits for the next close where the gap still exceeds the band.
4. **Cash.** Earns the session's CDI (lake, `2026-10-21-macro/cdi_daily.parquet`) net of 22.5% income tax (the
   shortest fixed-income bracket). IOF on redemptions under 30 days is not modelled.
5. **Execution** (`core.execucao.refazer`, as in the Conservadora's final proof of 20/10): B3 fees 0.030%, half the
   quoted spread of the month before the order, square-root impact with Y = 1; income tax by average price, 15% on the
   month's net gain, exemption for months with stock sales up to R$ 20 thousand, dividends reinvested into the cost
   base. Capital R$ 10 thousand, R$ 100 thousand and R$ 1 million.
6. **Liquidity data.** `docs/dados/2026-11-11-liquidez/liquidez.parquet`: the same three measures as the 20/10 file,
   cut from the lake's `monthly_metrics.parquet` from 2009-11 to 2026-10 (the 20/10 file starts in 2011-11 and the
   Composite starts in 02/2010). Checked by its `SHA256SUMS`.

## Anchors (the script stops if one fails)

- The Composite curve of each phase equals `docs/dados/composto-curvas-2026-10-28.parquet` (relative gap <= 1e-9).
- `refazer` with the engine's 10 bps, no liquidity, no tax, reproduces the engine curve (<= 1e-9).
- `refazer` with the overlay and a target share of 1 on every session equals `refazer` without it (<= 1e-9).

## Gate (fixed now)

The curve that counts is the **exit value** (net worth minus the tax of selling everything at that close), compared
with the plain Composite replayed with the same costs and tax.

**Approved** when, at R$ 10 thousand **and** R$ 100 thousand, in **all three phases**:
- maximum drawdown at least **10 percentage points** shallower than the plain Composite's, and
- CAGR not more than **1.0 percentage point** below the plain Composite's.

R$ 1 million is reported and does not decide.

- Approved: the arm becomes a candidate for a fictitious account (its own spec) and the order logic goes to the
  advisor review required by CLAUDE.md before any real order. The real buy of 19/12 does not change by this result.
- Rejected: the line is closed. No other target, window, band or cash rule without a new pre-registration.

## Reported, not gating (one change from the headline each)

Target 16% and 24%; band 0.05 and 0.20; cash at 0%. For every arm: CAGR, maximum drawdown and Sharpe over the CDI of
the exit value, final exit value, cost per year, tax paid and tax still due at exit, mean share in stocks, exposure
orders per year, and P[arm > plain Composite] (arithmetic and log, `psr_da_diferenca`).

## Trial count

One new row in `docs/tentativas.csv` (the headline arm). The five sensitivity arms decide nothing and are listed in
the output; a later decision that leans on one of them needs its own row and registration.

## Simplifications (written before the numbers)

Those of the 20/10 proof, plus: the order is sent at the same close that ends the volatility window (the engine's
convention; live, the 17:55 call would use the day's indicative close); the cash sits in an instrument with daily
liquidity at 100% of the CDI; IOF under 30 days ignored (the 0% cash arm bounds it from below); the freeze rule looks
at the whole net worth, cash included; the signal comes from the engine's 10 bps curve, not from the replayed one
(they differ only by costs and tax).

## Expectation (registered)

The band and the real cost erode the screen's result: maximum drawdown between -22% and -30% against about -41%, CAGR
within 1.0 point of the plain Composite at R$ 10 thousand (little or no tax on the way: monthly sales under the
exemption for years), and 0.5 to 1.5 points below it at R$ 1 million, where every de-risking sale realizes taxed
gains. 2 to 6 exposure orders a year. Approved at R$ 10 thousand; R$ 100 thousand is the uncertain one.

## Code

- `core/execucao.py`: `ExposureOverlay` (frozen dataclass: `exposure`, `cash_rate`, `band`, `limits`) and an optional
  `overlay` argument of `refazer`; `None` keeps today's behaviour byte for byte. `overlay` with `zona` is an error.
- `scripts/vol_target_motor_avaliar.py`: the run above; output to `docs/vol-target-motor-2026-11-11.txt`.
- `scripts/exportar_liquidez.py`: the liquidity cut.
- Tests first, one at a time, each seen failing: `tests/test_execucao.py` (overlay), `tests/test_vol_target_motor.py`
  (gate and export).

## Tasks

- [x] 1. `ExposureOverlay` in `refazer`: share of 1 is the plain replay; cash at a rebalance earns the rate; an
      exposure order between rebalances scales every position and is taxed by average price; band; freeze; order cost;
      `zona` + overlay refused; missing session in `exposure` refused.
- [x] 2. Liquidity export 2009-11 to 2026-10 with `SHA256SUMS` and `LEIAME.md`.
- [x] 3. Gate function and the evaluation script with the three anchors.
- [x] 4. One run; output committed; DECISOES entry; `docs/tentativas.csv` row; `graphify update .`.
