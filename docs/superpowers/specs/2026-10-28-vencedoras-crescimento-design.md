# Extreme winners confirmed by profit growth — design and pre-registration (attempt 50)

_2026-10-28, session 56. Written and committed before any code or curve of this attempt exists. User's request:
"execute the strategies with the stocks (all, including BDRs and ETFs) that returned more than 100% over the last 12
months", narrowed by the user to arm B ("only those in the top half of profit growth"). Arm A (the raw rule) is not run
and is not a trial. DECISOES 163._

## 1. The rule

At each rebalance of the engine (decision month m = the month of the rebalance session):

1. **Candidates:** the names the engine leaves eligible (point-in-time `universe_monthly` of month m with a full
   training window, delisted names included), as for the Conservadora.
2. **Extreme winner:** total return over the 12 calendar months m-12 .. m-1 (no skipped month: the user's "last 12
   months", not the 12-1 momentum convention), from `monthly_total_returns.parquet`, all 12 months present, **above
   100%**.
3. **Confirmed by profit growth:** `core.crescimento_lucro.nota_de_crescimento` (the signal that passed its own IC
   gate, DECISOES 161) **strictly above its median over every scored candidate of month m** (the month's universe, not
   only the winners). A name without a score (no year-ago filing) is not confirmed.
4. **Picks:** the confirmed winners ordered by profit-growth score, through `core.sinais.carteira_da_nota` with at
   most 20 names and 4 per sector (one class per issuer, no name without a sector), as the Conservadora.
5. **Weights:** each pick gets `min(1/n, 20%)`; a sector whose picks sum above 30% is scaled down to 30%; if the
   ex-ante volatility of the stock weights (Ledoit-Wolf covariance of the training window, as the Conservadora's
   check) is above 40%, every stock weight is multiplied by `40% / vol`. The fund's fixed rules
   (`core.risco.checar_carteira` on the stock weights) must then hold; a violation stops the run (wiring bug).
6. **Cash:** `1 - sum of stock weights` goes to the CDI, a total-return index built from the frozen
   `docs/dados/2026-10-21-macro/cdi_daily.parquet` and added as a column to the prices **of this arm only**. A
   rebalance with no pick is 100% CDI (never "keep the old portfolio").

## 2. Everything else is fixed

Engine `walk_forward_dinamico`, `CFG_BT` (1,260-session training window, rebalance every 63 sessions, 10 bps per side,
also on CDI moves), phases 0/21/42, prices cut at 16/10/2026, frozen folders `docs/dados/2026-10-22/` (prices,
universe, monthly returns, dividends, sectors) and `docs/dados/2026-10-24-cvm/` (statements dated by CVM receipt).
The Conservadora row must reproduce the published rerun (`check_plumbing`) before the arm's numbers are read.
Risk-free rate for Sharpe and DSR: the same frozen daily CDI. No network.

**Deviations from the chat, written before:** (a) the chat said "monthly"; the rule runs on the engine's 63-session
rebalance so it is compared with the Conservadora on the same dates (the three phases cover the month offsets);
(b) the user asked for all assets including BDRs and ETFs; only stocks enter, because the lake's BDR/ETF price
export keeps only names traded in the last 15 days (survivors only: a backtest on it is biased upwards), BDR dividends
are not in the lake, and ETFs are not in the point-in-time universe.

## 3. Gates (read once)

One trial row: N 49 -> 50. Candidate only if, in all 3 phases, P[> Conservadora] >= 0.95, P[> BOVA11] >= 0.95 and
DSR >= 0.95 (N = 50). Reported, not gates: names per rebalance (min/median/max), rebalances with no pick, mean CDI
share, CAGR, MDD, the year table with 2025 and 2026, years beating BOVA11 and the Conservadora.

- **Candidate:** pre-register a forward paper log; nothing goes to real money before it.
- **Not a candidate:** the "extreme winners" line is **closed**. No retuning of the threshold (100% -> 50%), of the
  confirmation (top half -> top third), of the cap or of the cadence.

## 4. Expectation, written before

Not a candidate is the likelier outcome. Evidence outside the test period: Brazil before 2010 (JKP), 12-month momentum
t 1.27 and MAX (lottery) t 0.53/0.87, none significant; extreme past winners tend to underperform (Bali, Cakici and
Whitelaw 2011) and crash in rebounds (Daniel and Moskowitz 2016). Only 11 stocks were above 100% on 27/10/2026, so the
portfolio will often hold few names and a large CDI share. Caveat for the reader: P[> BOVA11] may pass on the CDI share
alone (BOVA11 7.2% a year, MDD -50%); the Conservadora gate is the real one.
