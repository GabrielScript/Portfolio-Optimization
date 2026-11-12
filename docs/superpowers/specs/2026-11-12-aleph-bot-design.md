# Aleph Bot: an intraday agent on B3's liquid names (pre-registered 2026-11-12)

Written and committed before any return-based number. Branch `feat/aleph-bot`. The user, 2026-11-12: a day-trading
bot that "maximize meu patrimônio", learns from its errors "após ser punido", reads charts, reacts to news and CVM
filings, trades only with statistical conviction; "40 papéis é muito pouco"; day trade (flat every night).

This reopens the reinforcement-learning line closed the same day (DECISOES 353, 355: DRL v2, v3, MARS, MARS-R all
fail) because the problem is a different one: intraday bars and decisions, not daily data and a monthly book.

## What exists (looked at, no return involved)

- XP's MetaTrader 5 demo serves bars from 2021-11-11: M15 complete, M5 for about 100,000 bars a name (PETR4 from
  2022-03), M1 for about one year. Bars carry open, high, low, close, tick volume, real volume and spread.
- The lake's `daily_liquid_panel` (2026-08-03 to 2026-11-11, 769 securities): median daily volume of R$ 20 million
  or more in 98 stocks, 19 tickers ending in 11 (units, ETFs, FIIs) and 19 BDRs.
- Quoted spreads read on 2026-11-12 at 15:30: 2 to 6 bps on the liquid names, 48 to 96 bps on MLAS3.
- Sixteen years of daily data in the lake; CVM filings with dates. News with a timestamp only since 2026-11-06
  (the Sentinela's items).

## Universe

For each month M: the stocks and units of the house universe (lake `universe_monthly`) whose median daily volume
over the 63 sessions before M is R$ 20 million or more, plus the index ETFs BOVA11, SMAL11 and IVVB11. No BDRs
(their intraday price follows the US stock and the dollar: a later phase, with its own data), no FIIs. About 110
names today. A name is tradeable at a decision only if its quoted spread is 10 bps or less.

## Data and its limits

- Bars M5 (M15 where M5 does not reach) of every name ever in the universe, downloaded from the demo terminal to
  `D:/aleph_bot/bars/`, frozen with SHA256SUMS; the repo keeps the manifest, not the bars.
- **Survivorship:** the terminal serves only symbols listed today. Names that left the exchange since 2021 are
  missing; the bias favors the bot and is reported (count of universe-months without bars).
- **Demo bars are the broker's, not B3's tape.** Stage 0 checks each name's daily close from the bars against the
  lake's close; a name off by more than 0.5% on more than 1% of its days is dropped.
- **Split, fixed now:** study period 2021-11-11 to 2024-12-31; holdout 2025-01-01 to the freeze date, read once
  per stage.

## Simulation rules (all stages)

- Decision on a bar's close, fill on the next bar's open. Long or flat only; every position closed by 17:40; no
  position overnight; no leverage.
- Cost per side: half the bar's quoted spread (at least half a tick) plus 2.5 bps of exchange fees; brokerage zero
  (XP's day-trade fee is to be confirmed by the user; every table is repeated at double the cost).
- Day-trade income tax (20% of the month's net gain) is reported beside every result, not netted in the gate.

## Stage 0: data (no trial)

Download, the two checks above, the universe by month, and whether CVM filings in the lake carry a delivery time
of day. If they do not, pattern (b) below is dropped before any reading and N grows by one less.

## Stage 1: do intraday patterns survive the cost? (three event studies, N 126 -> 129)

Signs and parameters fixed here; nothing is tuned.

- **(a) Overreaction.** A name's 30-minute return minus beta x BOVA11's, divided by the name's own 20-day
  volatility of that 30-minute window, at or below -3, with no CVM filing of the name that day: long at the next
  bar, out after 60 minutes or at 17:40.
- **(b) Filing drift.** A "Fato Relevante" or results filing delivered in the session: if the name's abnormal return
  over the 15 minutes after delivery is at or above +2 (same scaling), long at the next bar, out at 17:40.
- **(c) Opening gap.** Overnight abnormal return (open against the previous close, less beta x BOVA11's) at or below
  -2 daily standard deviations with no filing since the previous close: long at 10:15, out at 11:30.

**Gate per pattern:** on the study period, at least 300 trades, mean net return per trade above zero with t >= 3
(standard errors clustered by day); then, read once, on the holdout the mean net return above zero with t >= 2.
A pattern passing on the study period and failing on the holdout fails.

If no pattern passes, the project stops here and the result is recorded.

## Stage 2: the agent (only if stage 1 passes something; N grows by 1)

Not a policy-gradient agent (four fail in this house): an ensemble that prices actions and acts only on conviction.

- **Inputs per name and decision:** the last 48 bars (return, range, volume, spread), BOVA11's same bars, the time
  of day, the stage-1 pattern flags, the daily context from the lake (21 and 252-session return, 63-session
  volatility, the Composite score of the month), a filing flag. Charts are these numbers drawn; no vision model.
- **Model:** five small temporal convolution networks, different seeds, each predicting the 10%, 50% and 90%
  quantiles of the net return of "long for 30, 60 or 120 minutes".
- **Action rule:** enter the best horizon when the ensemble's mean predicted net return is positive and at least
  twice the ensemble's standard error, and the mean 10% quantile is above -1.5%. Each trade 5% of the bot's
  capital, at most 10 open, at most 20% in one name.
- **Learning from punishment:** walk-forward, refit every month on all bars before it; each trade's realized net
  result is a training example, losses included at full weight.
- **Gate (holdout, read once):** mean daily net P&L above zero with t >= 3, the DSR at N >= 0.95, and a better
  result than the stage-1 patterns traded mechanically (else the network adds nothing and the mechanical rule is
  the candidate).

## Stage 3: forward on the demo (only for what passes stage 2; not a trial, a holdout)

- Its own book of R$ 100,000 fictitious on XP's demo, its own magic number, apart from the Composite's executor.
- Fixed guardrails outside the learning: demo account only, flat by 17:40, stop for the day at -1% of the bot's
  capital, at most 40 trades a day, the spread rule, the `PARAR` file.
- The Sentinela's items for the universe enter as event flags from this stage on (forward only; no history).
- Nightly update of the ensemble with the day's trades.
- **Milestones, fixed now:** 3 months: realized costs within twice the model's, else stop and fix. 6 months: net
  P&L above zero, else half size. 12 months: net P&L with t >= 2, else stop. No real money before the 12-month
  milestone, and then only by a new decision of the user's.

## Not in scope

BDRs, short selling, leverage, an LLM or a vision model inside the decision loop, quantum methods (the QML reading
failed the gate, DECISOES 339; a comparison arm may be proposed only after stage 2 passes), the real account.

## Prior, stated before the data

Most likely reading: no stage-1 pattern survives the cost on the holdout. Of the three, (a) has the best published
support (short-horizon reversal after liquidity shocks) and (b) the best fit with what this house already reads.
The chance that stage 3 reaches its 12-month milestone is small; the reason to run it is that it is a new
hypothesis, cheap on the demo, and stage 1 can end it in days.

## Compute

Stages 0 and 1 on the laptop (bars streamed name by name; RAM is the constraint). Stage 2 on Kaggle's free GPU.

## Amendment before any number (2026-11-12, stage 0 findings; no return computed)

- The lake's CVM filings (`sentiment/ipe/ipe.parquet`, 749,165 rows, 2003 to 2026-11-07) carry a delivery date and
  no time of day. **Pattern (b) is dropped unread**, as stage 0 provided; stage 1 has two event studies and N goes
  126 -> 128. Filing dates still serve the "no filing" conditions of (a) and (c): a filing dated the same day (for
  (a)) or dated the previous session's day or the same day (for (c)) excludes the event.
- **Stage 1 runs on M15 bars for every name**, so all names share the same span (2021-11-11 on); M5 reaches only
  about 2022-03 on the liquid names and is kept for stage 2. On M15: (a) the 30-minute return is two bars, entry at
  the next bar's open, exit at the open four bars later or at the 17:30 bar's close; (c) entry at the open of the
  10:15 bar, exit at the open of the 11:30 bar.
- The house universe (`universe_monthly`) holds stocks and units only. A day without a trade counts as zero volume
  in the 63-session median.
