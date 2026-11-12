# Stock lending: income on the book we hold, and the loan market as a signal (pre-registered 2026-11-12)

Written and committed before any return-based number. Branch `feat/aluguel`. The user asked for it after the
question "is there any return left to extract": every selection rule tried since October ties or loses to the
Composite, so the two places left are money the book does not collect and information no score of ours reads.

## What exists (looked at on 2026-11-12, no return involved)

- The lake keeps two of B3's daily bulletin tables: `bdi_BTBLoanBalance` (per ticker and market: contracts and value
  of the day, minimum / average / maximum rate of the lender and of the borrower) and `bdi_BTBLendingOpenPosition`
  (per ticker: shares on loan, average price, balance).
- **25 sessions only, 2026-10-05 to 2026-11-10.** B3's bulletin serves the last 21 days and the lake started keeping
  them in October. There is no history to backtest on.
- The rates read as a fraction a year (PETR4 0.0020, MGLU3 0.0679, CSNA3 0.1036 on 2026-11-10): an assumption,
  to be checked against B3's own page for one ticker before any number is written.

## Part A: the income (a measurement, not a trial; N does not move)

For the Composite's current 15 names and weights (the live table of the app), on each of the 25 sessions: the
lender's average rate of each name, the book's weighted rate, and the share of the book that had any loan contract
that day. Reported as a gross yearly rate on the book and its range over the sessions.

What the measurement cannot give, and the report must say: the broker keeps part of the rate and its split is the
user's contract (to be read on the broker's statement); a share on loan cannot be sold until it is recalled
(B3's term for a recall, to be confirmed), which matters on a rebalance day; the tax treatment of the income (to be
confirmed with the user's accountant, not assumed here). The number is an upper bound on what lending could add,
not a promise.

## Part B: the signal (needs history first)

Three scores, fixed here, each known at the close of the last session of a month for the next month:

- `fee`: minus the borrower's average rate (value-weighted across markets) over the month's last 5 sessions.
- `short`: minus shares on loan over the name's mean daily traded shares of the last 21 sessions (days to cover).
- `squeeze`: minus the 21-session change of `short`.

A high rate or a large short position is read as informed selling; the sign is fixed now and is not flipped after
the reading.

**Step 0, data.** A source with at least eight years of per-ticker open positions and rates (B3's historical
lending files, or a vendor the user already pays). If none is found, part B is not read on the past: the lake keeps
accumulating, the three scores are logged monthly from 2026-11, and the reading happens when there are 36 months,
or never. No look at partial returns in between.

**Stage 1, if history exists.** The house IC harness (`avaliar_valor.ics_e_cobertura`): monthly Spearman IC against
the month's total return in the month's universe; gate IC >= 0.02 and t >= 2 over the whole span; subperiods and
the rank correlation with the Conservadora, value and profit-growth scores as description.

**Stage 2, only for scores that pass stage 1.** Two arms per passing score against the Composite, three phases, the
house gate (every phase, all >= 0.95: P[> Composite], P[> BOVA11], DSR): the score as one more equal component of
the Composite, and the Composite without the names in the worst decile of the score (a veto). Each arm read once;
N grows by the number of arms read.

## Prior, stated before the data

Part A: a few tenths of a point a year on a book of low-volatility, high-yield names (those are cheap to borrow);
worth knowing, not a change of scale. Part B: the published evidence for lending fees as a predictor is strong in
the US and thin in Brazil; the binding problem here is step 0.
