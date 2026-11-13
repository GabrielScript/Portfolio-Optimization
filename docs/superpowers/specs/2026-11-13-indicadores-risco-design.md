# The app table's risk and return columns as point-in-time signals (pre-registered 2026-11-13)

Written and committed before any return-based number. Branch `feat/indicadores-risco`, from `main` (`462937d`),
worktree `../po-risco`. The user's question of 2026-11-13: take the table "Todas as ações, BDRs e ETFs" and pick the
names with the best "retorno 12m, retorno 3a, retorno 6a, alfa a.a, melhores beta, Sharpe, Sortino, CVaR, t do
alfa, ROE, patentes"; the analysis and the offer are in DECISOES 357; the user's go-ahead: "Pode rodar por favor".

## Why it cannot be read off the table

The table's columns are computed to today. Choosing names on them and judging the choice on the same history is
look-ahead. The honest question: had the same columns been computed at each past month, with only what was known
then, would they have ordered the next month's returns, and do they say anything the Composite's three components
(Conservadora, value, profit growth) do not already say?

Left out, already read: the 12-month return (the momentum leg of the Conservadora; attempts 50 and 54) and patents
(DECISOES 352 and 354). BDRs and ETFs stay out of this reading: the frozen universe is the house's stock universe
(one ranking with BDRs failed in DECISOES 316, and an ETF has no ROE).

## Data (frozen, no network)

`docs/dados/2026-10-22/` (daily total-return closes, monthly total returns, the monthly point-in-time universe:
median 143 names a month, 36 to 227), `docs/dados/2026-10-24-cvm/` (CVM statements with their receipt dates) and
`docs/dados/2026-10-21-macro/cdi_daily.parquet`. The same files the Composite was registered on.

## Signals, for a decision month M (first day of M; every input is dated strictly before M)

W3 is the last 756 sessions before M, W1 the last 252. A name needs a close on at least 90% of the sessions of the
window (681 of 756, 227 of 252), otherwise it has no score for that signal. The formulas are the table's own
functions, called on the window instead of on today's history (`core.indicadores.desempenho_por_acao`,
`risco_por_acao`, `multiplos`), so the reading is of the columns the user sees.

| Signal | Definition | Higher is better |
|---|---|---|
| `SHARPE` | Sharpe of the daily excess over the CDI, W3 | yes |
| `SORTINO` | Sortino of the same excess, W3 | yes |
| `CVAR` | daily 95% expected shortfall (a positive loss), W3 | no (smaller loss ranks higher) |
| `ALPHA` | annual CAPM alpha against BOVA11, W3 | yes |
| `T-ALPHA` | its Newey-West t, W3 | yes |
| `BETA` | beta against BOVA11, W1 (the table's window) | no (low beta ranks higher) |
| `RET-3y` | total return of the 36 complete months before M | yes |
| `RET-6y` | total return of the 72 complete months before M | yes |
| `ROE` | 12-month net income of the parent over equity less minorities, equity > 0, latest statement received before M (at most 18 months old) | yes |
| `RISK-ALL` | mean of the house normal scores of the nine above, with the ones a name has (at least five of the nine) | yes |

"Melhores beta" is read as low beta (the direction with a published premium); the IC of high beta is the same
number with the sign flipped, so a strongly negative IC answers the other reading. One window per signal, the
table's: no window is searched. Nine signals, not eight as DECISOES 357 says (a miscount there).

## Stage 1: one reading

The house IC harness (`core.poder.ic_por_mes`, the month's score against the month's total return in the month's
universe), 2012-01 to 2026-09, 177 months.

- **Gate (the house's):** mean IC >= 0.02 and t >= 2.
- **New information:** each month the signal's normal score is regressed across names on the normal scores of the
  Conservadora, value and profit growth (a missing component counts as 0, the mean; with an intercept); the IC of
  the residual is read the same way. A signal is a **stage 2 candidate** only if it passes the gate **and** its
  residual IC has mean > 0 and t >= 2.
- Ten readings are taken at once: under no effect about a quarter of a signal is expected above t 2 by chance. A t
  between 2 and 2.8 (Bonferroni for ten) is reported as weak.
- Description, not gate: coverage, the three subperiods (2012-2016, 2017-2021, 2022-2026), the mean rank
  correlation with each of the three components and among the ten signals.

## Stage 2 (only for a candidate; not built unless one exists)

One arm per candidate, `Comp+<signal>-w`: the earned-weight rule of DECISOES 354 (score = z(Composite) +
lambda_M z(signal), lambda_M = clip(mean IC of the signal / mean IC of the Composite, 0, 1) over the months before
M, 0 with fewer than 36), 15 names, at most 4 per sector, house rules, 10 bps, three phases, read once. Gate:
P[> Composite] >= 0.95 and DSR >= 0.95 in every phase. Each arm is one row (N 128 before). With no candidate, N
does not move: an IC reading is not a strategy configuration.

## Prior, stated before the data

- `BETA` and `CVAR`: positive IC, probably through the gate (0.02 to 0.05), because low risk is the strongest leg
  of the Conservadora; residual IC with t below 2.
- `RET-3y` and `RET-6y`: IC within 0.01 of zero, either sign; fail.
- `SHARPE`, `SORTINO`, `ALPHA`, `T-ALPHA` over three years: IC 0.00 to 0.03, at most one through the gate; residual
  t below 2 (they mix momentum and low volatility, both already in).
- `ROE`: IC 0.01 to 0.03, even odds at the gate; the residual is the only one with a fair chance (the value leg has
  EBIT over capital, not ROE), still under one in three.
- `RISK-ALL`: through the gate; residual t below 2.
- Expected number of stage 2 candidates: zero, perhaps one. Expected stage 2 outcome if one exists: a tie.

## Code

`core/indicadores_no_tempo.py` (the table at a month; the residual), `scripts/indicadores_risco_ic.py` (the
reading), `tests/test_indicadores_no_tempo.py`. Nothing the app, the paper logs or the accounts import changes.

## Amendment 1 (2026-11-13, after stage 1 was read, before any stage 2 number)

- Stage 1 (`docs/indicadores-risco-ic-2026-11-13.txt`, commit `d42699c`): anchors exact (0.0854 / 0.0701 /
  0.0332). Nine of ten pass the gate; `BETA` fails (IC 0.027, t 1.58). Residual IC with t >= 2: `RET-3y` 0.029
  (t 2.93), `CVAR` 0.026 (2.53, weak), `ALPHA` 0.022 (2.20, weak), `RISK-ALL` 0.022 (2.11, weak).
- The four are run, as registered: choosing among them now would be a choice made after the data.
- Against my prior: wrong on `RET-3y` (expected within 0.01 of zero; it is 0.066, t 4.84, the one strong
  residual), on the three-year ratios (expected at most one through the gate; all four pass at 0.053 to 0.058) and
  on `BETA` (expected to pass; it fails). Right that `ROE` and the ratios mostly repeat the components (`ROE`
  residual -0.001); wrong that `ROE` was the likeliest to add something.
- What stage 1 also says: `SHARPE`, `SORTINO`, `ALPHA`, `T-ALPHA` and `RET-3y` are one signal (mean rank
  correlation 0.95 to 1.00), so the `ALPHA`, `RET-3y` and `RISK-ALL` (0.94 with `RET-3y`) arms are near duplicates
  and count as three rows of one idea. The residual was taken on the Conservadora's score, a volatility half-split
  plus yield and momentum ranks: part of what is left may be 12-month momentum and low risk at a finer grain
  rather than three-year information.
- Build: `feat/informados` merged into this branch for `core.informados.earned_weight` and `weighted_score`
  (DECISOES 354). Anchor per phase, before any arm: the Composite's replay on its saved curve of DECISOES 350
  (1e-9); the LogDet anchor of 354 is dropped (no determinant arm here). lambda from the monthly ICs of 2012-01 on,
  as in 354, so 0 before 2015-01. DSR at N = 132 (128 + these four rows).
- Arms: `Comp+RET3y-w`, `Comp+CVAR-w`, `Comp+ALPHA-w`, `Comp+RISKALL-w`.
- Prior for stage 2: each arm shares 11 to 13 of 15 names with the Composite; alpha against the Composite within
  1.5 points a year either side; P[> Composite] between 0.2 and 0.9 and not the same side in the three phases for
  at least one arm; no arm passes. If one passes: a forward fictitious account at most, never the book or the
  19/12 buy.
