# The Composite with residual momentum (pre-registered 2026-11-16)

The user asked whether any regression or econometric technique could help Aleph Quant; the answer (DECISOES 374 ended
the model line: Ridge > trees > nets on the same six inputs) ranked residual momentum first among untested signal
changes, and the user approved ("Pode registrar a especificação e rodar"). Written and committed before any number of it
is read. Branch `feat/momentum-residual` from `feat/composto-faixa` (today's Composite code plus the same-run replay).

## The literature

Blitz, Huij and Martens (2011, J. Empirical Finance; Blitz, Hanauer and Vidojevic 2020 for emerging markets): momentum
measured on the part of returns that factor exposures do not explain earns about the same premium as total-return
momentum with about half its volatility and much smaller crashes, because total-return momentum loads on the factors
that just did well and reverses with them.

## The rule (one arm, `Composite-resmom`)

Everything as the Composite today (engine, data, Conservadora, value and growth legs, 15 names, 4 per sector, equal
weights, limits, 63 sessions, 10 bps), except the momentum inside the Conservadora's within-half rank: `momentum_12_1`
is replaced by `momentum_residual`, for decision month m and each candidate i:

1. Market = the equal-weighted mean monthly total return of the month's candidates (the names the Conservadora scores
   that month, the liquid universe), per month, over the names with a return that month.
2. OLS of i's monthly total return on a constant and the market over the 36 months m-36 .. m-1 (the window the
   Conservadora's 36-month volatility already requires complete; no new coverage loss).
3. Residuals of that fit over the 11 months m-12 .. m-2 (the 12-1 window); score = their sum divided by their sample
   standard deviation (Blitz-Huij-Martens' scaling). A name without the 36 complete months has no score, as today.

Only the market factor: the B3 has no point-in-time size and value factor series in the lake, and building them would
be a second design choice. The Conservadora's rule is otherwise unchanged (low-vol half + mean of the dividend-yield
and momentum percentile ranks within the half).

Code: `core.sinais.conservadora` gains a keyword `momentum` defaulting to `momentum_12_1` (the official Composite and
every earlier reading unchanged; existing tests keep guarding it); `core/momentum_residual.py` holds the new signal.

## Reading

`scripts/momentum_residual_avaliar.py`: phases 0 / 21 / 42; the Composite replayed in the same run by today's code
(its gap to attempt 51's saved curve printed, not enforced, as in the 2026-11-15 band addendum); the arm next to the
Composite, the Conservadora, BOVA11 and the CDI; turnover and names shared with the Composite. N for the DSR = 194
(193, DECISOES 374, + this arm). **Gate: P[> Composite] >= 0.95 and DSR >= 0.95 in every phase.** If it passes, a
forward ghost account only; the official book stays the pure Composite.

Descriptive, no gate: the monthly Spearman IC of each momentum alone (12-1 and residual) against the month's total
return in the month's candidates, 2010-02 to the last month, with Newey-West t (2 lags), and their rank correlation.

## Expectation (before the run)

Momentum is half of the within-half rank, the Conservadora one of the score's three legs, so the arm differs from the
Composite by one or two names per decision. Expected CAGR within about a point of the Composite either way, a slightly
smaller worst drawdown, and no pass (about 15%). Residual momentum's own IC near total-return momentum's.
