# The Composite with sector-relative value and growth (pre-registered 2026-11-16)

After DECISOES 375 the user asked what the "sector-relative" test was and approved it ("Pode rodar o setorial e o
painel amplo"). Written and committed before any number of it is read. Branch `feat/composto-setorial` from
`feat/momentum-residual` (today's Composite code, the same-run replay reading).

## The literature

Cohen and Polk (1998), Asness, Porter and Stevens (2000), Novy-Marx (2013): value and profitability measured against
the firm's own industry predict returns better than raw measures, because structural differences between industries
(banks' book values, utilities' payouts) are noise for a firm-level bet.

## The rule (one arm, `Composite-setorial`)

Everything as the Composite today (engine, data, the Conservadora leg, 15 names, 4 per sector, equal weights, limits, 63
sessions, 10 bps), except the value and growth legs' raw scores. For each decision month and each of the three raw
scores (Magic Formula and Trending Value, which make the value leg; profit growth, the growth leg), among the names
with that score that month:

1. The score becomes the name's percentile rank (`rank(pct=True)`, ties at the mean) **within its sector** (the same
   `dados.setores` the 4-per-sector cap uses);
2. a sector with fewer than 5 scored names that month keeps the name's percentile rank among all scored names;
3. a name without a sector keeps the all-names percentile rank (the cap skips it anyway).

The legs are then built exactly as today (`nota_media_dos_escores` on the two value scores; the Composite's normal-score
mean over the three legs). The Conservadora leg (low volatility, dividend yield, 12-1 momentum) is unchanged: it is a
risk-and-payout rule, not a valuation.

## Reading

`scripts/composto_setorial_avaliar.py`: phases 0 / 21 / 42; the Composite replayed in the same run; the arm, the
Conservadora, BOVA11, the CDI; turnover and names shared. N for the DSR = 195 (194, DECISOES 375, + this arm). **Gate:
P[> Composite] >= 0.95 and DSR >= 0.95 in every phase.** If it passes, a forward ghost account only.

Descriptive, no gate: the monthly Spearman IC 2012-01 to the last month of the value leg and of the growth leg, raw and
sector-relative, with Newey-West t (2 lags).

## Expectation (before the run)

About 10-15% to pass. The sector cap already removes much of the sector bet (it binds in 51 of 66 decisions, DECISOES
347), so the arm should differ by one or two names per decision; small sectors (2-4 names) fall back to the global
rank. Residual momentum (375) showed that removing the common part of a B3 signal can remove signal.
