# Net payout yield in the Conservadora (NPY) — design and pre-registration (IC test, then attempt 52)

_2026-10-29, session 59. Written and committed before any code, IC or curve of this signal exists. User's /goal
(step 1 of: NPY, CDA best ideas, PTAX + H2). DECISOES 169 (the one real deviation of our Conservadora from the paper)
and 170 (the 1000x share-count slips fixed, which unblocked it)._

## 1. The signal (fixed now)

Blitz and van Vliet (2018, p. 4): "total net payout yield (NPY) to shareholders (Boudoukh et al., 2007). This
shareholder yield consists of dividend yield and the net change in shares outstanding (calculated as the latest level
divided by the 24-month average)." For security s of issuer i and the signal of month m:

- `DY(s, m)` = `core.sinais.dividend_yield_12m` (unchanged: payouts with ex date in m-12 .. m-1, each over its ex-day
  close).
- `S(i, t)` = `shares_today_units` of issuer i in row t of `docs/dados/2026-10-29-cvm/market_caps.parquet`
  (ON + PN shares, split-adjusted, so a buyback is a fall and an issue a rise; row t uses only documents received
  before t, lake README and `b3lake.fundamentals`). The issuer is `core.risco.emissor(s)`.
- `issuance(i, m) = S(i, m) / mean(S(i, m-23) .. S(i, m)) - 1`, only when all 24 rows exist and are > 0;
  otherwise `issuance = 0` (the name keeps its DY: the candidate set stays exactly the Conservadora's, so the only
  change is the signal). Coverage is reported.
- `NPY(s, m) = DY(s, m) - issuance(i, m)`: an issue lowers it, a buyback raises it.

**NPY-Conservadora score:** `core.sinais.conservadora` with `NPY` in the place of `DY` (low-vol half first, then the
mean of the percentile ranks of NPY and 12-1 momentum inside each half). Nothing else changes; no clip (ranks bound the
effect of an outlier).

Known data limits, accepted and stated: the lake repairs 1000x runs using later months (README; 86 universe
issuer-months were moved on 29/10), and IPO months have no 24-month history (issuance = 0 there by the rule).

## 2. Stage A: IC test (read once)

Monthly Spearman IC against the month's total return in the month's universe (`scripts.avaliar_conservadora`
harness: `retornos_no_universo`, `ic_por_mes`), frozen folder `docs/dados/2026-10-22/`, **2012-06 to 2026-09** (the
first month with 24 rows of counts: 2010-07 .. 2012-06).

- **Gate, both required:** (1) the project's S2 gate, NPY-Conservadora mean IC >= 0.02 and t >= 2; (2) NPY must not
  be worse than DY: mean of the monthly paired difference (NPY-Conservadora IC minus DY-Conservadora IC, same
  months) >= 0. The low-vol half carries most of the IC, so (1) alone would pass almost by construction.
- Reported, not gates: the DY-Conservadora IC on the same months; the paired difference's t; NPY alone vs DY alone;
  IC by subperiod (2012-16, 2017-21, 2022-26); share of universe name-months with a 24-month issuance; mean
  |issuance|; correlation of the two scores.
- **Fail:** stop. No attempt row; DY stays; NPY is display only.
- **Pass:** stage B runs as written below, with no change, after its own commit adding the trial row.

## 3. Stage B: attempt 52 (read once, only after stage A passes)

`NPY-Conservadora`: `core.sinais.carteira_da_nota` on the NPY-Conservadora score, 15 names, 4 per sector, one class
per issuer, equal weights, the fund's fixed rules checked with the Ledoit-Wolf covariance of the window (violation =
refusal, the engine keeps the current portfolio). Engine `walk_forward_dinamico`, `CFG_BT` (1,260-session window,
every 63 sessions, 10 bps a side), phases 0/21/42, prices cut at 16/10/2026, frozen `docs/dados/2026-10-22/`, CDI from
`docs/dados/2026-10-21-macro/`, counts from `docs/dados/2026-10-29-cvm/`. The Conservadora row must reproduce the
published rerun (`check_plumbing`) before the arm is read. No network.

The trial row is added in the stage-B commit, made after stage A passes and before stage B's reading: **N 51 -> 52**
(the `docs/tentativas.csv` header counts strategies evaluated out of sample; an IC test is not one, as in DECISOES
157/161). **Gate:** in all 3 phases P[> Conservadora] >= 0.95, P[> BOVA11] >= 0.95 and DSR >= 0.95 (N 52) —
`scripts.qubo_universo_avaliar.strategy_gate`. Reported: CAGR, MDD, turnover, refusals, year table with 2025-2026.

- **Candidate:** pre-register a forward paper log against the same-day Conservadora (the paper-v3 pattern).
- **Not a candidate:** NPY is closed as a Conservadora input; the Conservadora keeps DY.

## 4. Expectation, written before

Brazilian issuers issue shares far more often than they buy back, so the issuance term mostly demotes issuers. Our DY
Conservadora's composite already loads on value and quality, and the paper's NPY gain over DY is small; expected: gate (1)
passes (the score keeps the low-vol half, which carries most of the IC), the paired difference is near zero with
either sign (so gate (2) is a coin flip), and if stage B runs it is not a candidate (P[> Conservadora] well below 0.95
in at least one phase).
