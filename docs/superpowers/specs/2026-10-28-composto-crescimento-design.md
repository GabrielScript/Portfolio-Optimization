# Conservadora + value + profit growth composite, top 15 equal weight — design and pre-registration (attempt 51)

_2026-10-28, session 56. Written and committed before any code or curve of this attempt exists. User's approval of
"step 1" (Conservadora + profit growth, + value) of the session-56 roadmap. DECISOES 164._

## 1. The rule: the Conservadora with another score

Identical to the Conservadora (`core.sinais.estrategia_conservadora`) except the score that orders the names:

1. **Score of month m:** attempt 49's composite, built by the same code
   (`scripts.qubo_universo_avaliar._value_score(mensal, com_crescimento=True)`): over the names with a Conservadora
   score, the mean of the normal scores of (a) the Conservadora score, (b) value = mean of the normal scores of the
   Magic Formula and Trending Value (S2, DECISOES 79: IC 0.059, t 5.4/5.5), (c) profit growth (dNI, dGP over lagged
   assets, DECISOES 161: IC 0.033, t 3.63), with the ones a name has (the Conservadora's always). Equal weights;
   nothing fitted. All three signals passed their own pre-registered IC gate.
2. **Portfolio:** `core.sinais.carteira_da_nota` with 15 names and 4 per sector (one class per issuer, no name without a
   sector), equal weights; the fund's fixed rules checked with the Ledoit-Wolf covariance of the training window; a
   violation is a refusal (the engine keeps the current portfolio), as the Conservadora.

So any difference from the Conservadora comes from the two added signals.

## 2. Everything else is fixed

Engine `walk_forward_dinamico`, `CFG_BT` (1,260-session training window, rebalance every 63 sessions, 10 bps per side),
phases 0/21/42, prices cut at 16/10/2026, frozen folders `docs/dados/2026-10-22/` and `docs/dados/2026-10-24-cvm/`,
risk-free = the frozen daily CDI (`docs/dados/2026-10-21-macro/`). The Conservadora row must reproduce the published
rerun (`check_plumbing`) before the arm is read. No network.

## 3. Gates (read once) and what follows

One trial row: N 50 -> 51. Candidate only if, in all 3 phases, P[> Conservadora] >= 0.95, P[> BOVA11] >= 0.95 and
DSR >= 0.95 (N = 51). Reported, not gates: CAGR, MDD, annual turnover of both, refusals, the year table with 2025-2026,
years beating BOVA11 and the Conservadora.

- **Candidate:** pre-register a forward paper log (paper-v3); the composite also becomes a sleeve of the Monte Carlo
  bench (step 2).
- **Not a candidate:** the composite line is **closed**: no Conservadora + growth-only arm, no other weights, no
  "without value". The signals stay as display/allocation inputs; a new composite needs a new signal that passed its own
  IC gate. The bench then uses only the Conservadora, BOVA11 and the CDI.

The script saves the three phases' daily curves (arm, Conservadora, BOVA11) to
`docs/dados/composto-curvas-2026-10-28.parquet` with a `SHA256SUMS` line, as input for the bench.

## 4. Expectation, written before

Not a candidate is the likelier outcome. Attempt 49 ran this same score through the QUBO selection and showed a
phase-0-only edge (P[> Conservadora] 0.80/0.50/0.43); a top-15 equal-weight version of the same score is likely the same
shape. Value and profit growth change with quarterly filings, so the arm's turnover is expected above the
Conservadora's.
