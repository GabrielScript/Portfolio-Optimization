# QPO over the full universe with value in mu — conditional design (attempt 48)

_2026-10-27, session 55. Written and committed while attempt 47 (`2026-10-27-qubo-universo-design.md`) was running
and **before its output was read**, so nothing of that curve shaped this design. DECISOES 156._

## 1. When it runs

Only if attempt 47 is **not** a candidate. Its stopping rule allows a next attempt only with new information in mu.
If 47 is a candidate, this design is not run (no trial row). If it runs, one row is added to `docs/tentativas.csv`
in the commit that triggers it (N 47 -> 48), before the run.

## 2. The new information, and why value (decided without the 2010-2026 test period)

The QUBO's mu in 47 is the Conservadora score alone (low volatility, dividend yield, 12-1 momentum). Value is the
candidate chosen by evidence that does **not** overlap the test period (2010-2026):

- JKP Global Factor Data (Jensen, Kelly & Pedersen 2023; CC BY-NC 4.0, used as a research prior only), monthly
  capped-value-weighted theme returns: **Brazil before 2010**, value t = 2.84 (12.5%/yr, 184 months), the strongest
  of the 13 themes; emerging markets before 2010 t = 4.44; world ex-US (all years) t = 4.90.
- Profit growth was second (Brazil pre-2010 t = 1.77). It has no signal in this project yet, so it would need its own
  IC test first: not part of this attempt.
- Value already exists here as a signal that passed the project's IC gate (DECISOES 79: Magic Formula and Trending
  Value; their 2012-2026 ICs overlap the test and are not used to choose or weight anything below).

## 3. The only change against attempt 47: the score that feeds alpha

At each decision month m, over the same candidates as 47 (names with a Conservadora score, a sector and one class
per issuer):

1. `z_c = escore_normal(conservadora score)` and `z_v = escore_normal(value score)`, where the value score is
   `nota_media_dos_escores(Magic Formula, Trending Value)` (`core/views.py`, the S3 definition: the mean of the two
   normal scores, with the ones available), from the frozen CVM folder `docs/dados/2026-10-24-cvm/` (DFP and ITR
   since 2010, each value dated by its CVM receipt), point in time.
2. Composite score = the mean of `z_c` and `z_v` where a name has both, `z_c` alone where it has no value score
   (same rule as `nota_media_dos_escores`).
3. Alpha as in 47: `IC0 x vol x z`, z = the composite's cross-sectional z-score, **IC0 = 0.059 unchanged** (the
   Conservadora's 1998-2007 IC; no IC is estimated on the test period, and no weight between the two signals is
   fitted: equal weights).

Everything else is attempt 47's: candidates, k = 15, lambda = 1, Ledoit-Wolf, at most 4 picks per sector, annealing
(two runs, agreement >= 80%), max-Sharpe weights and caps, engine, phases, costs, frozen price folder, gates
(P[> Conservadora] >= 0.95, P[> BOVA11] >= 0.95 and DSR >= 0.95 in all 3 phases, N = 48), reported lines (the
user's 90%-of-years target among them), and the stopping rule (one run; if it fails, the next attempt needs another
new signal that passed its own IC gate).

## 4. Expectation, written before

Not a candidate is still the likelier outcome: value adds information, but the Conservadora score already holds the
dividend-yield part of value, the 20% name cap and max-Sharpe weights limit how far mu can move the portfolio, and
P >= 0.95 against the Conservadora over 17 years needs a large edge (DECISOES 153: about 4x the first QPO's).
