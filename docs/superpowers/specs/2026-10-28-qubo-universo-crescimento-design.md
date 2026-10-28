# QPO over the full universe with value and profit growth in mu — design and pre-registration (attempt 49)

_2026-10-28, session 55. Written and committed after the profit-growth IC test passed
(`docs/crescimento-lucro-2026-10-28.txt`: IC 0.033, t 3.63) and before any code or curve of this attempt exists.
Triggered by section 4 of `2026-10-27-crescimento-lucro-design.md` (DECISOES 157). DECISOES 161._

## 1. The only change against attempt 48: a third signal in the score

At each decision month m, over the same candidates as 47-48 (names with a Conservadora score, a sector and one class
per issuer):

1. `z_c`, `z_v` as in attempt 48 (Conservadora score; value = `nota_media_dos_escores(Magic Formula, Trending Value)`);
   `z_g = escore_normal(profit growth)`, profit growth = `core.crescimento_lucro.nota_de_crescimento` (mean of the
   normal scores of dNI and dGP over lagged total assets), all from the same frozen folders.
2. Composite = the mean of the normal scores a name has among `z_c`, `z_v`, `z_g` (`z_c` always present). Equal
   weights: nothing fitted, no IC estimated on the test period.
3. Alpha as in 47-48: `IC0 x vol x z`, z = the composite's cross-sectional z-score, **IC0 = 0.059 unchanged**.

Everything else is attempt 48's (k = 15, lambda = 1, Ledoit-Wolf, at most 4 picks per sector, annealing with two runs
and the 80% agreement check, max-Sharpe weights and caps, engine, phases, costs, frozen folders). One trial row:
N 48 -> 49.

## 2. Gates (read once) and what follows

Candidate only if, in all 3 phases, P[> Conservadora] >= 0.95, P[> BOVA11] >= 0.95 and DSR >= 0.95 (N = 49), and the
annealing runs agree on >= 80% of rebalances. Years beating BOVA11 and the Conservadora: reported, not a gate.

- **Candidate:** pre-register a forward paper log (paper-v3); nothing goes to real money before it.
- **Not a candidate:** the QPO line is **finished**. No further mu attempt inside the QPO; a signal that passes its
  own IC gate goes to the Conservadora/allocation work instead (profit growth is already such a signal).

## 3. Expectation, written before

Not a candidate is still the likelier outcome: profit growth adds information (correlation 0.14 with the Conservadora,
0.25 with value), but beating the Conservadora with P >= 0.95 in all three phases over 17 years needs a large and
stable edge; 48 showed a phase-0-only edge.
