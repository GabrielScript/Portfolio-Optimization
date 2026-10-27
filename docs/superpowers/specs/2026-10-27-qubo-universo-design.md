# QPO over the full universe (QUBO selection by annealing) — design and pre-registration

_2026-10-27, session 55. DECISOES 153 (direction) and 155 (this design). Branch `feat/qubo-universo`, worktree
`../po-qpo`. Committed before any out-of-sample curve of this arm exists._

## 1. Why this test, and the honest expectation

The first QPO reading (DECISOES 147) chose k = 15 of the Conservadora's **top 20** names. The user called this the
crucial error: the QUBO was an overlay on the Conservadora's ranking, not an independent strategy. The 20 came from
a quantum-simulator memory limit, so the solver set the question (DECISOES 153). This test asks the user's question:
**does the same QUBO, choosing among every eligible name of the lake's point-in-time universe, beat the Conservadora
and BOVA11?**

- **Feasible:** at n = 67-230 candidates, C(178, 15) = 2.4e21 subsets. No simulator (state vector, weight-k
  subspace, GPU or TPU) reaches that, and no NISQ device gives a useful answer on a dense 150-variable QUBO. The
  QUBO is solved by **simulated annealing**, the classical counterpart of quantum annealing, with two independent
  runs per rebalance as the check. Nothing here runs on quantum hardware; "QPO" names the model.
- **Expectation, written before running:** likely **not** a candidate. Errors in expected returns cost about 10x
  errors in variances (Chopra & Ziemba 1993), and the alpha is still the Conservadora score, so the arm adds a
  covariance-aware selection on the same information. The first reading's selection effect was +0.6/+0.2/-1.3 p.p.
  (P 0.647/0.488/0.152). A wider candidate set gives the selection more room, in both directions.
- **Stopping rule (fixed now):** one run, read once. If it fails, the QPO line is closed. The next attempt needs
  **new information** in mu (a new signal that passed its own IC gate), never a re-parametrisation of this one.
  The user's target "beat BOVA11 and the Conservadora in 90% of the years" is **reported** at every reading and is
  never a gate or a tuning target: chasing it on 17 years of history is fitting the past (the Conservadora itself
  beat BOVA11 in 12 of 17 years).

## 2. What changes against the first QPO reading (and nothing else)

| Item | First reading (DECISOES 144-147) | This test |
|---|---|---|
| Candidates | top 20 by score (one class per issuer, with sector) | **every** name with a score, a sector and one class per issuer (`core.qpo.candidates` with `FULL_UNIVERSE`) |
| z-score of the score | within the 20 | within all candidates (the textbook Grinold-Kahn cross-section) |
| Sector rule in the selection | none (the weights' 30% cap) | **at most 4 picks per sector** (as the Conservadora's `carteira_da_nota(nota, setores, 15, 4)`) |
| Solver of record | enumeration (`solve_exact`, n <= 28) | **simulated annealing** (`core.qubo_universo.anneal_solver`): two independent runs, lower energy wins |
| Quantum solver | QAOA on Kaggle (fidelity gate) | none: annealing is the quantum-inspired solver |

Unchanged: k = 15; lambda = 1; IC0 = 0.059; `alpha = IC0 x vol x z`; Ledoit-Wolf Sigma of the 1,260-session window;
`Q = (lambda / (2 k^2)) Sigma - diag(alpha) / k`; max-Sharpe weights inside the picks (`CFG_MAX_SHARPE`: 20% per
name, vol 40%, sector 30%, rf = CDI of the date, `checar_carteira`, `CarteiraRecusada`); `walk_forward_dinamico`,
`CFG_BT` = 1260 / 63 / 10 bps, `FASES = (0, 21, 42)`, `Congelamento()`; the frozen folder `docs/dados/2026-10-22/`
(hash-checked), prices cut at 16/10/2026; Selic and CDI as of 27/10/2026.

Why the sector cap is part of the model: with the top 20 the picks were already spread, but over the whole universe
the score's low-vol half and dividend-yield rank lean to a few sectors. On the spike's first date the uncapped
problem's solution filled a sector to the cap of 4 (picks only, no return looked at). Without the cap the max-Sharpe
step would have to squeeze, say, 8 utilities into 30%, and the arm would lose for a reason unrelated to selection.
The cap also keeps the comparison with the Conservadora like for like.

Negative alpha: about half the candidates (the high-vol half) get a negative alpha (51% on the spike's first date),
so the QUBO rarely picks them. That is the model working: "full universe" means every eligible name is a candidate.

## 3. Solvers: what was measured before this commit (no return was computed)

- **Oracle, n = 20:** on all 198 recorded QUBOs of the first reading (k = 15, no sector cap), the SCIP MIQP
  (`solve_miqp`: binary x, `sum x = k`, sector rows, epigraph `t >= x'Px` with `P = S - lambda_min(S) I`,
  `S = (lambda / (2 k^2)) Sigma`, the identity share of the diagonal moved into the linear term) reproduced
  `exact_energy` 198/198, all proven optimal, 0.06 s each. The annealer matched 198/198 (1.1 s each).
- **Full universe (spike, 4 dates: 2011-03, 2015-06, 2020-06, 2026-06; n = 67, 108, 128, 159):** SCIP 10.0 cannot
  certify optimality at this size. The plain dense form did not finish 4 solves in 30 minutes; with the tightened
  split, the sector rows and a feasible start, the smallest date (n = 67) stopped at the 600 s limit with a 79.9%
  gap. Four independent annealing seeds found the **same picks** on all 4 dates, 0.8 s each.
- **So:** the solver of record is annealing. "Exact" is not claimed at full size; agreement of independent runs
  is the check. `solve_miqp` stays as the verified n = 20 oracle.

Annealing (fixed): swap moves (one pick out, one name in) keep `|x| = k` and the sector caps by construction;
Metropolis acceptance; geometric cooling from the mean |dE| of 256 random swaps to 1e-3 of it over 20,000 steps;
best of 8 restarts from random feasible starts; energies re-scored exactly. Per rebalance: seeds = the decision
date's ordinal and the next integer; the lower energy wins; agreement = equal energies (to 1e-12).

## 4. Arms and trials

| Curve | Role | `docs/tentativas.csv` |
|---|---|---|
| QUBO-universo: annealing picks, max-Sharpe weights | **the strategy trial (gated)** | **+1 row: N 46 -> 47** |
| QPO-exact (top-20) | context: P[QUBO-universo > top-20] answers "did removing the nesting help?" | existing row |
| Conservadora (15, equal weights) | reference (published curve; the plumbing check) | reference |
| BOVA11 | reference | reference |

## 5. Gates (checked once)

- **Strategy gate:** QUBO-universo is a candidate only if, in **all 3 phases**, P[> Conservadora] >= 0.95 **and**
  P[> BOVA11] >= 0.95 (`psr_da_diferenca`, arithmetic) **and** DSR >= 0.95 (`tabela_do_periodo`: the IR against
  BOVA11, N = 47).
- **Solver check:** the two annealing runs agree on >= 80% of all rebalances (the QAOA gate's match threshold).
  Below that, the reading says the picks are not reproducible and the arm is not a candidate whatever its curve.
- **Reported, not gates:** years beating both BOVA11 and the Conservadora (the user's 90% target), the year table,
  CAGR, MDD, Sharpe over CDI, IR, MinTRL, P[> top-20], agreement per phase, candidates per rebalance, share of
  negative alpha, the fullest sector, refusals.
- **Reading:** candidate -> a forward paper log (paper-v3) is pre-registered next; nothing goes to real money before
  it. Not a candidate -> the QPO line is closed (stopping rule, section 1).

## 6. Run

`python scripts/qubo_universo_avaliar.py > docs/qubo-universo-2026-10-27.txt`, once, after this commit. The script
stops before any arm if the Conservadora's CAGR does not reproduce the published one (`check_plumbing`).
