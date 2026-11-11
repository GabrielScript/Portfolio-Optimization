# XQuad (Quip Network) as the QUBO solver of the full-universe Composite selection

Date: 2026-11-11. Branch `feat/quantum-xquad`, worktree `../po-quantum`. The user: "Vamos testar o quip network e o
xquad com nossos dados ... para cada um dos anos entre 2010-2026 ... com todos os ativos da B3 ... ao lado da
composite". **Pre-registration: committed before any number of this test exists.**

## What changes, and only that

Carrier: attempt 56, `QUBO-universo-crescimento` (`scripts/qubo_universo_avaliar.py crescimento`): every 63 sessions,
the QUBO of k = 15 out of **every eligible name** of the point-in-time universe (median ~118 candidates, 55-170),
alpha from the Composite's score (Conservadora + value + profit growth, equal weights), Ledoit-Wolf covariance,
lambda 1, at most 4 picks per sector; max-Sharpe weights on the picks (20% name cap, vol 40%, rf CDI). Frozen exports
`docs/dados/2026-10-22` (to 16/10/2026), phases 0, 21, 42, 10 bps, `walk_forward_dinamico` with `Congelamento()`.

Only the solver changes. Attempt 56 solved the capped QUBO with the project's own annealer (`core.qubo_universo`).
Here the same instances are exported and solved by:

1. **XQuad-CPU** (headline): `xqsa.SolverCPU` (xquad 0.5.2, simulated annealing, num_reads 200, seed = the
   decision date's ordinal) on an unconstrained binary model: x'Qx + A (sum x - 15)^2 + A sum_sectors (sum_{i in s}
   x_i + slack_s - 4)^2, slack_s on 3 bits weighted 1, 2, 1 (0..4); A = 2 x the largest absolute row sum of Q.
   Coefficients scaled so the largest is 10^6 milli units (the solver's integer grid). An infeasible best sample
   (count != 15 or a sector above 4) is repaired greedily: drop the highest-energy pick or add the lowest-energy
   name until feasible; the share repaired is reported.
2. **XQuad-Quip** (descriptive, phase 0 only): the same model submitted to the Quip test network (`aglais`) with
   `xqsa.quip.SolverQuip`, native topology, faucet-funded test tokens. If the network refuses or does not answer a
   decision within 15 minutes, that decision falls back to XQuad-CPU and is counted; with more than 20% fallbacks the
   arm is reported as not run.

Python 3.13 is required (the app stays on 3.12): the solve runs in a Linux container (`quantum-lab/Dockerfile`,
`python:3.13-slim`, xquad 0.5.2); instances go out and picks come back as files. The engine replays the picks with
`core.qpo.recorded_selector`.

**Classiq is not in this test.** Its QAOA runs one qubit per binary variable and its synthesis needs a Classiq account
login: ~118-170 variables per decision are beyond any simulator here (statevector ~2^25 at most) and beyond a
free-account hardware budget. Reported as a limit, not as a result.

## Anchors

- The exported instances, solved by the project's annealer, reproduce attempt 56's picks at every decision.
- The Composite curve is attempt 51's (`docs/dados/composto-curvas-2026-10-28.parquet`).

## Gate (fixed now)

House rule since DECISOES 284: the benchmark is the Composite alone. **Candidate** only if, in all three phases,
P[XQuad-CPU > Composite] >= 0.95 and its DSR >= 0.95 (N from `docs/tentativas.csv` plus this row). Otherwise the
quantum-solver line on the B3 stays closed.

## Reported

Calendar-year returns 2010-2026 of XQuad-CPU, XQuad-Quip (phase 0), attempt 56 (own annealer), the Composite,
BOVA11 and the CDI; per phase CAGR, volatility, Sharpe and Sortino over the CDI, MDD, Calmar, alpha and its Newey-West
t against the Composite and against BOVA11, P[> Composite], DSR; share of decisions where XQuad picks equal attempt
56's; energies against attempt 56's (same QUBO).

## Expectation (registered)

Attempt 56's two annealing runs agreed at 100% of decisions, so its picks are very likely the optimum of each QUBO.
XQuad-CPU should find the same picks in >= 90% of decisions and land within +-0.5 point of attempt 56's CAGR (20.1%,
16.5%, ~16% by phase), below the Composite in phases 21 and 42. **Fails the gate.** A better solver cannot add alpha
the objective does not contain (DECISOES 318-326, 339).

## Trial count

One row (XQuad-CPU). XQuad-Quip is the same configuration on another machine and adds no row.
