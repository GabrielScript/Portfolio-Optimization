# Hybrid quantum forecast and QPO, B3 and US — design and pre-registration

_2026-11-01, session 65. Written and committed before any curve or IC of this work exists. User's request: use QPO for
return forecasting and stock recommendation on the B3 and US stocks, backtest it correctly and compare with the Composite
and the Conservadora, from `Artigos_QPO/Quantum_portfolio{,2,3,4}.pdf`. This reopens the line DECISOES 162 closed ("no
further QPO attempt"): the user's call. Sections 1 to 7 are fixed from the commit on._

## 1. What the sources do and do not give

None of the four papers forecasts returns. All take the expected returns as an input.

| Paper | What it does | Out-of-sample evidence | Used here as |
|---|---|---|---|
| Rebentrost & Lloyd 2018 | HHL on the KKT system of equality-constrained Markowitz (eq. 20), shorts allowed; needs qRAM | None; states classical sampling can close the gap | Same optimum as a classical solve (Prop. 3) |
| Kerenidis, Prakash & Szilagyi 2019 | Quantum interior point for long-only Markowitz as an SOCP; output has tomography error delta | None; complexity on 50-100 S&P names | The convex arm's problem (Prop. 3) |
| Mugel et al. 2022 (same paper as `QPO4.pdf`, read in DECISOES 147) | Discrete QUBO, turnover term, VQE on PennyLane, VQE-Constrained, clustering | None: mu_t is that month's return, so its Sharpe of 6-20 has foresight | Turnover term; VQE and VQE-Constrained in the solver bench |
| Hodson et al. 2019 | QAOA, soft penalty vs hard XY mixer, fixed trading cost, long/short | None: each month optimized on that month's own mean | Trading cost; hard vs soft mixer in the bench |

`Quantum_port.txt` is an uncited AI summary, not a source; its "predicted return" mu'w* is the input times the weights.
So forecasting is new work here (branch B of DECISOES 143), judged by its own test. At sizes a simulator reaches, the
quantum solver can at best tie the exact optimum (Prop. 2), so it is checked as a solver, never credited with returns.

**Prior, written now:** no arm passes against the Composite. The Composite is the closest of 53 trials and failed one
phase (DECISOES 164).

## 2. Formal setup

Month t, eligible names U_t. Features x_(i,t) in R^6 read only months before t and statements with `available_from` < t.
Target y_(i,t): the normal score across names of the total return of months t, t+1, t+2. A forecast for month t trains
only on pairs with s <= t - 3, so every label used has ended by month t - 1 (purge, Lopez de Prado AFML ch. 7).

**Prop. 1 (the circuit is a trigonometric polynomial).** Let f(theta) = a + sum_j b_j <0|V_theta^dag U^dag Z_j U V_theta|0>,
V_theta = prod_j RY(theta_j), U any fixed unitary. Since RY(t) = cos(t/2) I - i sin(t/2) Y, each wire contributes only
cos^2(t/2), sin^2(t/2) and cos(t/2) sin(t/2), that is 1, cos t, sin t. So f lies in the span of the 3^6 = 729 products
prod_j b_(k_j)(theta_j), b in {1, cos, sin}. Consequence: least squares on those 729 features has training error at most
the circuit's. The circuit can beat this twin out of sample only through implicit regularization. This is the control
that separates what the circuit adds from what its features already hold (`tests/test_qml_sinal.py` checks the span to
1e-10).

**Prop. 2 (a solver cannot add return).** For a feasible set F and energy E, any distribution p on F has E_p[E] >= E* and
any sampled minimum >= E*. Hence the QAOA or VQE pick never beats the exact one on the objective, and an out-of-sample gap
between them is noise, not skill (rule of DECISOES 144).

**Prop. 3 (quantum convex solvers reach the same optimum).** Rebentrost-Lloyd solve the KKT system of the same quadratic
program and Kerenidis et al. its SOCP form, with output error ||w^ - w*|| <= delta ||w*||. For U(w) = mu'w - (lambda/2)
w'Sigma w, U(w*) - U(w^) <= ||grad U(w*)|| delta ||w*|| + (lambda/2) ||Sigma|| delta^2 ||w*||^2, and U(w^) <= U(w*) when w^ is
feasible. So the classical exact solve is their ceiling and is what the convex arm runs.

**Prop. 4 (turnover in the QUBO).** With x, h binary, tau sum_i (x_i - h_i)^2 = x'(tau diag(1 - 2h))x + tau |h| (Lean,
DECISOES 150-151). So it only changes the diagonal of Q and keeps a weight-k mixer valid.

**tau, derived, not tuned.** Dropping one name for another trades 1/k at c = 10 bps on each side: cost 2c/k. Sum_i (x_i -
h_i)^2 counts 2 per swap, so cost per rebalance = (c/k) sum_i (x_i - h_i)^2. The QUBO's alpha is a yearly rate and there are
f = 4 rebalances a year, so in the objective's units **tau = f c / k = 4 x 0.001 / 15 = 2.667e-4**.

## 3. Forecast test (signal, no N until a portfolio runs)

**Features, the Composite's own inputs:** momentum 12-1; low volatility (negative standard deviation of up to 36 monthly
returns, at least 12, so the US can train before 2019); dividend yield 12m; Magic Formula; Trending Value; profit growth.
Each month, each becomes a percentile u in (0, 1) among the names with it, theta = pi (u - 1/2), and a missing value is
theta = 0. Same inputs for every model.

**Models, each fixed here, one configuration, no search:**

| Model | Definition |
|---|---|
| `VQC` | PennyLane 0.45.1 `default.qubit`, backprop, float64: `AngleEmbedding` RY on 6 wires, `StronglyEntanglingLayers` L = 3 (54 angles), <Z_j> on all wires, linear head (7): 61 parameters |
| `MLP` | 6-7-1, tanh: 57 parameters (the circuit's size) |
| `Fourier` | Ridge on the 729 features of Prop. 1 (the dequantized twin), lambda by GCV on the training window |
| `Ridge` | Ridge on theta, lambda by GCV on the training window |
| `Composto` | Attempt 51's score, nothing fitted: the benchmark |

Torch models: MSE, Adam lr 0.02, batch 1,024, 400 steps on the first month, then 50 steps warm-started from the month
before, 3 seeds averaged. Refit monthly on the last 120 months, at most 20,000 rows drawn by a generator seeded with the
month, the same rows for every model. B3: first refit 2009-11, IC read 2012-01 to the last complete month, as DECISOES 157.
US: first refit 2019-11, IC read from there.

**Reading:** monthly Spearman IC, t by Newey-West with 2 lags (the 3-month target overlaps). **Gate: IC >= 0.02 and t >= 2,
as S2 and DECISOES 161.** Reported, not gates: paired IC difference with its t for VQC - Fourier, VQC - MLP, VQC - Composto.
A quantum edge is claimed only if VQC passes and VQC - Fourier has t >= 2.

## 4. Portfolio arms (read once, three phases)

All use equal weights inside the picks unless stated, because the comparators are equal-weight and the QUBO's objective is
the equal-weight portfolio. (Attempts 47-49 used max Sharpe inside; this differs from them on purpose, written here.)

| Arm | Rule |
|---|---|
| `VQC-15` | VQC forecast ordered through `estrategia_por_nota`: 15 names, 4 per sector, the Composite's rule |
| `Fourier-15` | The same with the classical twin |
| `QUBO-giro` | Composite alpha (`qubo_alphas`, IC0 0.059), full universe, k = 15, lambda = 1, 4 per sector, turnover term with h = the portfolio held and tau above; simulated annealing, two runs, as DECISOES 155 |
| `MV-continuo` | The QUBO's continuous relaxation: max alpha'w - (1/2) w'Sigma w, w >= 0, sum w = 1, 20% per name, 30% per sector, 40% vol; one class per issuer; `core.optimization.otimizar` with `Objetivo.KELLY`, lambda = 1. What Prop. 3's solvers output |

`VQC-15` runs even if VQC fails section 3, because the user asked for the comparison. It is a candidate only if VQC also
passes section 3.

**N.** Four new configurations: N 53 -> 57, rows in `docs/tentativas.csv` in this commit. Cost: the Composite and every
later reading now face a higher DSR bar. **On the US, N + 0:** the header of `docs/tentativas.csv` says "A mesma
configuração em outros dados ou noutro universo [...] não conta de novo", the rule of the US spec.

**Not run, and why:** Mugel's clustering (equal weights across whole clusters tends to the whole-universe portfolio, which
already lost: 11.3% vs SPY 15.5% on the US; it exists to fit hardware); long/short (the plan is unlevered and long-only);
HHL or quantum interior point on real sizes (no hardware, and Prop. 3 bounds them by the classical solve); historical-mean
Markowitz (Kelly, max Sharpe, min variance: already trials, 13-14% vs 16-17%, DECISOES 76); Mugel's dynamic tables (mu_t with
foresight would be a look-ahead backtest).

## 5. Engines and gates

**B3:** `walk_forward_dinamico`, `CFG_BT` (1,260 sessions, every 63, 10 bps per side), phases 0/21/42, cut at 16/10/2026,
frozen `docs/dados/2026-10-22/` and `2026-10-24-cvm/`, CDI `2026-10-21-macro/`. The Conservadora must reproduce 17.41/16.41/
16.90% (`check_plumbing`) and the Composite 21.4/19.9/19.1% before an arm is read.

**US:** `core.us_teste`, monthly grid, every 3 months, 36 months of training, 10 bps per side, phases 0/1/2, exports frozen
by hash after the 01/11 Tiingo batch. tau is the same (same f, c, k).

**Gate, each arm, every phase, all >= 0.95:** B3: P[> Composite], P[> Conservadora], P[> BOVA11], DSR. US: P[> Composite],
P[> Conservadora], P[> SPY], DSR. `QUBO-giro` also needs the two annealing runs to agree on >= 80% of rebalances. A pass makes
it a candidate for a forward paper log, nothing more. Nothing is tuned after a reading. The year table and the R$ columns
(BOVA11, CDI, Selic) are reported, never gates.

## 6. Solver bench (descriptive, no N)

On `QUBO-giro`'s real instances reduced to the top-n candidates with k = n // 2, n in {12, 16}, 30 dates evenly spaced in
phase 0 with one run each, and one mid-sample date with 30 seeds (the two statistical axes). Same shots (4,096) for all:

- **Exact** enumeration, the answer key; **simulated annealing**; **uniform** feasible sampling (the null);
- **QAOA-hard:** Dicke start + XY ring (Hodson's alternating operator ansatz, long-only), p = 3, existing `qaoa_subspace`;
- **QAOA-soft:** |+> start + X mixer, penalty A = max E - min E over a sample, as Hodson eq. 25, p = 3;
- **VQE:** RY + CNOT ladder, penalty as above (Mugel fig. 1), 3 layers;
- **VQE-Constrained:** the 10 lowest-energy feasible samples per date, then the best with the turnover term (Mugel's post-processing, causal: only the holdings so far).

Every bitstring is scored by one independent checker of the original problem (`|x| = k`, energy). Reported: feasibility,
approximation ratio, P[optimum], median and IQR, raw and post-processed apart, with the same repair on the uniform samples
(the only fair credit for repair). Deviation from Hodson: no shorts, so no two-qubit-per-asset encoding.

## 7. Compute

Local CPU first: `.venv-qpo` (PennyLane 0.45.1, torch 2.14.1 CPU). The bench and the VQC are small (6 and 12-16 qubits). A
synthetic timing spike, with no market data, decides. Kaggle (free, private dataset of anonymized features only) if the VQC
projects above 8 hours locally. RunPod only if Kaggle cannot, since it costs money. Heavy steps avoid the 01/11 batch's
gold steps, the paper window (08:25-08:50) and the nightly (21:30-00:30). The app does not import torch or PennyLane:
predictions are exchanged as files, as in the first QPO.

## 8. Output

`docs/qpo-hibrido-2026-11-01.txt` (IC tables, gates, year tables with CDI and Selic), derived files under `docs/dados/`, a
screen section with the same tables and each arm's latest picks, labelled information and not a recommendation, and a
DECISOES entry.
