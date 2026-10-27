# Quantum Portfolio Optimization (QPO) — design and pre-registration

_2026-10-27, session 52. DECISOES 143 (direction) and 144 (this design). Branch `feat/qpo`, worktree `../po-qpo`._

## 1. Intent, and the honest answer first

The user asked for a quantum portfolio strategy (PennyLane + PyTorch, NVIDIA GPU on Kaggle/RunPod) for extremely
aggressive investors that beats the Conservadora and BOVA11 in every year since 2010.

- **Feasible:** the pipeline. A QUBO asset-selection problem, solved by QAOA on a GPU statevector simulator
  (PennyLane 0.45.1 `lightning.gpu` on a Kaggle T4; CUDA-Q `nvidia` as a cross-check), built and replayed locally
  in the existing walk-forward engine.
- **Not promisable:** "beat both every year". The Conservadora itself beat BOVA11 in 12 of 17 years. The
  year-by-year table is reported; it is not a gate.
- **What quantum can and cannot add.** At n <= 28 candidates, exact enumeration of the same QUBO is cheap
  classically, and a simulated QAOA can at best reach the exact optimum. So quantum adds nothing computationally
  here by construction. Two separate questions are measured: (a) is the **QUBO model** (covariance-aware
  selection on the Conservadora signal) a better strategy, which is solver-independent; and (b) how faithful
  **QAOA** is to the exact optimum. Any return edge must come from the inputs, not from the solver. None of the 5
  papers in `Artigos_QPO` shows an out-of-sample return edge. JPMorgan's standalone QAOA failed on the two
  largest indices (QPO3). The "tunneling finds the global minimum" and "-5% in crashes" claims in
  `Quantum_portfolio_use.pdf` are unsupported.

## 2. Goals, non-goals, anti-goals

- **Goal:** one pre-registered out-of-sample test of "QUBO selection of k = 15 among the top-n Conservadora
  names, max-Sharpe weights" against the Conservadora and BOVA11, 2010-2026, 3 phases, with QAOA's fidelity
  measured on every rebalance.
- **Non-goals:** leverage (the code is unlevered: "extremely aggressive" means the growth-optimal risk aversion
  lambda = 1 in the QUBO plus max-Sharpe weights, never borrowing). US BDRs (not in a frozen folder; a second
  pre-registration). Real QPU hardware. Changing the app, `assets.py`, `core/market_data.py` or `core/paper.py`.
- **Anti-goals:** tuning anything after an out-of-sample curve has been seen. Reporting a silent CPU fallback as
  GPU. Calling a QAOA/exact difference in OOS return an "edge" (it is noise: same model, different solver).

## 3. Why the historical-mean QUBO of the 143 outline was dropped (change logged in DECISOES 144)

With mu = the Jorion-shrunk historical mean, the QUBO plus max-Sharpe weights is essentially the Kelly-with-rules
arm, whose result is already known: CAGR 13.0-14.3% against the Conservadora's 16.3-17.3% (DECISOES 76). The
version with a fighting chance uses the only signal in this project that passed its IC gate: the Conservadora
score. The test then isolates one thing: does covariance-aware **selection** beat plain rank truncation on the
same signal?

## 4. The model (fixed here)

At each decision date t (last session of the training window; month m = the month of t):

1. **Candidates `C_t`:** the Conservadora score `s_i = conservadora(retornos_mensais, proventos, m, eligible)`
   (`core/sinais.py:61`, public data before m only). Walk the score in descending order and keep names with a
   sector and one class per issuer (the higher score wins), as in `carteira_da_nota`, **without** the 4-per-sector
   cap. Stop at `n = N_MAX` names (N_MAX is fixed by the timing spike in section 7; at most 28). There is no sector
   cap in the QUBO: the weight step enforces the 30% sector cap.
2. **Signal in return units (Grinold-Kahn, alpha = IC x vol x score):** `z_i` = the cross-sectional z-score of
   `s_i` within `C_t` (ddof 0). `alpha_i = IC0 * sigma_i * z_i`, where `sigma_i = sqrt(Sigma_ii)` and
   **IC0 = 0.059**, the Conservadora's 1998-2007 IC (DECISOES 76), which predates every OOS date. The 2008-2026 IC
   (0.081) is not used: it overlaps the test.
3. **Risk:** `Sigma` = the Ledoit-Wolf covariance, annualized, from `estimar_momentos(treino[C_t])`
   (`core/estimation.py:123`), the same estimator the engine uses.
4. **QUBO:** with `x in {0,1}^n` and equal weights `w = x / k`, minimize the negative mean-variance utility
   `E(x) = -alpha'x / k + (lambda / 2) x'Sigma x / k^2` subject to `1'x = k`, with **k = 15** (the Conservadora's
   size) and **lambda = 1** (growth-optimal: the "extremely aggressive" choice). As a matrix:
   `Q = (lambda / (2 k^2)) Sigma - diag(alpha) / k` and `E(x) = x'Qx`.
   Scale check (my estimate): alpha/k is about 0.0014 per z unit and a name's marginal risk is about 0.0027, so
   both terms bite.
5. **Exact solver:** `x* = argmin_{|x| = k} E(x)` by enumerating all C(n, k) subsets. This is the ceiling QAOA can
   at best match. It is verified against brute force over all 2^n bitstrings on small n.
6. **Weights inside the chosen set:** the existing max Sharpe (`Objetivo.MAX_SHARPE`, Charnes-Cooper,
   `core/optimization.py:249`, rf = CDI of the date) through `estrategia_otimizada_nas_regras`
   (`core/backtest.py:282`) on `treino[chosen]`, with `peso_maximo = 0.20`, `n_ativos_max = None`,
   `vol_maxima = 0.40` and Jorion-shrunk mu. This is the user's fixed objective (maximum Sharpe). The fixed risk
   rules hold (sector cap 30% in the optimizer, `checar_carteira`, `CarteiraRecusada`). Max Sharpe may put zero
   weight on some of the 15 names; that is allowed.

## 5. Arms (each OOS curve is one `docs/tentativas.csv` row; N goes from 41 to 46 in the pre-registration commit)

| Arm | Selection | Weights | Role |
|---|---|---|---|
| QPO-exact | exact QUBO optimum, k = 15 of n | max Sharpe | **the strategy trial (gated)** |
| QPO-QAOA | QAOA's best sampled feasible bitstring, same QUBO | max Sharpe | solver fidelity (gated separately) |
| Conservadora-MS | Conservadora's own 15 (`carteira_da_nota`) | max Sharpe | control: separates selection from weighting |
| Topk-MS | `_resolver_com_cardinalidade` (`core/optimization.py:290`), max Sharpe on all eligible names, k = 15 | max Sharpe | control: a different strategy (continuous solve, then truncate), **not** a solver of the same QUBO |
| Random-15 | 15 uniform of `C_t`, 20 seeds, median curve | max Sharpe | luck baseline |

The existing references are not new rows: the Conservadora (15 names, equal weights, `estrategia_conservadora`)
and BOVA11 buy-and-hold.

Engine: `walk_forward_dinamico` (`core/backtest.py:213`), `CFG_BT` = 1260 / 63 / 10 bps
(`scripts/avaliar_universo_dinamico.py:74`), `FASES = (0, 21, 42)`, `Congelamento()`. Data: **all five files
from the frozen `docs/dados/2026-10-22/`**, hash-checked by `conferir_hashes` (`scripts/avaliar_robustez.py:105`).
The prices are cut at 16/10/2026 as in `scripts/rerun_conservadora.py`, so the Conservadora curve is the
published one.

## 6. Gates (fixed now; checked once)

- **Strategy gate:** QPO-exact is a candidate only if `psr_da_diferenca` P[QPO-exact > Conservadora] >= 0.95
  **and** P[QPO-exact > BOVA11] >= 0.95 in **all 3 phases** (arithmetic; the log version is reported).
- **Solver gate:** QAOA is "faithful" if, over all rebalances of the 3 phases, its pick equals the exact pick in
  >= 80% of them **and** the mean approximation ratio is >= 0.98, where
  `r = (E_worst - E_qaoa) / (E_worst - E_opt)` over the feasible set (|x| = k).
- **Reading:** with both gates passed, QPO is a candidate and the production solver is exact enumeration (it is
  instant at this n). The solver gate alone passing means "QAOA reproduces the classical answer"; it never means
  "quantum edge".
- **Reported, not gates:** the year-by-year table against both; subperiods 2010-14, 2015-19 and 2020-26; rolling
  36/60 months; `tabela_do_periodo` (Sharpe over CDI, IR, DSR with N = 46, MinTRL); P[QPO-exact >
  Conservadora-MS] (the selection effect); turnover; refusals.
- **Expectation written before running:** QAOA matches exact in most rebalances. QPO-exact differs from
  Conservadora-MS by little. Neither passes against the equal-weight Conservadora, because the max-Sharpe weights
  on historical mu cost more than the covariance-aware selection gains.

## 7. QAOA (solver; every choice below is settled leak-free, by approximation ratio only)

- **Ansatz:** QAOA with the **XY-ring mixer and a Dicke-state start** (Hadfield et al. 2019, quantum alternating
  operator ansatz). It stays inside the |x| = k subspace, so there is no penalty coefficient. Buonaiuto et al.
  (2023, QPO.pdf) show the penalty is the dominant and fragile hyperparameter. Fallback, only if the spike shows
  XY at n_max over the budget: the transverse-field mixer plus penalty `A (1'x - k)^2`, with
  `A = 2 max|Q_ij| n` (my call).
- **Training:** PennyLane 0.45.1 QNode with the Torch interface, `torch.optim.Adam` (lr 0.05), and the adjoint
  diff method on `lightning.gpu` (`lightning.qubit` locally). The diff method is checked against finite
  differences on a 6-qubit case first. p in {1, 2, 3}, 150 steps, warm start from the previous rebalance's angles
  (parameter concentration). Seeds: torch, numpy and the device seed all = 20261027 + rebalance index.
  No AMP; float64 statevector.
- **Readout:** 4096 shots; score every **feasible** sampled bitstring with the classical `E(x)` and keep the
  best. Low mean energy is not the selection criterion.
- **Choosing p and the fallback:** on the QUBOs of the first phase only, by mean approximation ratio, before
  any OOS curve exists. This uses no future return (r needs only the QUBO), so it is leak-free. Frozen after that.
- **Timing spike (before the pre-registration commit):** synthetic random QUBOs (no market data) at
  n in {16, 20, 24, 28} on Kaggle T4. `N_MAX` = the largest n whose projected total for all rebalances of the 3
  phases is <= **H = 20 GPU-hours** (Kaggle's free quota is 30 h/week).
- **GPU honesty (hard checks in the kernel log):** the device name printed by PennyLane, `nvidia-smi` before and
  after, and `torch.cuda.is_available()`. Any CPU fallback aborts the run. lightning.gpu needs compute capability
  >= 7.0: the T4 (7.5) qualifies; the P100 (6.0) is not used.
- **CUDA-Q cross-check:** on 5 QUBOs, the expectation <H_C> at the trained angles on CUDA-Q target `nvidia`
  must equal PennyLane's to 1e-6.

## 8. Compute split and reproducibility

1. Local (`.venv`): `scripts/qpo_montar.py` builds every rebalance's QUBO from the frozen data
   (`docs/qpo/qubos/fase{F}/{date}.npz`: Q, tickers, alpha, Sigma, k), the exact pick, and
   `docs/qpo/qubos/SHA256SUMS`.
2. Kaggle (Linux, T4, free): `qpo_kaggle/qpo_qaoa.py` solves each QUBO and writes one JSON per QUBO (QUBO sha256,
   bitstring, energy, r, p, angles, seed, device, versions, seconds).
3. Local: `scripts/qpo_avaliar.py` checks every JSON's QUBO hash, replays the recorded picks in
   `walk_forward_dinamico`, and writes `docs/qpo-2026-10-27.txt`. A missing pick stops the run.
4. RunPod (paid) only with the user's OK.

The QAOA code lives in its own environment (`requirements-qpo.txt`: pennylane 0.45.1, torch CPU locally). It is
never in `requirements.txt`, so the Streamlit app is untouched.

## 9. Order of work

1. QUBO, exact solver, and Ising conversion (`core/qpo.py`), test-first, checked against brute force.
2. QAOA solver (`qpo_kaggle/qpo_qaoa.py`), test-first on <= 10 qubits locally (`lightning.qubit`): it finds the exact
   optimum on small cases, stays feasible, and its gradient matches finite differences.
3. Strategies and replay (`core/qpo.py` + `scripts/qpo_montar.py`), with a test that the exact arm's picks are
   deterministic.
4. Timing spike on Kaggle, which fixes `N_MAX`. Then the **pre-registration commit**: this doc with N_MAX filled
   in, plus 5 rows in `docs/tentativas.csv`.
5. Build all QUBOs, solve them on Kaggle, choose p on phase 0, run every QUBO, replay, and report once.
