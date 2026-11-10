# DRL portfolio replication on the B3 lake (2010-2026, plus 2005-2009 as a thin-universe annex)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replicate the three supplied DRL papers on our own B3 data (stocks, units, ETFs, BDRs, delisted included) and decide, against the Composite, whether any of them is a portfolio strategy worth keeping.

**Architecture:** A new `drl/` package beside `core/`, in the same repository, reading `lago/precos_diarios.parquet` direct (5,415 sessions x 1,545 names, 2005-01-03 to 2026-11-09). One Gymnasium environment with three reward variants matching the papers, one point-in-time universe rule, and Stable-Baselines3 agents. Nothing in `core/` is modified; the existing simulator, paper and Composite are read-only imports.

**Tech Stack:** Python 3.12, `.venv-drl` (torch CPU, gymnasium, stable-baselines3, numpy, pandas, pytest), ruff line length 110.

**Spec:** The three papers in `drl_po/` (Sood et al. 2023 = P1, Rezaei & Nezamabadi-Pour 2025 = P2, RMS-driven DRL = P3), extracted in the transcript and summarised below.

## The three papers in one paragraph each

- **P1 (Sood, Papasotiriou, Vaicius, Balch — J.P. Morgan, FinPlan'23 / arXiv 2602.17098):** PPO from SB3 with **Differential Sharpe Ratio** (Moody-Saffell) as the per-step reward; observation = (N+1) x 60 matrix, one row per asset of 60 lagged daily log returns plus its current weight, the last row is cash carrying vol20, vol60/vol20 and VIX; weights via softmax; net [64,64] tanh; expanding-window standardisation; 5 seeded agents per round with the best-on-validation carried forward; 5y train + 1y burn-in + 1y test, 10 sliding windows. Benchmark: 60-day sample means + Ledoit-Wolf MVO. Reported Sharpe 1.17 vs 0.68.
- **P2 (Rezaei & Nezamabadi-Pour, Artificial Intelligence Review 2025):** A2C, DDPG, PPO, SAC, TD3 as in the FinRL library, no modification, reward = change in portfolio value, zero costs, DJIA-30, 2008-2023, one 75/25 split. Its own result is that all five agents reproduce the **equal-weight** portfolio almost exactly; the honest reading is that it is a baseline-confirmation study, not an edge.
- **P3 (RMS-driven DRL):** A2C/PPO/DDPG with **rolling-mean-standardisation** of the features (the "RMS") plus a drawdown penalty in the reward; DJIA-30, train 2010-10/2021, test 10/2021-04/2024; Sharpe 0.66 vs 0.42 for the index. No transaction costs.

## What we already know from our own history

- The standing benchmark in this project is the **Composite**, not an index. Every result is measured against it, and every new attempt increments N in `DECISOES.md` and must clear the same probability gate as the other strategies.
- The lake's cross-section is thin before 2020 (150 names with 200+ closes in 2010, 305 in 2020, 760 in 2026). Anything trained on 2010-2015 sees a different market than the one it is tested on. Results are therefore reported per era, and the era is fixed before any test number is looked at.
- Costs: B3 brokerage + emolumentos + spread/slippage. All three papers assume **zero costs**, so every run is done twice (zero-cost = fidelity, with-cost = decision) and only the costed number may be compared with the Composite.

## Global Constraints

- `core/market_data.py`, `core/paper.py`, `core/bcb.py` and `assets.py` are never edited.
- The Composite, the paper accounts and the options ledgers are never written by anything in `drl/`.
- No look-ahead: every feature at session t uses data up to and including t's close; the trade happens at t+1's open-to-close or t+1's close (fixed once, below).
- The universe at each rebalance is point-in-time: a name must have 252 sessions of history and a minimum traded value, both measured on data strictly before the rebalance date.
- Delisted names stay in the panel (survivorship). A name that stops trading is force-liquidated at its last close, at zero cost, and the row is flagged.
- Attempts are pre-registered: the grid and the splits are written to `DECISOES.md` **before** the test window is evaluated.

## Review Focus

- **Zero-cost results must never reach the decision table.** The most likely failure of this work is quoting a cost-free Sharpe as if it were achievable.
- **A DRL agent that lands on equal-weight** (the P2 outcome) must be reported as such, not as an edge.
- **The train/test distribution shift** across the 2019→2020 universe jump must be visible in the report, per era.
- **A name that stops trading mid-episode** must not produce NaN returns or a runaway position.
- **The best-seeded agent of a walk-forward round** must be chosen on validation, never on the test year.

---

### Task 1: The DRL environment and the data panel

**Files:**
- Create: `drl/__init__.py`, `drl/dados.py`, `drl/ambiente.py`
- Test: `tests/test_drl_dados.py`, `tests/test_drl_ambiente.py`

**Interfaces:**
- `drl.dados.painel(caminho: Path, inicio: str, fim: str) -> pd.DataFrame` — wide closes, sessions as index, names as columns, delisted included.
- `drl.dados.elegiveis(fechamentos, sessoes_hist=252, valor_minimo=...) -> pd.DataFrame` — boolean DataFrame (session x name), True where the name is tradeable at that session, computed with data strictly before it.
- `drl.ambiente.AmbienteAcoes(gymnasium.Env)` with `recompensa: Literal["valor", "dsr", "dsr_drawdown"]`, `custo_bps: float`, `N` names, action = simplex weights (softmax applied by the policy), observation build mode `"matriz"` (P1 faithful, N <= 40) or `"compacto"`.

- [ ] **Step 1: Write the failing tests** — panel loads and is sorted; a name with a gap is eligible only after 252 valid closes; a delisted name is liquidated at its last close and the episode continues with cash; the reward of a flat portfolio is 0 for all three variants; the DSR recursion equals the paper's formula on a hand-computed 3-step example.
- [ ] **Step 2: Run them, see them fail.**
- [ ] **Step 3: Implement** `drl/dados.py` and `drl/ambiente.py`.
- [ ] **Step 4: Run the tests and the file's suite, all pass.**
- [ ] **Step 5: Commit.**

### Task 2: Baselines on the same tape

**Files:** Create `drl/baselines.py`; Test `tests/test_drl_baselines.py`.

Equal-weight, min-variance and Ledoit-Wolf MVO over the same 60-session lookback and the same eligible set, plus Ibovespa from `index_daily.parquet` and the Composite's recorded curve. Each returns the same metric frame as the agents, so one table compares everything.

### Task 3: Agents and the pre-registered walk-forward

**Files:** Create `drl/agentes.py`, `drl/protocolo.py`, `drl/relatorio.py`; Test `tests/test_drl_protocolo.py`.

- `drl.agentes.treinar(algoritmo, ambiente, total_timesteps, seed) -> modelo` for `"ppo" | "a2c" | "ddpg" | "td3" | "sac"` through SB3.
- `drl.protocolo.janelas(...)` — the fixed split list, written to DECISOES before running.
- `drl.relatorio.metricas(curva)` — return, vol, Sharpe, MDD, Calmar, turnover, cost drag; comparable with the Composite's own frame.

### Task 4: The runs

Fidelity grid (N = 30 most liquid, PIT): PPO+DSR, A2C/PPO/DDPG+return, A2C/PPO/DDPG+RMS — zero-cost and costed. Scale grid (N = every eligible name, capped at 200 per window): PPO+DSR and PPO+return only. Each run writes a CSV under `drl/saidas/` and a line in the log.

### Task 5: The reading

One table per paper, per era, zero-cost vs costed, against equal-weight, MVO, Ibovespa and the Composite. `DECISOES.md` entry with N incremented, the gate applied, and an explicit statement of what would have to be true for any agent to be kept.

## Self-review

- No task touches `core/`. Task 1 owns every look-ahead guard; Task 3 owns the selection discipline.
- The GPU question is settled: CPU only, `SubprocVecEnv` with 4 workers.
- Ray RLlib and FinRL-Meta are deliberately not dependencies; see the environment note in Task 3's interface block.
