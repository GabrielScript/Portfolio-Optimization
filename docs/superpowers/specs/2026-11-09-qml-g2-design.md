# G2 — Composite Quantum: pre-registration (written before any run)

Date: 2026-11-09. Branch: `feat/qml-quantum-portfolio` (worktree `../po-qml`). Status: pre-registered
experiment. The N in `docs/tentativas.csv` increases **only** if a global gate is executed and reported;
this document exists so that the configuration cannot move after the numbers are seen.

## Names

- The arm is **Composite Quantum** (the user's name, 2026-11-09). "Composite" is the original 15-name
  equal-weight book and is the **benchmark**; no separate benchmark name is needed.
- The arms below are the only ones this pre-registration allows. Any other arm is a new document.

## Why G2 exists

DECISOES 319/321 left the QML arm as a *candidate*: it beats the Composite on CAGR and Sharpe in all three
phases, but t(alpha vs the Composite) is 1.81 / 1.06 / 1.50, with 95% intervals that include zero
(MinTRL 71-680 years). The same t is the t of the mean daily return difference, so `t = IR * sqrt(years)`:
at 16.7 years, t = 2 needs IR >= 0.49, against 0.44 / 0.26 / 0.37 today. Decision frequency cannot buy
that (the alpha regression already uses daily returns); only a better information ratio or more time can.

## Arms

### A. Composite Quantum — daily prediction (the gate arm)

- Same engine, same 15-name carrier, same fixed rules as DECISOES 318/319 (Kelly lambda = 2, 20% per name,
  30% per sector, 40% ex-ante volatility, no leverage, 10 bps per side), **same 63-session rebalance**.
- The only change: the signal lookup for a decision on session d uses the model's P(up) **for d itself**,
  not for the last cached date. Today a decision can act on a prediction up to 62 sessions old. Every
  session from the first decision on is predicted, causally (the cache stays date-keyed).
- Cost: ~4,200 prediction dates instead of 396. Estimated 60-90 min on the CPU (4 threads); the 2-hour
  user budget covers it. No GPU: the circuit is 6 qubits and 45 parameters - a GPU is slower here.

### B. Composite Quantum + BDR (descriptive, not gated)

- Candidate set per rebalance: the Composite's 15 B3 names **union** the 32 names of `ativos_eua.ATIVOS_EUA`
  that are eligible that month (close in the app's panel, >= 252 sessions of history, >= 95% coverage).
- The optimizer allocates over the union under the same fixed rules, with the BDRs as one extra sector
  ("EUA") capped at 30%. IVVB11 and NASD11 are ETFs, not BDRs: they stay in the BDR list only if the user
  says so; the default of this document is to exclude them from the sector and treat them as ordinary ETFs
  of the panel (i.e. out of the BDR cap).
- **Data check (2026-11-09, app panel `lago/precos_diarios.parquet`)**: 16 of the 32 start on 03/09/2013,
  6 in 2019-2020, 6 in 2021-2022; all 32 exist together only from 21/06/2022. The user remembered 2021;
  the panel says 2013 for half the list. With dynamic eligibility the arm starts around 2014-09 with 16
  names and grows to 32. This is reported, not chosen after seeing returns.
- **Prior written here**: T ~ 12 years gives sqrt(T) ~ 3.46, so t = 2 needs IR >= 0.58 - above every IR
  the arm has shown (0.26-0.44). Expected verdict: **fails the gate**. It is run because it is cheap and
  because the user asked for it; a failure is a recorded result, not a reason to retune.

## Gate (decided before the run)

1. **Benchmark**: the Composite (original, 15 names, equal weight), annual alpha with Newey-West errors.
2. **Metric**: t(alpha vs Composite) >= 2 **and** the 95% interval of the alpha strictly above zero, in
   **all three phases** (0, 21, 42).
3. **Holdout**: the historical window has been seen (DECISOES 318-321), so it cannot be a holdout. The
   only clean holdout is forward: from the first session after this document, **12 months** of paper with
   the same rules, same metric. Stage 1 (historical) is necessary but not sufficient; a Stage-1 pass
   without the Stage-2 forward period is a candidate, not a passed strategy.
4. **Deflation**: the result is reported with the DSR at the union N of `docs/tentativas.csv`, and N
   increases by the arms actually run here. No configuration is scanned to find a pass.

## What this document forbids

- Changing the per-name cap, the carrier, the horizon, the tilt or the feature set after seeing numbers.
- Reporting a phase-by-phase winner as a pass (319's protocol error).
- Calling the BDR arm a pass on CAGR alone: with T ~ 12 years only the t/CI rule counts.
- Any run on the real-money book or on the app's comparators. Both stay on the Composite.

## Files this run will write

- `core/qml.py` changes: optional daily prediction path (causal, cache-keyed), optional BDR merge.
- `scripts/qml_g2_avaliar.py` (new).
- `docs/qml-g2-2026-11-09.txt`, `docs/dados/qml-g2-comparacao.csv`.
