# G3 — the QML signal in the selection, not in the weights: pre-registration (written before any G3 run)

Date: 2026-11-10. Branch: `feat/qml-quantum-portfolio` (worktree `../po-qml`). Status: pre-registered
experiment. The configuration below cannot move after the numbers are seen. The frozen holdout arm of
G2 Stage 2 (QML-Composite, Meio-Kelly, opened 09/11/2026) is **not** changed by anything here.

## What was seen before this document (and therefore cannot be a holdout)

A code review of the arm on 10/11 ran attribution controls on the stored P(up) cache (v3, no retraining),
same engine, dates, costs and Composite carrier as DECISOES 318/319. CAGR in phases 0 / 21 / 42:

| Arm (Meio-Kelly inside the Composite's 15 names) | 0 | 21 | 42 |
|---|---|---|---|
| Composite (benchmark, equal weights) | 19.91% | 18.51% | 17.76% |
| QML P(up) (the frozen arm) | 22.24% | 19.45% | 19.84% |
| No signal (p = 0.5: minimum variance of the 15 names) | 20.53% | 18.07% | 19.19% |
| Anti-signal (1 - p) | 18.03% | 15.87% | 17.43% |
| Classical logistic regression, same features and blocks | 20.74% | 19.61% | 18.42% |
| Shuffled P(up), seeds 0 / 1 / 2 | 21.94 / 19.97 / 24.51% | 19.37 / 17.75 / 19.12% | 17.20 / 15.78 / 15.96% |

Rank IC of P(up) against the forward 63-session return, 195 decision dates: **+0.127** in the engine's
point-in-time universe (about 116 names, hit rate 0.74, overlap-adjusted t 4.9), **+0.030** inside the
Composite's 15 names (hit rate 0.51, t 0.9). The logistic baseline: +0.102 and +0.010; pooled correlation
between the two probabilities 0.82.

Reading: the signal carries direction across the tradable universe and almost none inside the 15 names
where the arm spends it. There, about half of the arm's gain over the Composite is the fully invested
half-Kelly objective acting as minimum variance, and a shuffled signal lands on both sides of the real
one. A wider placebo set (20 more seeds) was still running when this document was committed.

## Hypothesis

Grinold: IR ~ IC x sqrt(breadth). The signal should be spent where its IC is 0.13 over ~116 names: in
the **selection** of the book. The optimizer is left out on purpose - every optimized-weight variant so
far concentrated into 8-10 names and lost to equal weights on the same names (DECISOES 318).

## Arms (the only ones this document allows)

All arms: the engine and data of DECISOES 318/319 (`walk_forward_dinamico`, 1,260-session window,
63-session rebalance, 10 bps per side, lake exports as frozen by the script, last session 06/11/2026),
P(up) from the stored v3 cache with the existing as-of lookup, the Composite's book rule
(`carteira_da_nota`: 15 names, 4 per sector, one class per issuer, equal weights) and the fixed-rule
check. Phases 0, 21 and 42.

- **A. Composite + QML score (the gate arm).** Score = z(Composite score) + 0.5 x z(2p - 1), both
  z-scores taken across the names that have a Composite score on the decision month; a name without
  P(up) gets p = 0.5. The weight 0.5 is the value already in `scripts/qml_avaliar.py`
  (`seletor_combinado`, the `--rapido` default); it is not scanned.
- **B. QML top 15 (descriptive).** The same book rule ordered by P(up) alone over the engine's eligible
  names. Shows the signal with full breadth and no Composite.
- **C. Composite + logistic score (control).** Arm A with the classical logistic probabilities instead
  of the circuit's. Answers whether the quantum circuit matters; it cannot pass the gate for the QML line.
- **D. Composite + shuffled score, seeds 0-4 (placebo).** Arm A with each date's P(up) dealt to the names
  at random. The null distribution of "adding a 0.5-weight noise score to the Composite".

## Gate (decided before the run; the G2 gate, unchanged)

1. Benchmark: the Composite. Annual alpha with Newey-West errors on daily returns.
2. Arm A passes Stage 1 only if, in **all three phases**: CAGR above the Composite, Sharpe not below it,
   and t(alpha vs Composite) >= 2.
3. Additional reading, reported whatever it is: arm A's CAGR against the five placebos of the same phase
   (A must beat at least four of five in every phase to be called distinguishable from noise) and
   against arm C.
4. A Stage-1 pass makes a candidate, not a strategy: the historical window has been seen many times
   (N in `docs/tentativas.csv` plus this branch's arms). Stage 2 would be a forward paper account under a
   new document. A fail is a recorded result, not a reason to retune the weight, the horizon or the book.

## Prior, written here

Not optimistic. The Composite already contains low volatility and momentum through the Conservadora, so
the price-feature signal is probably correlated with it and the incremental IC is smaller than 0.127.
The optimized-weight cousin of arm A (`QML+Composite (lambda=0.5, Max. Sharpe)`) gave 19.92 / 16.88 /
17.20% against the Composite's 19.91 / 18.51 / 17.76%. Expected verdict: A beats the Composite on CAGR in
two or three phases by 0.5-1.5 points and **fails** t >= 2.

## What this document forbids

- Changing the 0.5 weight, the 15-name book, the horizon, the features or the cache after seeing numbers.
- Reporting one phase as a pass, or the IC as if it were a return.
- Any change to the frozen holdout arm, to the app's comparators or to the real-money book.

## Files

- `scripts/qml_g3_avaliar.py` (arms A-D; `--atribuicao` reruns the controls of the table above),
  `tests/test_qml_g3.py`.
- `docs/qml-g3-2026-11-10-f{0,21,42}.txt`, `docs/dados/qml-g3-f{0,21,42}.csv`,
  `docs/dados/qml-logistica-v3.parquet` (the logistic baseline's probabilities).
