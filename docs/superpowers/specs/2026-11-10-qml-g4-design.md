# G4 — Fama-MacBeth filter, PCA factor covariance and the GRS test: pre-registration (written before any G4 run)

Date: 2026-11-10. Branch: `feat/qml-quantum-portfolio` (worktree `../po-qml`). Status: pre-registered
experiment, the user's go-ahead on the recommendation of 10/11 ("sigo sua recomendação"). Nothing below
changes the frozen holdout arm (DECISOES 324), the app's comparators or the real-money book.

## What is already known (seen before this document)

DECISOES 325: the QML-Composite (Meio-Kelly) arm's +1.8 points over the Composite split into about 0.6 from
the fully invested half-Kelly acting as minimum variance and about 1.2 from a signal that 17-26% of shuffled
placebos match inside a phase. Rank IC of P(up): +0.127 over the eligible universe, +0.030 inside the
Composite's 15 names. G3 (QML score in the selection) failed in all three phases. The top-quintile condition
of item 1 below was designed knowing that; it is a filter for future signals and is disclosed as such.

## 1. Fama-MacBeth filter (a gate for future signals, applied now to the two that exist)

- Dates: the rebalance dates of phases 0/21/42 stored in the P(up) cache v3 (the first session of each
  consecutive pair; ~195 dates, 21 sessions apart). Window ends with the last session 06/11/2026.
- Names on a date: the engine's point-in-time eligible names with a full 1,260-session training window,
  a Composite score that month, a P(up) and a realized forward return.
- Per date, OLS with intercept: forward 63-session simple return on z(Composite score) and z(signal),
  each z-score taken across that date's names and clipped to +-3. A date with fewer than 10 names is skipped.
- Coefficient = time mean; standard error = Newey-West on the coefficient series with 3 lags
  (63-session labels sampled every 21 sessions overlap three times).
- Two samples: (a) all names; (b) the top quintile of the Composite score on each date (where a
  15-name book is chosen).
- **Filter rule**: a signal may go to a backtest only if t(signal | Composite) >= 2 in (a) **and** its
  coefficient in (b) is positive with t >= 1.
- Signals scored now: the QML circuit (cache v3) and the logistic baseline (`qml-logistica-v3.parquet`).
- Prior: QML passes (a) - the universe-wide IC is strong - and fails (b), consistent with G3.

## 2. PCA factor covariance (one gate arm)

- Estimator: statistical factor model on the **correlation** matrix of every eligible name of the
  1,260-session training window (not only the candidates); K = number of eigenvalues above the
  Marchenko-Pastur edge (1 + sqrt(N/T))^2; correlation = U_K L_K U_K' + Psi with Psi diagonal chosen so the
  diagonal is exactly 1; covariance = D C D with the sample standard deviations D; then the candidates'
  block, annualized x252. K is never chosen by hand and never scanned.
- Arms (phases 0/21/42; engine, dates, costs, P(up) cache v3, the Composite's 15 names as carrier,
  half-Kelly lambda = 2 under the fixed rules - all as DECISOES 318/319):
  - A1 QML-Composite (Meio-Kelly), Ledoit-Wolf - the frozen arm, reference.
  - **A2 QML-Composite (Meio-Kelly), PCA factor covariance - the gate arm.**
  - A3 No signal (p = 0.5: minimum variance of the 15), Ledoit-Wolf.
  - A4 No signal, PCA factor covariance (descriptive: does a better covariance help the minimum-variance part).
  - A5 Logistic (Meio-Kelly), Ledoit-Wolf (control, enters the GRS set).
- Gate for A2: the G2 gate, unchanged - against the Composite, in all three phases: CAGR above, Sharpe not
  below, t(alpha) >= 2 (Newey-West, daily).
- Secondary reading, reported whatever it is: alpha and t of A2 against A1 and of A4 against A3 (the
  covariance's own contribution).
- Prior: the PCA covariance and Ledoit-Wolf give similar minimum-variance books on 15 names; expected
  difference within +-0.5 point of CAGR; A2 **fails** t >= 2.

## 3. GRS test (descriptive)

- Per phase: portfolios = A1-A5 daily returns over the CDI; factor = the Composite's daily return over the
  CDI. Gibbons-Ross-Shanken F(N, T-N-K) with maximum-likelihood residual and factor covariances.
- It answers whether the five arms have **jointly** zero alpha against the Composite. It assumes iid
  normal errors, which daily returns violate (fat tails, autocorrelation): read with that caveat.
- Not a gate. A sanity check is part of the tests: with one portfolio GRS equals the squared OLS t of alpha.

## What this document forbids

- Changing the Marchenko-Pastur rule, the lags, the clipping, the samples, the filter thresholds or the arms
  after seeing numbers. Reporting one phase as a pass. Any change to the frozen holdout arm.

## Files

- `core/multivariate.py` (`pca_factor_model`, `pca_covariance`, `fama_macbeth`, `newey_west_mean_se`,
  `grs_test`), `tests/test_multivariate.py`; an optional `covariance` estimator in `core.qml.qml_strategy`
  (default None keeps the frozen arm's Ledoit-Wolf path).
- `scripts/qml_g4_evaluate.py`; `docs/qml-g4-2026-11-10-f{0,21,42}.txt`, `docs/qml-g4-fama-macbeth-2026-11-10.txt`,
  `docs/dados/qml-g4-*.csv`.
