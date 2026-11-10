# Forward holdout #2 — QML-Composite (Meio-Kelly) with the PCA factor covariance: pre-registration

Date: 2026-11-10. Branch: `feat/qml-quantum-portfolio` (worktree `../po-qml`). Status: pre-registered
before any code change to the harness and before the account is opened. The user asked to open this one
shadow account and only this one ("Só abra a 1. por favor").

## What is already known (seen before this document)

G4 (DECISOES 326) ran five arms over the three phases (0/21/42) with the frozen engine:

| arm | CAGR f0/f21/f42 | alpha vs Composite | t(alpha) vs Composite |
|---|---|---|---|
| A1 QML-Composite (Meio-Kelly), Ledoit-Wolf — the frozen holdout | 22.2 / 19.4 / 19.8 % | 3.4 / 2.1 / 2.9 pt | 1.81 / 1.06 / 1.50 |
| **A2 QML-Composite (Meio-Kelly), PCA factor covariance** | **22.4 / 19.7 / 20.4 %** | **3.5 / 2.3 / 3.3 pt** | **1.89 / 1.20 / 1.74** |
| A3 No signal, Ledoit-Wolf | 20.5 / 18.1 / 19.2 % | 2.2 / 1.2 / 2.6 pt | 1.24 / 0.68 / 1.45 |
| A4 No signal, PCA factor covariance | 21.1 / 18.2 / 19.5 % | 2.8 / 1.3 / 2.9 pt | 1.53 / 0.73 / 1.59 |

A2 beat A1 in all three phases, by 0.2 to 0.8 point of CAGR, with the same number of names and essentially
the same volatility and drawdown; and part of that gain survives without a signal (A4 vs A3), i.e. it comes
from the minimum-variance part, not from better P(up). The G2 gate (t(alpha) >= 2 in all three phases) still
**fails** — A2's best t is 1.89. So A2 is not adopted: it is a candidate worth watching forward, which is
exactly what this account is for.

## The arm

- Fictitious R$ 10,000 account, no leverage, same rules as the frozen holdout: rebalance every 63 sessions,
  10 bps per side charged on the turnover only, shares held fixed between rebalances, marks from the lake's
  closes (forward-filled; a held name with no price is an error, never a zero).
- Carrier and signal: unchanged from the frozen account — the QML circuit's P(up) over 63 sessions, the
  Composite's 15 names as the point-in-time universe, tilt 1.0, `CFG_KELLY` (half-Kelly), the fixed caps in
  `LIMITES` (20% per name, 30% per sector, 40% vol).
- **The only difference from the frozen account is the covariance estimator**: `pca_covariance` from
  `core.multivariate`, fitted through `qml_strategy(..., covariance=...)` on **every eligible name** of the
  1,260-session training window (not only the 15 candidates), then cut to the candidates and annualized
  x252. K is the number of eigenvalues above the Marchenko-Pastur edge and is never chosen by hand. The
  frozen account keeps its Ledoit-Wolf path (its `covariance` stays `None`).
- **Model seed: pinned.** The new account trains the circuit with `QMLConfig(init_seed=20261110)`; the
  frozen account keeps `init_seed=None` (legacy, training-order dependent). Rationale: a forward record is
  only auditable if anyone re-running it on the same lake gets the same book, and DECISOES 325 already said
  new arms should pin the seed. The cost is that the two accounts differ in seed as well as covariance; that
  is disclosed, and the seed was drawn before any weight of this account existed, so it is not a tuned
  parameter.
- Ledger: `simulador/qml_holdout_pca/` — `estado.json` (strategy name "QML-Composite (Meio-Kelly), PCA
  covariance"), `ordens.csv`, `marcas.csv`. The frozen ledger `simulador/qml_holdout/` is not touched.

## Start and marks

- Start = the **first session on or after 2026-11-10** present in the frozen lake at the opening run. If the
  lake's last session is earlier, the account stays unopened and is opened by `iniciar` on the first run
  after the lake updates; no backfilled marks, no retroactive book.
- Opening = one buy at the target weights net of 10 bps; every session after that gets one mark, whether or
  not anyone looks at it.
- Whoever runs the daily `rodar` for the frozen holdout runs it for this account too.

## How it is judged

- Read at 12 months of marks, against two benchmarks on the same dates: the **Composite** (the app's
  comparator) and the **frozen Ledoit-Wolf holdout** (same signal, same carrier, different covariance).
- Metrics: CAGR, volatility, MDD, and alpha/t of this account against each benchmark (Newey-West, daily).
- **A year has little power.** With this arm's ~18% volatility and a ~2% tracking error against the frozen
  account, one year of daily data gives a t of roughly 0.4 for a 1-point annual gap; the 12-month reading can
  confirm a large gap or a blow-up, not decide a 0.5-point one. The account is a sanity check of the
  backtest, not a verdict, and its first real decision point is the 12-month reading plus the frozen
  account's, taken together.
- Stopping: no early stop on performance; the account is closed only if the arm is rejected by a later
  pre-registered test or by the user.

## What this document forbids

- Running `iniciar` before this spec is committed. Changing the estimator, the seed, the carrier, the caps or
  the step after seeing marks. Restating this account's numbers as evidence the gate passed — the gate of
  DECISOES 326 stays failed until a pre-registered test says otherwise.
- Any edit to the frozen holdout's ledger, its rules or its `init_seed`.

## Files

- `scripts/qml_holdout.py` (account descriptor: ledger folder, strategy name, covariance estimator, seed,
  start date; the default account reproduces the frozen behaviour byte for byte), `tests/test_qml_holdout.py`.
- Ledger `simulador/qml_holdout_pca/`. DECISOES 327.
