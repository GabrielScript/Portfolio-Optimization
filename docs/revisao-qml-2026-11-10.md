# Review of the QML code written at low effort (2026-11-10)

Scope: `core/qml.py`, `core/multivariate.py`, `scripts/qml_avaliar.py`, `qml_g2_avaliar.py`, `qml_g3_avaliar.py`,
`qml_g4_evaluate.py`, `qml_holdout.py` and their tests, read line by line against the PennyLane, PyTorch and Python
pattern guides. Every claim below was measured on the lake exports, not inferred.

## What holds

- No look-ahead in the backtest path. The cached P(up) used at each of the 66 decisions per phase is dated exactly the
  last row of the engine's training window (staleness 0 in phases 0, 21 and 42).
- The engine (`walk_forward_dinamico`) is point-in-time and keeps the 185 delisted names; the Composite reproduces.
- The 63 existing tests pass. The circuit (AngleEmbedding Y, StronglyEntanglingLayers, <Z> per wire) is a correct
  PennyLane 0.45 TorchLayer; backprop on `default.qubit` in float64 is the right differentiation choice.
- `pca_factor_model`, `newey_west_mean_se` and `grs_test` implement their formulas correctly.

## Defects (severity first)

1. **Training population is not the traded population.** `build_panel` samples every column of the price file:
   25.2% of the 49,719 samples are names not eligible in their month, plus 681 ETF rows (BOVA11, SMAL11, DIVO11,
   IVVB11). Each date's z-scores and label median are taken over that population, so the circuit learns a ranking of
   a universe the book never trades.
2. **Survivorship in the labels.** A name without a 63-session forward return (delisted inside the horizon) is
   dropped from training: 554 samples (1.1%). Small in count; it is the textbook bias and costs nothing to remove.
3. **Unpurged early stopping.** Validation is the last 15% of samples, but the 63-session labels of the training tail
   overlap it: the stopping epoch is chosen on partly seen outcomes.
4. **Order-dependent initialization.** The circuit is built before `_fit` seeds Torch, so a block's starting weights
   depend on which blocks the process trained before. The cached P(up) cannot be reproduced block by block (already
   noted and kept for the frozen holdout through `init_seed=None`).
5. **Redundant readout.** `scale * (W x + b) + bias` has two parameters with no identifiable role; weight decay also
   pulls the periodic rotation angles towards 0, which has no meaning on a circle.
6. **G2 benchmark phase.** `qml_g2_avaliar.py` computed BOVA11 on the full price frame while the arms ran on
   `iloc[fase:]`; in phases 21 and 42 the alpha against BOVA11 compared curves that started on different sessions.
   Fixed (one line).
7. **The reading's interpretation.** The headline arm (tilt inside the Composite's 15 names) got about 0.6 of its 1.8
   points from minimum variance, and placebos matched it 17-26% of the time (already recorded in DECISOES 325-326).

## Done

- `core/qnn.py` (new; `core/qml.py` untouched because the frozen holdout runs on it) fixes 1-5: eligible-only panel
  read for the month the trade earns in, holder-return labels, purged split, a generator per block, one readout with
  L2 on the readout only, any trainer through the same walk-forward (the logistic baseline included).
- 17 tests in `tests/test_qnn.py`, including an analytic check of the circuit (<Z_0> = 1, <Z_i> = cos(theta) with idle
  rotations) and a finite-difference check of the backprop gradient. Five mutations of the guards (no purge, no
  eligibility filter, order-dependent seed, one session of look-ahead, global RNG in training) each fail a test.
- Pre-registered before any v4 backtest: `docs/superpowers/specs/2026-11-10-qnn-v4-design.md` (commit e9acc35).
