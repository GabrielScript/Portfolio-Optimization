# QNN v4: corrected quantum signal, screened into every classical method (pre-registered 2026-11-10)

Written and committed before any v4 backtest ran. Test branch only; nothing here reaches the app, the comparators,
the forward holdouts or the real-money book.

## Why

The 2026-11-10 review of `core/qml.py` (written by a weaker model at low effort) found no look-ahead in the test
path, but found pipeline defects that change what the circuit learns:

1. 25.2% of the training samples were names not eligible in their month, plus 681 ETF rows; each date's z-scores and
   label median were taken over that wider population, not over the names the book can trade.
2. Names without a 63-session forward return (delisted inside the horizon) were dropped from training (1.1%).
3. Early stopping used a validation tail with no purge: 63-session labels of the training tail overlap it.
4. Initial weights depended on which blocks the process had trained before (cache not reproducible).
5. Redundant readout parameters; L2 penalty on periodic rotation angles; no analytic check of the circuit.

`core/qnn.py` fixes these (see its docstring). `core/qml.py` is untouched because the frozen holdout runs on it.

## Signal

- Circuit unchanged: 6 qubits, AngleEmbedding (Y), 2 StronglyEntanglingLayers, <Z> per wire, linear readout.
- Panel: one cross-section every 21 sessions; names eligible in the trading month with a full 252-session feature
  window; label = holder's 63-session return above the cross-section median.
- Walk-forward: a new model every calendar year, trained on labels realized before 1 January; purged 15% validation.
- P(up) = mean over seeds 0, 1, 2. Fixed before the run; no other seed set is tried.
- Classical baseline through the same panel and blocks: logistic regression (C = 1).

## Backtest

- Engine `walk_forward_dinamico`, attempt 51's `CFG_BT` (1,260-session training window, rebalance every 63
  sessions, 10 bps a side), the lake's point-in-time universe with the 185 delisted names, `Congelamento()`.
- Window: lake exports cut at 2026-11-06 (the window of DECISOES 318), phases 0, 21 and 42.
- Sanity check: the Composite must reproduce 19.91 / 18.51 / 17.76% CAGR.

## Arms

Twelve base methods X, built by the same factories as the Aleph Quant comparison screen: Composite, Conservadora,
Markowitz, Bayes-Stein (Jorion), minimum variance, Kelly (lambda = 2), Michaud resampled, risk parity (ERC), HRP,
tree HRP, HRP with the paper's distance, 1/N.

For each X: `X` alone and `X + QNN` = X run on the training window cut to the top half of its names by P(up)
(`core.qnn.screened`, keep_share 0.5, at least 20 names). The screen is the only link, so it is the same for every
method and the allocator is never changed. Controls, same screen: `X + logistic`, `X + shuffled` (each date's P(up)
dealt at random, seed 0), `X + anti` (1 - p). Michaud runs only `X` and `X + QNN` (30 minutes a curve).

Also, as a re-test of the old headline: the tilt arm inside the Composite's 15 names (`qml_strategy`, Meio-Kelly,
tilt 1) with v4 P(up), with the stored v3 P(up), with no signal (p = 0.5), with the logistic and with a shuffle.

## Gates (fixed here)

- Signal gate, per X: `X + QNN` passes when, in all three phases, its CAGR and Sharpe are above X's, the alpha of
  `X + QNN` regressed on `X` (Newey-West) has t >= 2.87 (5% two-sided, Bonferroni over 12 methods), and its CAGR is
  above `X + shuffled` and `X + anti`.
- Quantum gate, per X: the signal gate passes and `X + QNN` has CAGR above `X + logistic` in all three phases.
  Without it the result is credited to the features, not to the circuit.
- House gate: any arm proposed for further work must also beat the Composite in all three phases with t >= 2.
- Tilt re-test: passes when v4 beats the Composite with t >= 2 in all three phases and beats its no-signal control.

Descriptive, no gate: rank IC of P(up) against the 63-session holder return in the eligible universe (Newey-West,
3 lags), per seed, for the mean and for the logistic.

Trials: 12 screened arms + 1 tilt arm = 13 new trials for the DSR count.
