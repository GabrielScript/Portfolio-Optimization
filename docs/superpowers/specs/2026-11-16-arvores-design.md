# Tree ensembles (Random Forest, Gradient Boosting, LightGBM) on the B3 — design and pre-registration

_2026-11-16. Written and committed before any tree forecast, IC or curve exists. The user asked whether LightGBM,
Random Forest, Gradient Boosting, LSTM, DeepVAR/GPVAR or Conditional Portfolio Optimization were ever tested (none
was) and approved running the three tree models ("Pode rodar e testar os modelos LightGBM, Random Forest e Gradient
Boosting"). Branch `feat/arvores` from `feat/kan` (`2cee7bb`): it reuses the hybrid QPO's frozen
panel, purge and reading (spec `2026-11-01-qpo-hibrido-design.md`, sections 2, 3 and 5, and the KAN spec of
2026-11-07), so the trees face exactly the rows, target and gates Ridge, the MLP, the VQC, the KAN and the Composite
faced. Sections 1 to 6 are fixed from the commit on._

## 1. What is already known

Same package (`D:/b3-datalake/exports/qml/b3/pacote.npz`: 45,883 rows, 205 forecast months, 9,747-20,000 training
rows a month, target = the month's normal-scored 3-month return), monthly Spearman IC 2012-01 to 2026-07: Ridge 0.127,
Composite 0.114, KAN-1 0.111, MLP 0.109, VQC 0.104, Fourier 0.077. Every non-linear model so far lost to the linear
one; KAN-15 and VQC-15 made 12.9% and 13.2% a year against the Composite's 21.4% on this harness (the monthly refit
churns the picks).

**Prior, written now:** each tree model's IC lands within 0.10-0.13; about 10% that any of them beats Ridge with
t >= 2; no portfolio arm passes the gate (expected CAGR 12-18%, below the Composite).

## 2. Inputs and refit (same as every model of 213)

Inputs: the package's six angles (momentum 12-1, low volatility, dividend yield 12m, Magic Formula, Trending Value,
profit growth; missing = 0). Trees are invariant to monotone transforms of each input, so the angles are used as they
are. Each forecast month is fitted from scratch on that month's training rows (`pacote.treinos`, purge already built
and checked on the app's side) and predicts that month's rows. No warm start, no validation split, no early stopping,
no hyperparameter search: every setting below is fixed here.

## 3. Models, one configuration each

| Model | Library | Settings |
|---|---|---|
| `RF` | scikit-learn `RandomForestRegressor` | 300 trees, `min_samples_leaf` 100, `max_features` 2 (of 6), `max_samples` 0.5, squared error |
| `GBR` | scikit-learn `GradientBoostingRegressor` | 200 trees, learning rate 0.05, depth 3, `subsample` 0.8, `min_samples_leaf` 100, squared error |
| `LGBM` | LightGBM `LGBMRegressor` | 200 trees, learning rate 0.05, `num_leaves` 8, `min_child_samples` 100, `subsample` 0.8 every iteration, `reg_lambda` 1.0, `deterministic` true, `force_col_wise` true, L2 loss |

Leaves of at least 100 rows because the target is mostly noise (a monthly IC near 0.1 means R^2 near 1%); depth 3 / 8
leaves allow up to three-way interactions, the only thing a tree can add over Ridge besides the shape of each curve.
Seeds 0, 1, 2 for each model, predictions averaged (as the MLP and the KAN); 3 threads per model.

If LightGBM cannot be installed in `.venv-qpo`, `LGBM` is run with scikit-learn's `HistGradientBoostingRegressor`
(the same histogram gradient-boosting algorithm): 200 iterations, learning rate 0.05, `max_leaf_nodes` 8,
`min_samples_leaf` 100, `l2_regularization` 1.0, no early stopping; the output says which one ran.

## 4. Forecast test (signal)

Window 2012-01 to 2026-07. **Gate per model: IC >= 0.02 and Newey-West t (2 lags) >= 2.** Reported, not gates: paired
differences against Ridge, the MLP and the Composite, each with its t; per-seed ICs; the mean impurity importance of
each input in the last month (descriptive only). A tree edge is claimed for a model only if it passes the gate and both
its differences against Ridge and against the Composite have t >= 2.

## 5. Portfolio arms and gate

`RF-15`, `GBR-15`, `LGBM-15`: each model's forecast through `estrategia_por_nota` (15 names, 4 per sector, equal
weights), the rule of VQC-15 and KAN-15. Engine, data and checks as the KAN reading: `walk_forward_dinamico`, `CFG_BT`
(1,260 sessions, every 63, 10 bps a side), phases 0/21/42, cut 16/10/2026, frozen data of 2026-10-22; the Conservadora
must reproduce 17.41/16.41/16.90% and the Composite 21.4/19.9/19.1% (this harness predates DECISOES 349/360, so the
comparison is like for like) before any arm is read. **Gate, every phase, all >= 0.95:** P[> Composite],
P[> Conservadora], P[> BOVA11], DSR. Arms run whatever the signal test says; one is a candidate for a forward fictitious
account only if its model also passes section 4. The official book stays the pure Composite whatever happens.

**N.** Three new configurations, three rows in `docs/tentativas.csv` in this commit; the DSR uses N = 193 (190, DECISOES
372, + 3).

## 6. Compute and output

Local CPU, `.venv-qpo`, the three models in parallel (3 threads each; RAM well under 2 GB). Predictions are added to
`previsoes.npz` with the 213/KAN columns checked by hash before and after. Output: `docs/arvores-2026-11-16.txt` (IC
table, differences, importances, portfolio tables, year tables, gate) and a DECISOES entry. Nothing goes to the app.
