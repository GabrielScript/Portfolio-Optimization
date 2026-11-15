# The Composite's inputs through an infinitely wide random-features net (pre-registered 2026-11-15)

Written and committed before any number of it is read. Branch `feat/composto-rf`, from `feat/composto-faixa`
(`cde5e5b`, which holds the same-run Composite replay of the band reading and the growth age rule of DECISOES 360),
worktree `../po-rf`.

## Why

The user, 15/11, asked whether the mean-field limit of wide neural nets (SGD on a two-layer net as N -> infinity is a
Wasserstein gradient flow, a potential mean-field game) could improve the Aleph Quant's returns. The answer: the
theory is about how a wide net trains, not a source of signal; the one testable bridge is random features ("Virtue of
Complexity", Kelly, Malamud and Zhou 2024; contested by Nagel 2025). The user approved one pre-registered reading.

The question it answers: does a nonlinear map of the Composite's own inputs order next month's returns beyond what
the Composite's linear mean of normal scores already says? The inputs are the same, so no new information enters;
the only possible gain is interactions among the legs that the Composite's equal-weight sum cannot express.

## The model (one configuration, no search)

A two-layer net with random first-layer weights w ~ N(0, 2 gamma I) and cos/sin features, trained by ridge on the
second layer, converges as its width P -> infinity to kernel ridge regression with the Gaussian kernel
k(x, x') = exp(-gamma |x - x'|^2). The registered model is that limit: no width, no seed, no draw to choose.

- **Inputs** (decision month M, every input dated strictly before M, as in the Composite): the six legs of the
  Composite's three components, each the month's normal score (`core.views.escore_normal`) among the names that
  have it, a missing leg set to 0 (the mean):
  `vol` (`core.sinais.volatilidade_36m`), `momentum` (`momentum_12_1`), `dy` (`dividend_yield_12m`), `formula`
  (the Magic Formula score of `scripts.avaliar_valor.notas`), `trending` (its Trending Value score), `crescimento`
  (`core.crescimento_lucro.nota_de_crescimento`, with the age rule). A name enters month M only with `vol` and
  `momentum` (the Conservadora's own requirement), among the month's point-in-time universe.
- **Target** of a training row (name i, month t): the normal score of i's total return of month t among the month's
  rows.
- **Training window** for M: the rows of the 60 months M-60 .. M-1 (the return of M-1 is known on M's first day).
  Fewer than 36 months with rows: no score at M.
- **Kernel width:** gamma = 1/6 (one over the number of inputs; the inputs have unit variance, so this is
  scikit-learn's `gamma="scale"`). Fixed.
- **Ridge alpha:** from the grid (0.1, 1, 10, 100, 1000), chosen in January of each year (and at the first scored
  month) on the months before it only: fit on M-60 .. M-13, score M-12 .. M-1, take the alpha of largest mean monthly
  Spearman IC (a tie goes to the larger alpha); the alpha is kept until the next January. Then the fit on the whole
  window M-60 .. M-1.
- **Score at M:** the fitted function on the month's inputs.

## Data (frozen, no network)

`docs/dados/2026-10-22/` and `docs/dados/2026-10-24-cvm/`: the files the Composite and the band were read on. The
scores of every month 2010-01 .. 2026-10 are computed once by `scripts/composto_rf_painel.py` and saved with their
SHA-256 under `docs/dados/composto-rf/`; both stages read only that file.

## Stage 1: one IC reading

The house harness (`core.poder.ic_por_mes`), 2012-01 to 2026-09, as in DECISOES 357.

- Anchors first: the Conservadora, value and profit growth reproduce their published ICs (0.085, 0.070, 0.033,
  tolerance 0.003), or nothing is read.
- **Gate:** mean IC >= 0.02 and t >= 2.
- **New information:** each month the score's normal score is regressed across names on an intercept and the normal
  scores of the three components (a missing component counts as 0); the residual's IC is read the same way. The
  score is a **stage 2 candidate** only if it passes the gate and its residual IC has mean > 0 and t >= 2.
- Description, not gate: the same reading for a linear ridge on the six inputs (same window, same alpha rule), which
  says whether a gain comes from nonlinearity or from reweighting the legs; the three subperiods; the mean rank
  correlation with the Composite's score; the alphas chosen.
- An IC reading is not a strategy configuration: N does not move (189).

## Stage 2 (only for a candidate; not built otherwise)

One arm, `RF-kernel-15`: the 15 names of highest score by `carteira_da_nota`'s rules (at most 4 per sector, one class
per issuer), equal weights, `checar_carteira` as the Composite's (a refusal keeps the book), the Composite's dates,
engine, costs (10 bps) and limits. A decision month without a score uses the Composite's score that month (logged).
`scripts/composto_variantes_avaliar.py rf`: phases 0 / 21 / 42, compared with the Composite replayed in the same run
(the band's addendum: the saved curve predates DECISOES 349/360). DSR at N = 190. Gate: P[> Composite] >= 0.95 and
DSR >= 0.95 in every phase. If it passes: a forward ghost account only; the official book (app, MT5 demo, the 19/12
buy) stays the pure Composite.

## Prior, stated before the data

- Raw IC passes the gate, near the Composite's own (the inputs are its legs).
- Residual IC: mean within 0.01 of zero, t below 2: no candidate. One chance in four, at most, of a candidate.
- The linear ridge: residual IC near zero as well (reweighting the legs was tried as Composite variants and EIC
  weights, DECISOES 342-350).
- If a candidate exists, stage 2 ties: P[> Composite] between 0.2 and 0.9 in the phases, no pass.

## Code

`core/random_features.py` (the kernel, the fit, the alpha choice: numpy and scipy only), `core/composto_rf.py` (the
inputs of a month, the rolling scores, the residual, the score as a `NotaDoMes`), `scripts/composto_rf_painel.py`,
`scripts/composto_rf_ic.py`, the `rf` family in `scripts/composto_variantes_avaliar.py` (stage 2 only),
`tests/test_random_features.py`, `tests/test_composto_rf.py`. Nothing the app, the paper logs or the accounts import
changes.
