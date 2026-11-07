# HERC-CDaR alone and on the Composite's picks, B3 — design and pre-registration

_2026-11-07. Written and committed before any number of these arms exists. User's request: has HERC (Raffinot 2018,
"The Hierarchical Equal Risk Contribution Portfolio", `Downloads/HERC.pdf`) been tested; test it alone and combined
with the Composite. Branch `feat/herc` from `feat/mix-bdr` (`b095816`). Sections 1 to 5 are fixed from the commit on._

## 1. What is already known

HERC with variance as the risk measure (Palomar 12.3.3, arm 64: Ward on the correlation distance, 12 clusters, 1/N
inside, equal risk contribution across) ran on the B3 in the book's first batch (DECISOES 234, N 68): P[> Composite]
0.0001-0.003, P[> Conservadora] 0.01-0.05, DSR 0.08-0.19; 2020 +9% vs the Composite's +30%, 2025 +25% vs +44%. On
the US, HERC 1/N, IVarP and CDaR all failed (DECISOES 221). Raffinot's paper finds the **CDaR** version best (downside
risk), which was never run on the B3.

**Prior, written now:** HERC-CDaR alone fails like the variance HERC (it reads only risk, no expected return; about 5%
it passes). HERC-CDaR-15 keeps the Composite's signal and only reweights its picks; most likely it ties or trails the
Composite (equal weights are hard to beat inside a small portfolio, attempts 47-49), about 10% it passes.

## 2. HERC-CDaR, fixed here

On the training window the engine hands the strategy (1,260 sessions of daily simple returns, ending at the decision):
correlation distance d_ij = sqrt((1 - rho_ij) / 2); **Ward** linkage; the tree is cut into **k clusters** by splitting,
from the root, the node with the greatest height until there are k (k = 12 on the whole universe, the arm 64
convention; **k = 4 on 15 names**, about sqrt(15), Raffinot's gap statistic replaced, as arm 64 did, by a fixed count).
Top-down from the root: each split gives its left child the share CDaR(right) / (CDaR(left) + CDaR(right)) of the
node's capital, where a group's CDaR is the **CDaR at 99%** (Raffinot: the 1% quantile) of its equal-weight daily
returns, uncompounded (the project's `cdar`); inside each of the k clusters, 1/N. Then the fixed caps: 20% a name,
30% a sector, each cut and its excess spread over the names with room in proportion to their weights, repeated until
both hold; then the Composite's own check (`checar_carteira`, the 40% volatility cap): a violation keeps the portfolio.

## 3. The arms

- `HERC-CDaR`: every eligible name with a sector, k = 12.
- `HERC-CDaR-15`: the Composite's 15 picks of the decision (attempt 51's score and rule, `carteira_da_nota`: 15 names,
  4 per sector), weighted by HERC-CDaR with k = 4 instead of 1/15 each.

## 4. Engine and gate

`walk_forward_dinamico`, `CFG_BT` (1,260 sessions, every 63, 10 bps per side), phases 0/21/42, cut 16/10/2026,
frozen `docs/dados/2026-10-22/`, `2026-10-24-cvm/`, `2026-10-21-macro/`; the Conservadora must reproduce
17.41/16.41/16.90% and the Composite 21.4/19.9/19.1% first. **Gate, every phase, all >= 0.95:** P[> Composite],
P[> Conservadora], P[> BOVA11], DSR. Reported, not gates: CAGR, MDD, volatility, turnover, year table, the largest
weight in the latest portfolio.

**N.** Two configurations, two rows in this branch's `docs/tentativas.csv` in this commit; the DSR uses the union of
every branch's trials, **N = 102** (100 + 2).

## 5. Output

`docs/herc-2026-11-07.txt` and a DECISOES entry. Nothing goes to the app.

## Addendum (2026-11-07, before any arm's number): skfolio's HERC replaces section 2's

The user asked to check skfolio; its `HierarchicalEqualRiskContribution` (1.12.2, the paper's algorithm) differs from
section 2, and the reference implementation is the better contract, so section 2 is replaced by it before anything
ran: Pearson distance sqrt((1 - rho)/2), Ward, the **number of clusters by the two-order difference to gap statistic**
(skfolio's `compute_optimal_n_clusters`, at most max(8, sqrt(n))) for both arms, **inverse-risk weights inside each
cluster**, a cluster's risk = the CDaR of its inverse-risk portfolio, and at each split from the root alpha = 1 -
(sum of the left clusters' risks) / (sum of both sides'); **CDaR at 95%** (skfolio's default beta), uncompounded.
Caps, check, arms, gate and N are unchanged. The code is held to skfolio's weights (before caps) by a test run in
`../.venv-skfolio`, the test-only oracle of DECISOES 168.
