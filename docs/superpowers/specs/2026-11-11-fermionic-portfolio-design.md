# Fermionic portfolio: determinantal selection on the Composite score

Date: 2026-11-11. Branch `feat/fermionic-portfolio`, worktree `../po-fermion`. The user, after the solver line closed
(DECISOES 342): "Mudo a pergunta pro sinal ... crie um novo algoritmo, modelo ou fórmula para quantum portfolio para
nosso contexto"; approved the proposal below in chat. **Pre-registration: committed before any number exists.**

## Intent

Every quantum solver tested so far answered the same question (a QUBO: pairwise risk against score) and the question
itself loses to the Composite. This test changes the question, not the solver. The quantum object used is the
statistics of fermions: the probability of a configuration of free fermions is a determinant (Macchi 1975,
determinantal point processes), so two similar particles repel (Pauli exclusion). Names are the particles, the
Composite score is their quality, return correlation is their similarity. It runs exactly on a laptop: no qubits.

## The rule

At each decision (every 63 sessions), over the candidates N = every eligible name with a Composite score and a sector,
one class per issuer (`core.qpo.candidates`, no count limit: the full universe):

    S* = argmax over |S| = 15, at most 4 per sector, of   F(S) = 2 theta sum_{i in S} z_i + log det C_S

- z: the cross-sectional z-score (ddof 0) of the Composite score over N.
- C: the correlation matrix of the Ledoit-Wolf covariance of the training window over N (`estimar_momentos`,
  `shrink_mu=False`, the same estimate as attempts 51 and 56); C_S its principal submatrix.
- theta = 1, fixed here, never tuned. F is the log-probability of S under the k-DPP with kernel
  L = diag(e^{theta z}) C diag(e^{theta z}).

Weights: equal, 1/15, checked by the house rules (`checar_carteira`, a violation is `CarteiraRecusada`), exactly as
the Composite (`core.composto.estrategia_por_nota`). So against the Composite **only the selection changes**. With
no more than 15 candidates they all go in.

Solver: deterministic swap ascent. Two starts (the Composite's own picks = greedy by score under the caps; greedy by
marginal gain of F under the caps), steepest single swap (one out, one in, caps kept) until no swap raises F by more
than 1e-12; the higher F wins, ties to the first start. No seeds.

## Why this is not the QUBO again (statements checked in the tests)

1. Hadamard: log det C_S <= 0, with equality only when the picks are mutually uncorrelated.
2. Chain rule: log det C_S = sum_j log(1 - R^2_j), R^2_j the share of name j's variance explained by the picks
   before it (any order). For Gaussian returns -1/2 log det C_S is the total correlation of the set: redundancy of
   every order, where the QUBO's x' Sigma x sees pairs only. Not a quadratic form in x.
3. Limit: as theta -> infinity the rule is exactly the Composite (greedy by score maximizes a modular function on
   the caps' matroid). The Composite is the theta = infinity member of the family.
4. Trade at theta = 1: giving up 0.2 of z-score pays when it removes a correlation of about 0.57 with the set
   (log(1 - 0.57^2) = -0.4 = 2 x 0.2). That is the reason for theta = 1 as the unit: redundancy must be large to
   beat score.

Prior art: DPPs for diverse subset selection are standard in machine learning (Kulesza and Taskar 2012); their use
for portfolios is rare and, to our knowledge, not with a fundamental composite score on the B3. No literature search
was run: novelty is not claimed beyond this project.

## Data and engine

Frozen exports `docs/dados/2026-10-22` (to 16/10/2026), phases 0, 21, 42, `CFG_BT` (10 bps), `walk_forward_dinamico`
with `Congelamento()`, score = `_value_score(mensal, com_crescimento=True)`: attempt 51's set-up unchanged.

## Anchor (before reading the arm)

The same code with theta = 1e6 must reproduce attempt 51's published Composite curve
(`docs/dados/composto-curvas-2026-10-28.parquet`) in every phase: largest relative difference of the curve below
1e-9. If it fails the run stops and nothing is read.

## Gate (fixed now)

House rule since DECISOES 284: the benchmark is the Composite alone. **Candidate** only if, in all three phases,
P[Fermionic > Composite] >= 0.95 (`psr_da_diferenca`) and DSR >= 0.95 (N from `docs/tentativas.csv` plus this row).
Otherwise the line is closed. A lower drawdown without the gate is reported, not adopted.

## Reported

Per phase: CAGR, volatility, Sharpe and Sortino over the CDI, MDD, Calmar, alpha and Newey-West t against the
Composite and BOVA11, P[> Composite], DSR; calendar years 2010-2026 next to the Composite, BOVA11 and the CDI;
share of picks in common with the Composite's, mean pairwise correlation and log det of both pick sets, refusals.

## Expectation (registered)

Every diversification-aware selection lost to the Composite's plain ranking so far (attempts 56, 300, HRP arms).
Expected: 11-13 of 15 names in common with the Composite, CAGR within +-1.5 points of it, slightly lower volatility
and drawdown, P[> Composite] between 0.2 and 0.7. **Fails the gate** (probability of passing about 10%).

## Trial count

One row (theta = 1). The theta = 1e6 anchor is the Composite itself and adds no row. No second theta is run; a
different theta would be a new pre-registration and a new row.

## Non-goals

No quantum hardware or simulator, no GPU, no change to the app, the paper logs or the 19/12 buy.
