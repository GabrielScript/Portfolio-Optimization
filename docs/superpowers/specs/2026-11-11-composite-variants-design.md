# Composite: three design choices found in the code review, tested one at a time

Date: 2026-11-11. Branch `feat/fermionic-v2`, worktree `../po-fermion`. The user: "faça a revisão criteriosa no código
da nossa estratégia principal ... precisamos extrair o máximo de retorno, CAGR, alfa, t de alfa, beta, calmar,
sortino e sharpe". **Pre-registration: committed before any number of these arms exists.**

## What the review found (attempt 51's path: `core/sinais.py`, `core/valor.py`, `core/crescimento_lucro.py`,
`core/views.py`, `core/qubo_universo.composite_score`, `core/composto.py`, `core/backtest.py`)

No look-ahead: signals use months and CVM receipts strictly before the decision month, share counts are point in
time, weights decided on the close of t-1 earn from t. One defect, not changed here because `core/valor.py` feeds
the live paper log: 229 (issuer, account, period) groups have two filings (83 with different values) and the pick
between them is arbitrary (`sort_values` is not stable in `fundamentos_no_mes`; `aggfunc="last"` follows file order
in `variacao_anual`, which matches receipt order in 58.5% of the groups). 0.11% of the rows.

Three choices were inherited from other needs and never tested on the Composite:

1. **Quarterly rebalance of a monthly signal.** The score changes every month; the portfolio is rebuilt every 63
   sessions, so two thirds of the time it holds a ranking one or two months old.
2. **Five years of history to be a candidate.** The engine drops a name without the whole 1,260-session window, a
   requirement of the covariance estimate that the Composite uses only for the rule check. The signals need 36
   months. DECISOES line 102: a median of 72% of the eligible names survive 1,260 sessions.
3. **A missing component is skipped, not neutral.** The score is the mean of the normal scores a name has. A name
   with one or two components (every financial: no Magic Formula, no Trending Value) has a wider score
   distribution than a name averaged over three, so it reaches the top 15 more often for the same information.

## Arms (one row each, N 63 -> 67)

Each changes one thing in attempt 51 and nothing else:

- **Composite-21**: `janela_teste = 21` (rebalance every 21 sessions).
- **Composite-3y**: `janela_treino = 756`, prices cut 504 sessions later so that the decision dates are attempt
  51's; candidates need three years of history.
- **Composite-neutral**: a missing component (value, growth) counts as a normal score of 0; the score is the sum of
  the three over 3.
- **Composite-all**: the three together.

Frozen exports `docs/dados/2026-10-22` and `2026-10-24-cvm`, phases 0, 21, 42, 10 bps, `walk_forward_dinamico`
with `Congelamento()`, 15 names, 4 per sector, equal weights, house rules.

## Anchor

The Composite's own strategy, run by this script, reproduces attempt 51's published curve (below 1e-9), and each
arm's curve starts on the same date.

## Gate (fixed now)

Per arm: candidate only if P[arm > Composite] >= 0.95 and DSR >= 0.95 in all three phases. Four correlated arms: a
pass is a candidate for a forward fictitious account next to the live Composite, never a change to the live rule.

## Reported

Per phase and arm: CAGR, volatility, Sharpe, Sortino (`core.metrics`), MDD, Calmar, alpha, Newey-West t and beta
against the Composite and BOVA11, P[> Composite], DSR, annual turnover, refusals, mean number of candidates, names
shared with the Composite on common decision dates, and CAGR at 30 bps approximated as CAGR - 0.002 x turnover
(marked approximate); calendar years 2010-2026.

## Expectation (registered)

Composite-21: turnover about doubles (3 -> 5 to 7 a year), CAGR +0 to +1.5 points at 10 bps, about zero at 30 bps;
P[> Composite] 0.5 to 0.9. Composite-3y: 20-35% more candidates, CAGR within +-1 point. Composite-neutral: fewer
financials, CAGR -1.5 to +0.5 points. Composite-all: near the sum. Probability that any arm passes the gate in the
three phases: about 10%.
