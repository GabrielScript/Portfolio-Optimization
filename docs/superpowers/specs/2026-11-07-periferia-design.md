# Periphery tilt of the Composite, B3 — design and pre-registration

_2026-11-07. Written and committed before any number of this arm exists. User's request ("periferia"): test idea 2
of `docs/novas-arquiteturas-2026-11-07.md` (main `2bf673f`): tilt the Composite's picks toward the periphery of the
market's correlation network. Branch `feat/periferia` from `feat/kan` (`2cee7bb`): same engine, frozen data and gate
as the KAN-15 reading. Sections 1 to 5 are fixed from the commit on._

## 1. Source and prior

Pozzi, Di Matteo and Aste (2013, Sci. Rep. 3:1665): stocks on the periphery of correlation-filtered networks (MST,
PMFG) gave portfolios with better risk and return out of sample than central ones, on US stocks. Central names move
with the market; peripheral ones diversify. Not tested on the B3 here before; the graph/tree HRP we ran set weights,
this sets the ranking.

**Prior, written now:** about 20% that Periferia-15 passes the gate (the note's figure); the likelier outcome is a
lower drawdown with a CAGR at or below the Composite's, which fails P[> Composite].

## 2. The network, at each decision

The engine hands the strategy the training window of daily simple returns ending at the decision session (no
look-ahead). Take its last **252 sessions**, the eligible names (all of them, NaN-free by the engine). Pearson
correlation rho_ij, distance d_ij = sqrt(2 (1 - rho_ij)), **minimum spanning tree** (scipy). On the tree, with hop
distances: degree, betweenness (pairs whose path crosses the node), closeness (1 / mean hop distance), eccentricity
(largest hop distance) and eigenvector centrality (principal eigenvector of the adjacency). **Centrality = mean of
the percentile ranks of degree, betweenness, closeness, eigenvector and minus eccentricity** (Pozzi et al.'s hybrid,
equal weights). **Periphery = 1 - centrality**, as a percentile.

## 3. The arm

`Periferia-15`: score = 0.5 x percentile of the Composite score (attempt 51's, among the names that have one) + 0.5 x
percentile of periphery among the same names; then the Composite rule unchanged (`estrategia_por_nota`: 15 names, 4
per sector, equal weights, the fixed limits checked). Names without a Composite score stay out, as in the Composite.
One configuration, no search over the window, the tree or the 50/50 weight.

## 4. Engine and gate

`walk_forward_dinamico`, `CFG_BT` (1,260 sessions, every 63, 10 bps per side), phases 0/21/42, cut 16/10/2026,
frozen `docs/dados/2026-10-22/`, `2026-10-24-cvm/`, `2026-10-21-macro/`; the Conservadora must reproduce
17.41/16.41/16.90% and the Composite 21.4/19.9/19.1% first. **Gate, every phase, all >= 0.95:** P[> Composite],
P[> Conservadora], P[> BOVA11], DSR. Reported, not gates: CAGR, MDD, volatility, turnover, year table, the share of
picks the arm shares with the Composite, latest picks. A pass makes it a candidate for a forward paper log only.

**N.** One new configuration, one row in this branch's `docs/tentativas.csv` in this commit. The DSR uses the union
of every branch's trials, **N = 95** (94 of the KAN reading + this one).

## 5. Output

`docs/periferia-2026-11-07.txt` and a DECISOES entry. Nothing goes to the app.
