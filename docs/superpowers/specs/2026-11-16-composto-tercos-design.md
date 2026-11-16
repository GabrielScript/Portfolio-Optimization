# The Composite in three staggered thirds (pre-registered 2026-11-16)

The user, 16/11, asked which rebalance interval the Composite uses (63 sessions) and was going to propose shorter
ones; the answer (DECISOES 346: every 21 sessions gives 21.2 / 21.1 / 21.2% against 21.4 / 19.9 / 19.1%, turnover
5.5 against about 3, fails the gate) offered a rule that removes the rebalance-date luck without the monthly turnover.
The user approved it ("pode registrar e rodar o teste das 3 partes"). Written and committed before any number of it
is read.

## The rule (one arm, `Composite-tercos`)

One book, three sleeves of one third of the capital each. Sleeve k (k = 0, 1, 2) is the Composite exactly as it runs
today (engine, data, score, 15 names, 4 per sector, equal weights, limits, 63-session rebalance, 10 bps a side),
started k x 21 sessions after sleeve 0: sleeve k is the Composite's phase 21k curve, replayed in this run by today's
code. Before its first decision a sleeve's third earns the CDI. The sleeves are never rebalanced against each other:
each one's value floats; no cross-sleeve netting of trades is credited (each sleeve pays its own costs, conservative).

Book value at t = (1/3) sum_k S_k(t), S_k(t) = G(t) / G(t0) before sleeve k starts at s_k, and
G(s_k) / G(t0) x V_k(t) / V_k(s_k) after, with G the CDI accumulation and V_k the sleeve's replayed curve.
The book is built from phase 0's first session; no other number of sleeves or offset is run.

## Reading

`scripts/composto_tercos_avaliar.py`: replays the Composite's phases 0 / 21 / 42 (the replay's gap to attempt 51's
saved curve is printed, not enforced, as in the 15/11 band addendum), builds the book, and reads it against the
Composite of each phase on that phase's dates (book restricted to them), with the Conservadora, BOVA11 and the CDI.
N for the DSR = 190 (189 + this arm). Gate: P[> Composite] >= 0.95 and DSR >= 0.95 in every phase. If it passes, it
becomes a forward ghost account only; the official book (app and MT5 demo) stays the pure Composite.
Descriptive, no gate: the book's CAGR next to the three phases' spread and mean, its turnover (mean of the sleeves').

## Expectation (before the run)

CAGR near the mean of the three phases (about 20-20.5%), below phase 0's and above phase 42's, so P[> Composite]
low in phase 0 and higher in phase 42: it fails the gate. What it buys is the removal of the date luck at the
Composite's own turnover, which the gate is not built to reward; a fail does not argue against running the real book
in thirds, which stays the user's operational choice (lot sizes at the current account may not fit three sleeves).
