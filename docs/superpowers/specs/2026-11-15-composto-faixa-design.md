# The Composite with a holding band (pre-registered 2026-11-15)

The user, 15/11, asked whether the Composite's 15 names are fixed, how they change at each rebalance, and whether the
sector and volatility limits should be other numbers or another approach. The answer kept the numbers (sizes 5-30, no
sector cap, free size and other weights all lost, DECISOES 342-350) and offered one rule of a different kind, which the
user approved: a band on the exits. Written and committed before any number of it is read.

## The rule (one arm, `Composite-faixa-20`)

At each decision (the Composite's own dates, engine, data, costs and limits unchanged):

1. Rank the eligible names by the Composite's score (`scripts.qubo_universo_avaliar._value_score(..., com_crescimento=True)`,
   the Composite's), skipping a name without a sector and the second class of an issuer (the `carteira_da_nota` skips).
2. A name held from the previous accepted book stays when it is among the first 20 of that ranking.
3. The other slots, up to 15, are filled in score order by `carteira_da_nota`'s rules, the kept names counting in
   the sector (at most 4) and issuer counts.
4. Equal weights; the book goes through `checar_carteira` as the Composite's does (a refusal keeps the current book,
   and the band then compares with the book actually held).

The first decision has no previous book: it is the Composite's. Band 20 is fixed here; no other width is run.

## Reading

`scripts/composto_variantes_avaliar.py faixa`: phases 0 / 21 / 42, the anchor (the Composite's replay matches attempt
51's curve), the arm next to the Composite, the Conservadora, BOVA11 and the CDI, the turnover profile. N for the DSR =
189 (188 + this arm). Gate: P[> Composite] >= 0.95 and DSR >= 0.95 in every phase. If it passes, it becomes a forward
ghost account only; the official book (app and MT5 demo) stays the pure Composite.

## Expectation (before the run)

Lower turnover (fewer exits of names that slip from 15th to 16th-20th), CAGR within about half a point of the
Composite's either way at 10 bps a side, so no pass: the band saves costs the engine already prices low.
