# Ibovespa rebalance events: validation before any return (2026-11-08)

Spec `docs/superpowers/specs/2026-11-08-ibov-recomposicao-design.md`, build step 3. No price, return or volume around
any event was read for this note.

## Sources

- **CVM index-fund holdings** (`scripts/ibov_eventos.py`, `core/recomposicao.py`): 215 month-ends 2008-11 to
  2026-09, 48-100 names a month; 220 events, 119 entrants (2009-2026). Frozen in `docs/dados/ibov-recomposicao/`.
- **B3's dated preview news** (`scripts/ibov_previas.py`, Firecrawl search + structured extraction, two query passes,
  84 searches, 318 page reads of 69 unique pages, 0 extraction errors): a 3rd preview found for 19 rebalances (2015,
  2019-2026), a 1st preview for about 14 (2019-2026). `docs/dados/ibov-recomposicao/previas.csv` and the raw answers.

## A construction fix found here (before any return)

The index funds trade at the close of the old portfolio's last session, which is often the last day of the month
before the rebalance (01/09/2025 was a Monday: CEAB3 and CURY3 were already in the funds on 29/08). Comparing with
the month-end just before missed them. The composition before is now the month-end **two** months before (four
months separate rebalances). Test-first (`test_an_entrant_the_funds_bought_on_the_old_portfolio_s_last_session...`).

## Agreement, 3rd preview against the CVM entrants

- **36 of 39** 3rd-preview entrants are also CVM entrants. The three misses: BPAC11 (2019-09), SULA11 (2020-01),
  TEND3 (2026-09: the October month-end the two-month rule needs is not in the lake yet).
- CVM-only entrants are mostly corporate events, not forced buys: mergers and renames (ALOS3, NATU3, VBBR3), class
  migrations (AXIA7, CPLE3, RENT4), spin-offs (AMOB3), re-listings (SANB11, KLBN11, ENGI11), plus a few the found
  preview page did not list (ASAI3, SOMA3 in 2021-09).

## Amendment to the event list (fixed now, still before any return)

- **Where a B3 preview was found (2015-01, 2019-09 onwards):** R1's events are the 3rd preview's entrants and R2's the
  1st preview's, at their publication dates. The CVM list is not used for them.
- **Rebalances without a found preview (2009 to 2019-05, except 2015-01):** R1 uses the CVM entrants whose security
  traded at least 252 sessions before the rebalance (drops names born of mergers, spin-offs and class migrations, and
  recent IPOs: fewer events, cleaner), bought at the close of the old portfolio's penultimate month-session (the 3rd
  preview's rule date).
- Exit for both rules: the close of the last session before the new portfolio starts (the first B3 session on or
  after the first Monday of January, May or September).
