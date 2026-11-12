# Stock lending, part A: the lender's rate on the Composite's book (measured 2026-11-12)

A measurement, not a trial (N does not move). Spec: `docs/superpowers/specs/2026-11-12-aluguel-design.md`.

Book: the 15 equal-weighted names of `paper-v3/composto.csv` (2026-10-27). Data: the lake's
`bdi_BTBLoanBalance`, 25 sessions, 2026-10-05 to 2026-11-10; per name and session the lender's average rate,
value-weighted across the markets with a contract that day; read as a fraction a year (B3's FAQ: rates are annual).

| name | mean % a year | min | max | sessions with a contract | BRL mi lent a day |
|---|---|---|---|---|---|
| ALLD3 | 4.93 | 4.87 | 5.00 | 25 | 0.05 |
| FIQE3 | 2.70 | 2.56 | 2.96 | 25 | 0.48 |
| JHSF3 | 0.45 | 0.37 | 0.75 | 25 | 3.17 |
| UGPA3 | 0.32 | 0.20 | 0.44 | 25 | 3.31 |
| BBSE3 | 0.30 | 0.15 | 0.53 | 25 | 7.57 |
| MLAS3 | 0.29 | 0.27 | 0.34 | 25 | 0.33 |
| CXSE3 | 0.15 | 0.09 | 0.33 | 25 | 6.06 |
| LOGG3 | 0.14 | 0.13 | 0.14 | 25 | 0.20 |
| VBBR3 | 0.09 | 0.07 | 0.10 | 25 | 2.40 |
| ITSA4 | 0.07 | 0.04 | 0.10 | 25 | 4.94 |
| PETR4 | 0.06 | 0.03 | 0.20 | 25 | 17.03 |
| LEVE3 | 0.06 | 0.01 | 0.13 | 25 | 0.16 |
| PSSA3 | 0.05 | 0.02 | 0.09 | 25 | 1.21 |
| MILS3 | 0.05 | 0.01 | 0.13 | 25 | 0.30 |
| SAUD3 | 0.05 | 0.05 | 0.05 | 25 | 1.46 |

- Equal-weighted book: **0.65% a year gross**; the median name pays 0.14%; without ALLD3 and FIQE3 the book pays
  0.16%. Two names give three quarters of the income, and ALLD3's whole market lends about BRL 50 thousand a day.
- Every name had a contract on every session, but a contract in the market is not a taker for our shares.
- Before it reaches the account: the broker's share of the rate (the user's contract), income tax at source as
  fixed income (22.5% up to 180 days), and a lent share is back only on D+2 or D+3 after the recall.
- Reading: an upper bound of about 0.2 to 0.6 of a point a year, as the spec's prior said. Not a change of scale.

Part B (the signal) stays blocked on step 0: no source with years of per-ticker history was found; B3 serves 21
days. The lake keeps accumulating.
