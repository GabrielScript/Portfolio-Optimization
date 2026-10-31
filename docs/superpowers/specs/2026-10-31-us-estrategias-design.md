# The Alepha strategies on US stocks

Pre-registered on 31/10/2026, before any strategy result on US data was computed. The question is the user's: on US
stocks, do the Composite, the Conservadora and the other strategies return more than the best funds?

## Data

Monthly total returns from the US lake (`exports/us_stocks`), a closed panel from 11/2016 to 09/2026:

- stockanalysis prices of the listed stocks and the dead names it still holds;
- Tiingo prices of the dead names it lacks, largest first, within the free plan;
- EDGAR first-filed statements and cover-page share counts, known from the filing date;
- Ken French's 12 industries as sectors (from the SIC code; "Money" is `Financeiro`), plus FF5 and momentum;
- FRED's CPI for the liquidity floor, and PTAX for the BRL comparison.

**Known limit:** the dead names are incomplete. The manifest and the output state how many arrived. Every result is
labelled with that count, and the test is rerun when more arrive. It does not count as a new attempt.

## Universe of each month

Common stocks on NYSE, NASDAQ or NYSE American, one security per company, without blank-check companies (SIC 6770).
A name enters if, in the 12 closed months before the decision, it traded in at least 11, its median daily dollar volume
reached US$ 1 million in today's money (deflated by the CPI known at the decision), its last close was at least US$ 1,
it traded in the last month, and no delisting was public. This is the B3 rule on a monthly grid.

## The strategies, with no change to their code

`Composto`, `Conservadora`, `Fórmula Mágica`, `Trending Value`, `Piotroski` and `Múltiplo do Adquirente`: the functions
already in `core/`, ordering 15 names at equal weights, 4 per sector at most, under the fixed risk rules. Equal weight
over the whole universe runs as a plain reference.

Kelly, minimum variance and HRP stay out of this run: they estimate from daily returns over 5 years, and the data has
no daily history for the dead names.

## Mechanics

- Monthly grid, a rebalance every 3 months, 36 months of training, 10 bps per side.
- Three phases: the first decision shifted by 0, 1 and 2 months, the analogue of the B3 test's 0, 21 and 42 sessions.
- 36 months instead of the B3 test's 60, because only 10 years exist. With 60, the out-of-sample period would start in
  2021 and lose the 2020 crash.
- Signals of decision month t read only months before t.
- A name that stops keeps its last close until the next rebalance, as in the B3 engine.

## What is measured

**Gate, the same as for the B3 strategies:** in all three phases, `P[> SPY] >= 0.95` and `DSR >= 0.95`, with N from
`docs/tentativas.csv`. The configurations were already counted for the B3, so they add nothing to N. This run is the same
configuration on another universe.

**Reported, not gates:** CAGR, volatility, Sharpe over the risk-free rate, maximum drawdown, turnover and the FF5 +
momentum alpha with its Newey-West t.

**Comparison in R$ through PTAX, from the same first decision:**

- SPY, QQQ, IWM and VTI;
- the 15 ETFs with the best trailing 12 months, rebalanced quarterly (point in time);
- the 15 ETFs and the 15 Brazilian equity funds with the best return over the period, labelled as hindsight;
- the 15 Brazilian equity funds with the best trailing 12 months, rebalanced quarterly (point in time);
- the B3 Composite and Conservadora, BOVA11, IVVB11 and CDI.

Point-in-time comparisons could be held in advance. Hindsight ones could not, and neither can be read as a promise.

## Sensitivities, read with the base case

1. Delisting: each name that stopped and was not acquired loses 30% in its last month (Shumway's average for
   performance delistings), since the data has no delisting return.
2. Dead names: the same run without the Tiingo names, which measures how much the fill changes.
3. Liquidity floor at US$ 5 million.

## What the result can decide

A pass makes the strategy a candidate for a forward log on US stocks. It does not move real money, because the panel has
fewer than 7 years out of sample and an incomplete set of dead names. A fail closes the US line for that strategy unless
a paid data month extends the test, which is a separate decision.
