# Scenario Monte Carlo bench (five grades) — design (descriptive, no gate, no trial)

_2026-10-28, session 56. Step 2 of the session-56 roadmap, approved by the user. Written and committed before the grade
sequence or any simulation is computed. DECISOES 165. Not a strategy test: no gate, no row in `docs/tentativas.csv`,
nothing here changes the Alepha plan, the paper logs or the fixed rules._

## 1. Question

"If the market is in environment X, what has each building block done, and how likely is each one to beat the other
over 1, 3 and 5 years?" A decision aid built from history, not a forecast: it resamples the past and cannot add
information that is not in it.

## 2. Sleeves and data (all frozen, no network)

- **Conservadora** and **BOVA11**: phase-0 daily curves of `docs/dados/composto-curvas-2026-10-28.parquet` (net of the
  engine's 10 bps costs; taxes ignored). **Composto** (attempt 51) enters only if attempt 51 is a candidate.
- **CDI**: `docs/dados/2026-10-21-macro/cdi_daily.parquet` (`cdi_pct` = percent per day), compounded on the curve's
  sessions.
- **Grade inputs**: BOVA11's total-return index from `docs/dados/2026-10-22/universe_prices.parquet` (from 12/2008) and
  the same CDI file.
- IVVB11 is left out of this version: its history starts in 04/2014, which would cut the window by four years; it
  belongs to the allocation-layer work (step 3).

## 3. The five grades (fixed before looking at the sequence)

At each month end t, three components:

1. **trend** = BOVA11 total return over the last 12 months (sign +);
2. **vol** = annualized standard deviation of BOVA11's last 63 daily log returns (sign −);
3. **dcdi** = annualized CDI (`(1 + cdi_pct/100)^252 - 1`) at t minus the same 6 months earlier (sign −: falling rates
   count as favorable).

Score = z(trend) − z(vol) − z(dcdi), each z with the mean and standard deviation over the bench window, equal weights.
Grade = quintile of the score over the window: 5 Optimistic, 4 Neutral-optimistic, 3 Neutral, 2 Neutral-pessimistic,
1 Pessimistic. **The labeling is in-sample** (full-window thresholds): it describes what happened in each kind of
environment; it is not a timing signal. A month's grade is the grade at the end of the month before (the state when
the month starts).

Window: the complete calendar months inside the curve (03/2010 to 09/2026). "Today's grade" is computed with the same
thresholds from the data up to 16/10/2026 (before the 26/10 crash), reported as a description.

## 4. Simulation

- Monthly Markov chain of grades: transition matrix = row-normalized counts of grade(m) -> grade(m + 1) in the window.
- A path starting in grade g: month 1 is a whole historical month drawn uniformly among the months that started in
  grade g; the next grade comes from the transition row; repeat for 60 months. Every draw takes **the same historical
  month for all sleeves**, so the co-movement of real crashes is kept as it happened.
- **Change from the chat:** the session-56 answer mentioned a t-copula; this version resamples whole joint months
  instead, which keeps the observed crash co-movement without fitting a dependence model.
- 10,000 paths per starting grade, seed 20261028, float64 on month summaries (no daily path is stored): each month
  is summarized per sleeve by its log return, its running minimum and maximum, and its internal drawdown, which give the
  path's maximum drawdown exactly.

## 5. Output (`docs/bancada-cenarios-2026-10-28.txt`)

1. The grade thresholds, months per grade, transition counts, and each sleeve's mean monthly return per grade.
2. Per starting grade and horizon (12, 36, 60 months): P[Conservadora > BOVA11], P[Conservadora > CDI],
   P[BOVA11 > CDI] (and the Composto's, if it entered); terminal return 5/50/95% quantiles per sleeve; maximum
   drawdown median and 5% worst per sleeve.
3. Today's grade and its three components.

Caveats printed in the header: in-sample labeling; history is resampled, not forecast; a sampled month brings its own
CDI level (2% in 2020, 14% in 2016), so comparisons inside a path are consistent but terminal levels are not "what to
expect from today"; about 40 months per grade, so each row is noisy; taxes ignored.

**Out of scope on purpose:** no mixes of sleeves, no leverage multipliers, no "best weight per grade". Those are the
allocation-layer test (step 3) and the leverage decision (step 5), each with its own pre-registration.
