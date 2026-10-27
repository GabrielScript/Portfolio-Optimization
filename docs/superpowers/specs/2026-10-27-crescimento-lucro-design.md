# Profit-growth signal: IC test — design and pre-registration

_2026-10-27, session 55. Written and committed while attempt 48 (`2026-10-27-qubo-universo-valor-design.md`) was
running and **before its output was read**. DECISOES 157._

## 1. Why this signal

The QPO's stopping rule (DECISOES 155-156) allows a new attempt only with new information in mu that passed its own
IC gate. The candidate comes from evidence outside the test period: in JKP Global Factor Data, **profit growth** is
the second strongest of the 13 themes in Brazil before 2010 (t = 1.77, 9.7%/yr, 184 months), and it is strong in
emerging markets before 2010 (t = 2.44) and in the world ex-US (t = 4.71). No signal of this family exists in the
project yet (the Conservadora is low volatility, dividend yield and momentum; value is the Magic Formula and the
Trending Value; Piotroski mixes levels and changes).

## 2. The signal (fixed now)

For month m and each issuer, from the frozen CVM folder `docs/dados/2026-10-24-cvm/` (DFP 2010+ and ITR 2011+,
12-month flows, each value dated by its receipt at the CVM):

- t = the latest `dt_refer` whose documents were received before m (`available_from < m`); t-1y = `dt_refer` one
  year earlier for the same issuer (same quarter end).
- `dNI = (net_income(t) - net_income(t-1y)) / total_assets(t-1y)` (the JKP `niq_at_chg1` family);
  `dGP = (gross_profit(t) - gross_profit(t-1y)) / total_assets(t-1y)` (gross profitability's change,
  Novy-Marx 2013).
- Score = the mean of the two measures' normal scores (`core.views.escore_normal`) over the month's candidates, with
  the ones available (the S3 rule for value); an issuer's classes share its score.
- Excluded: total assets <= 0 at t-1y; a missing t-1y filing; flows not known at m.

## 3. The test (the project's S2 gate)

Monthly Spearman IC of the score against the month's total return, in the month's universe, **2012-01 to
2026-09** (`ics_e_cobertura` of `scripts/avaliar_valor.py`, the same harness and period as DECISOES 79). **Gate:
mean IC >= 0.02 and t >= 2** (the power-checked gate of DECISOES 45/`e29d5db`). Reported: the IC by subperiod
(2012-16, 2017-21, 2022-26), months with IC > 0, names scored per month, and the correlation of the score with the
Conservadora's and the value score's normal scores (the overlap that would dilute it in a composite).

An IC test is not a portfolio configuration, so `docs/tentativas.csv` does not change here.

## 4. What follows

- **Gate passed:** a QPO attempt with the composite of the Conservadora, value and profit growth (equal weights of
  the three normal scores, IC0 = 0.059, everything else as attempts 47-48) is pre-registered next, one trial row.
- **Gate failed:** the QPO line stays closed; no further mu attempt without another signal that passes this gate.
- Expectation, written before: the gate is likely passed at a modest IC (0.02-0.04), since earnings-change signals
  are among the most replicated anomalies; whether it survives inside the QUBO is a separate question.
