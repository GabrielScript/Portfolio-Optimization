# The Composite with a BDR sleeve: 95/5 to 80/20, top-1 to top-10 (pre-registered 2026-11-14)

The user, 14/11: "Se usarmos as proporções 95/5, 90/10, 85/25 e 80/20 agora para as BDR's (Temos 902 no total) com
maior crescimento estável e consistentes a partir de 2021 ... ou melhor pegar a BDR vencedora, top-3, top-5 e até
top-10 nesse intervalo de 5 anos desses ativos com os maiores retornos e com menor volatilidade possível". "85/25" is
read as 85/15 (the four mixes step by 5 points).

Written and committed before any number of these arms is read.

## Arms (16 rows)

`Comp{100-w}+BDR-top{k}`: w in {5, 10, 15, 20}% of the book in a BDR sleeve, k in {1, 3, 5, 10} names in it; the
rest is the Composite. The Composite alone is the baseline.

## The BDR sleeve, decided month by month with the past only

Picking the names with the best 2021-2026 record and testing them on 2021-2026 would read the future: its backtest is
good by construction and decides nothing. The sleeve is therefore chosen at each month with the 12 months before it.

- Universe: every security of the lake's gold table (`D:/b3-datalake/gold/equity_adjusted.parquet`) with
  `asset_type == "bdr"`, delisted ones included (1,265 since 2020); prices `tr_close`.
- Decision: the close of each month's first session in the window, with data through that close; the book trades at
  that close and holds from the next session (the 80/20 mix's rule, `cryptolake.mix_long.sleeve_daily`).
- Eligible: in the 252 B3 sessions through the decision, traded on at least 90% of them and a median daily volume of
  at least R$ 1 million (the 12/3 ghost's rule, DECISOES 299). Without it the ranking picks thin BDRs whose stale
  prices show a volatility near zero: only 37 BDRs have a median volume above R$ 1 million over their whole history.
- Score: the 252-session total return over the annualized volatility of the daily returns of the same sessions
  ("os maiores retornos com menor volatilidade"); the top k by score, ties by ticker; equal weights in the sleeve.
- Fewer than k eligible: the sleeve holds those; none: the whole book is the Composite that month.

## The mix

- Composite leg: the saved Composite curve of each phase (`docs/dados/pos-idade-curvas-fase{0,21,42}-2026-11-13.parquet`,
  the baseline after DECISOES 360), as in the 80/20 tables.
- At each decision the book goes back to (1 - w) Composite and w sleeve, each sleeve name w / k.
- Costs: 50 bps of each BDR trade's value; 10 bps of the Composite leg's traded value.
- Tax: 15% of each BDR sale's gain over its average cost (BDRs have no R$ 20 thousand exemption; losses pay nothing and
  are not carried, as in the 12/3 ghost). The Composite leg pays none (a small account stays under the exemption).
  The headline is net of this tax; the pre-tax CAGR is reported beside it.
- A sleeve name without a price on a session keeps its last price until the next decision.
- Known gap: the lake has no BDR cash dividends, so `tr_close` of a BDR is its price only (the sleeve's return is
  understated by the dividend yield, about 1% a year for US large caps).

## Window and phases

Sessions from 04/01/2021 (BDRs open to retail investors since 10/2020; the first decision has 12 months behind it) to
16/10/2026 (the end of the Composite curves). Phases 0 / 21 / 42 are the Composite curve of each phase; the sleeve's
calendar is the same in every phase.

## Measures and gate

`scripts.xquad_avaliar.metricas` (CAGR, Vol, Sharpe and Sortino over the CDI, MDD, Calmar, P[> Composite], DSR, alpha
and t against the Composite and BOVA11); CDI from `data/b3lake/cdi_daily.parquet`; BOVA11 from the gold table.
N for the DSR = `contar_tentativas()` + 16. Gate per arm, net of tax: P[> Composite] >= 0.95 and DSR >= 0.95 in every
phase. An arm that passes becomes a forward ghost account only; none replaces the Composite.

## Expectation (before the run)

DECISOES 298 (k of 15 slots in 32 US names, 12-1 momentum) beat the Composite by 4 to 13 points a year in 2021-2026
without passing the gate. This sleeve is smaller (5-20% against 7-33%), ranked by return over volatility (which favors
steadier names and BDRs of ETFs) and pays the tax. Expected: CAGR above the Composite by about w times the sleeve's
excess return, P[> Composite] below 0.95 in at least one phase, no arm passing.
