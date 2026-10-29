# Fund managers' best ideas from the CVM CDA — design and pre-registration (IC test, then attempt 52)

_2026-10-29, session 59. Written and committed before any code of the signal, any score panel or any IC exists (only the
file layout and the confidentiality lags were probed). User's /goal step 2. DECISOES 175 (the gap), 176 (NPY stopped,
so N is still 51)._

## 1. Evidence and the idea

Cohen, Polk and Silli (2010, "Best ideas"): a manager's most overweighted stock (against the market) outperforms;
the rest of the portfolio does not. FGV studies on Brazil's CDA: best ideas of 2,201 equity funds, 2009-2016, beat the
IBX 100 by 0.69%-2.33% a quarter; the aggregate of all holdings trailed it by 0.20%. Here: the share of Brazilian
equity funds that hold a stock among their three largest active weights, known point in time.

## 2. Data (fixed now)

The lake's `fund_stock_holdings.parquet` (CVM CDA block 4, "Ações" owned and lent: cnpj, portfolio month-end, ticker,
value, confid_until) and `fund_nav.parquet` (cnpj, month-end, nav), 2005-01 to 2026-09, from branch `feat/cda-ptax`
(commit `46309de`); the export's sha256 is printed by the panel builder. Market caps:
`docs/dados/2026-10-29-cvm/market_caps.parquet`; universe and sectors: `docs/dados/2026-10-22/`. The issuer of a ticker
is `core.risco.emissor` (its first 4 letters); tickers whose issuer has no market cap (BDRs, units of other issuers,
foreign) are dropped.

**Point in time.** `DT_COMPTC` is not a publication date. Rules: a public position of month-end t is known from the 1st
day of month t+2 (funds file within 10 days); a confidential one from the 1st day of the month after
`confid_until + 10 days` (lags measured: <= 90 days to 2022, up to 180 days in 2024). A fund's portfolio t is usable at
signal month m only when **all** its positions are known by m (a partial portfolio would hide exactly the positions
being built). For each fund, m uses its latest usable t, if t >= m - 12 months. A position still confidential on the
download day (29/10/2026) is not in the files at all (the 2026-07..09 zips hold ~13k positions against ~42k before),
so portfolios dated after **30/04/2026** (within 180 days of the download) are not used: they may look complete
without being so. Export sha256: positions `cd997a5d06f1b41b4988ef0da06518c1a032768380bd2ebee350af5e78b14feb`,
navs `e12d47008b06a5a7eca999c0e00c3554bda862dfaceeacbb6f8893a4896121ee` (10,626,115 and 3,868,910 rows).

## 3. The signal

For each fund f and its portfolio t, with the benchmark of t: issuers of the universe of month t+1 (`universe_monthly`,
the month after the portfolio date) weighted by `market_cap` of row t+1 (the last close of t).

- `w_f(i)` = f's value in issuer i (classes summed) over f's value in benchmark issuers; `w_b(i)` = the cap weight.
- **Qualifying fund at t:** nav >= R$ 10 million; stock value / nav >= 0.67 (an equity fund by the CVM rule);
  >= 10 distinct benchmark issuers; active share `0.5 * sum |w_f - w_b|` >= 0.5 (no closet indexers).
- **Best ideas of f:** the 3 issuers with the largest `w_f - w_b` (ties: the larger `w_f`, then the issuer code).
- **Score(i, m)** = the number of qualifying funds (each with its latest usable t) naming i among their best ideas,
  over the number of qualifying funds; issuers of month m's universe not named score 0; a security takes its issuer's.
- Reported, not used: the consensus active weight (mean of `w_f - w_b` over qualifying funds), the contrast the FGV
  study found negative.

The builder writes `docs/dados/2026-10-29-cda/best_ideas_panel.parquet` (month, issuer, score, consensus,
qualifying funds) with a `SHA256SUMS` line; the IC and the attempt read only that frozen panel.

## 4. Stage A: IC test (read once)

Monthly Spearman IC of the score against the month's total return in the month's universe, **2011-01 to 2026-09**
(the first portfolios with market caps are 2010-06/07, usable from ~2010-12). Months with fewer than 30 qualifying
funds are dropped and counted.

- **Gate (the project's S2 gate):** mean IC >= 0.02 and t >= 2.
- Reported: IC by subperiod (2011-15, 2016-20, 2021-26), months with IC > 0, qualifying funds per month, names with
  score > 0 per month, the consensus IC, the rank correlation with the Conservadora score.
- **Fail:** stop; no attempt row; display only at most.
- **Pass:** stage B after its own commit adding the trial row (N 51 -> 52).

## 5. Stage B: attempt 52 (read once, only after stage A passes)

`BestIdeas-15`: `core.sinais.carteira_da_nota` on the score with ties broken by the Conservadora score (score +
1e-6 x the Conservadora's normal score: a tie-break only), 15 names, 4 per sector, one class per issuer, equal weights,
the fund's fixed rules (refusal keeps the portfolio). Engine, phases, data cut, CDI, plumbing check and gate exactly as
attempt 51 (`scripts/composto_crescimento_avaliar.py`): P[> Conservadora], P[> BOVA11] and DSR >= 0.95 in all 3 phases.

- **Candidate:** pre-register a forward paper log against the same-day Conservadora.
- **Not a candidate:** the line is closed.

## 6. Expectation, written before

The published edge used fresher holdings (45-60 days); ours are 5-7 months old by the confidentiality rule, and the
liquidity universe holds the stocks funds crowd into. Expected: IC small and positive (0.01-0.03), t between 1 and
2.5, so the gate is close to a coin flip; if stage B runs, not a candidate (P[> Conservadora] < 0.95 in some phase).
