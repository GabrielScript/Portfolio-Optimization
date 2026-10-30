# Copy the top-return equity funds (CVM daily report + CDA) — design and pre-registration (IC test, then attempt 54)

_2026-10-29, session 60. Written and committed before any fund return, ranking, score panel or IC of this signal
exists. What was looked at: the daily report's file layout (headers of the 2015 and 2026-09 zips), row and fund
counts per zip, the median days reported per month, and the export's total rows (3,930,291 fund-months, 2000-2026);
no quota, return or ranking was read. Export `fund_monthly_quota.parquet` (lake branch `feat/inf-diario`, `1b62b75`)
sha256 `296fd03dcb36e250bcf12664ce77085414bbfa91b2a0903aad2e801a711dfbab`. User's idea (PLAN §16 item 2, "the big
idea"): follow only the funds with the best returns and copy their positions and weights. N = 53 now._

## 1. The idea, the evidence and what is already known

Copying funds needs two things: which funds are good (their returns) and what they hold (their portfolios).
- **Holdings:** CVM CDA, the same files and point-in-time rules as attempt 52 (5-7 months old when public; 34-85% of
  stock value confidential at filing).
- **Fund returns:** CVM's daily report (informe diário), quota per fund and day, filed the next business day, closed
  funds included (no survivorship).

Evidence: fund performance persists only weakly. Carhart (1997): one-year persistence in US funds is mostly stock
momentum and costs, gone after a year; the evidence for Brazil is thin. Holdings-based copying (Frank, Poterba,
Shackelford and Weisbenner 2004, "copycat funds"): portfolios built from the last disclosed holdings earned returns
close to the actual funds' with 1-quarter-old holdings; ours are 2-3 times older.

**Known before this design (data snooping, disclosed):** attempt 52 (all qualifying funds' best ideas) passed its IC
gate (IC 0.032, t 3.75) and failed as a portfolio (P[> Conservadora] 0.12/0.13/0.05); the consensus active weight of
all qualifying funds had IC 0.000. Two questions, **both** must be answered yes (gate §4): (1) does restricting to the
top-return funds add anything over the same construction on all funds? (2) does it beat the best fund-based signal
already measured (attempt 52's 0.032)?

## 2. Data (fixed now)

- `fund_monthly_quota.parquet` (lake export of `b3lake.cli inf-diario`): cnpj, subclass, month, last report day,
  quota, nav, holders, inflow, outflow, days. Its sha256 is written in the panel builder and printed; the panel is
  built once and frozen (`docs/dados/2026-10-29-top-fundos/`, `SHA256SUMS`).
- CDA positions and navs, market caps, universe: exactly attempt 52's files and sha256 (spec
  `2026-10-29-cda-best-ideas-design.md` §2), the same `core.melhores_ideias` functions (`publico_a_partir`,
  `carteira_usada`, `ideias_por_carteira`, `referencia`), the same cut (portfolios after 30/04/2026 not used). The
  lake's nightly run does not include the `cda` or `inf-diario` steps; the builder refuses any file whose sha256
  differs from the one written here or in attempt 52's spec.
- CNPJ: both files write it as `00.000.000/0000-00`; matched as text. The share of qualifying funds with a quota
  history is printed (coverage), never a return.

**Fund monthly return.** For fund f and month-end m: `r_f(m) = quota_s(m) / quota_s(m-1) - 1`, always within **one**
series s (never one subclass's quota over another's: different fees). s = the class row (subclass null) when it
exists at both m and m-1; otherwise (CVM 175 classes that report only subclasses) the subclass with the largest nav at
m-1 among those with a quota at both m and m-1 (a rule that does not look at returns). A month's quota counts only if
its last report day is within the last 7 calendar days of the month; otherwise r_f of that month and the next is
missing. A monthly return below -60% or above
+150% is treated as a data error (missing). Quotas are net of fees: ranking is on what the fund's holders earned.

## 3. The signal

At signal month m (the first day of m), for each fund f qualifying under attempt 52's rules at its latest usable
portfolio t (`carteira_usada`, same 12-month validity):

1. **Trailing return** `R_f(m)` = compounded r_f over the 12 months m-13 .. m-2, skipping m-1: the report of m-1's
   last day is filed on the next business day, which may be after the first session of m, so m-1 is not used (no
   one-day leak). **One window only (12 months, Carhart's ranking window); no other window is computed.** Funds with
   any of the 12 months missing are not ranked.
2. **Top funds** `T(m)`: the 15 ranked qualifying funds with the highest `R_f(m)` (ties: larger nav at m-2, then cnpj).
3. **Score(i, m)** = the mean over f in T(m) of `w_f(i)` at f's portfolio t (issuer weight within f's benchmark
   issuers, exactly `ideias_por_carteira`'s `pesos`; 0 when f does not hold i). This is the copy of their positions
   and weights. Issuers of month m's universe not held by any top fund score 0; a security takes its issuer's score.
4. The baseline `Score_all(i, m)` = the mean of `w_f(i)` over **all** ranked qualifying funds (used by gate (c)).
   Reported only: the same mean of `w_f - w_b` over T(m) (top-15 active weight).

Months with fewer than 30 qualifying funds or fewer than 15 ranked funds are dropped and counted.

## 4. Stage A: IC test (read once)

Monthly Spearman IC of `Score` against the month's total return in the month's universe, **2011-01 to 2026-09**,
computed exactly as attempt 52's `scripts/cda_ic.py` (`core.poder.ic_por_mes`, `resumo_do_ic`; same t statistic).

- **Gate (all three):** (a) the project's S2 gate, mean IC >= 0.02 and t >= 2; (b) **mean IC >= 0.032**, attempt
  52's all-fund best-ideas IC (question 2); (c) the monthly paired difference `IC(Score) - IC(Score_all)` has a
  positive mean with t >= 2, same t statistic (question 1: the top-15 restriction itself must add something).
- Reported: IC by subperiod (2011-15, 2016-20, 2021-26), months with IC > 0, months read and dropped, funds ranked per
  month, the IC of `Score_all` and of the top-15 active weight, the rank correlation with the Conservadora score and
  with market cap (weights copied from funds lean to large caps: if the IC is mostly size, it says so).
- **Fail:** stop; no attempt row; the copycat line is closed. **Pass:** stage B after its own commit adding the trial
  row (N 53 -> 54).

## 5. Stage B: attempt 54 (read once, only after stage A passes)

`TopFundos-15`: `core.sinais.carteira_da_nota` on `Score` with ties broken by the Conservadora score (score + 1e-6 x
the Conservadora's normal score), 15 names, 4 per sector, one class per issuer, **equal weights**, the fund's fixed
rules (refusal keeps the portfolio). Engine, phases, data cut, CDI, plumbing check and gate exactly as attempts 51-52
(`scripts/cda_best_ideas_avaliar.py` with this score): P[> Conservadora], P[> BOVA11] and DSR >= 0.95 in all 3 phases.
Months the panel does not cover: every score is 0, so the Conservadora order (as attempt 52's addendum).

The funds' own weights are copied into the **ranking**, not into the position sizes: the fund's fixed rules (equal
weights, sector cap, 15 names) are what every attempt is judged under. Copying the sizes as well would be another
portfolio rule, so another trial (N + 1), only if the user asks after this one.

- **Candidate:** pre-register a forward paper log against the same-day Conservadora.
- **Not a candidate:** the line is closed.

## 6. Expectation, written before

Weak persistence, 5-7-month-old holdings and the top-15 being a noisy subset (a 12-month winner is often a sector or
small-cap bet that already paid) point to an IC no better than the all-fund signals: expected 0.00-0.03, so gates
(b) and (c) most likely fail. The test is worth running once because the user's hypothesis is specific and the data are new;
whatever the outcome, there is no second window, no other top-N and no rerun.

## 7. Files (to be written after this commit)

`scripts/top_fundos_painel.py` (frozen panel; prints coverage only), `scripts/top_fundos_ic.py` (stage A, read once:
`docs/top-fundos-ic-2026-10-29.txt` or the date it runs), `core/top_fundos.py` (fund returns, ranking, score; tests
first). Stage B harness only after the gate.
