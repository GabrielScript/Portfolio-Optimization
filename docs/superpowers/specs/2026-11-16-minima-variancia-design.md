# Minimum variance: code review and three pre-registered arms (2026-11-16)

The user's request: review the minimum-variance strategy of the strategies comparison rigorously and check whether its
return can be raised while keeping its low volatility. Written and committed before any arm reads real data.

## What the review found (read-only, phase 0, frozen exports to 16/10/2026)

The screen's arm is `estrategia_otimizada(replace(CFG_KELLY, objetivo=MIN_VOL), True, cdi)`: Ledoit-Wolf covariance of
1,260 sessions, `min w'Sigma w`, 20% per name, at most 10 names. CAGR 17.76%, volatility 15.15%, drawdown -38.9%,
turnover 1.30 a year (Composite, same window: 21.79%, 18.73%, -41.5%).

1. **No sector cap.** `CFG_KELLY.grupos` is empty and the arm never passes `checar_carteira`: utilities held 25-54%
   of the book at most decisions, above the house rule of 30% a sector. The screen's number is not a portfolio the
   rules would accept.
2. **The 10-name limit always binds.** The relaxed solution holds 18 to 41 names; the two-step heuristic keeps the 10
   largest and solves again. It costs 0.1 to 1.3 points of ex-ante volatility and is a limit chosen for Kelly.
3. **Jorion shrinkage is computed and thrown away** (`shrink_mu=True`; minimum variance ignores mu). Cosmetic.
4. **Stale prices attract it in the early years.** From 2010 to 2016 the weighted share of zero-return days of the
   held names was 5-7% against a universe median of 3% (CGAS5, COCE5, CTAX4, RIAA3): part of the measured low
   volatility there is thin trading, and 10 bps a side understates those names' cost. From 2023 on the bias is gone.
5. The solver, the constraint set, the weight clean-up and the engine are correct for this objective; no bug that
   changes a weight was found.

## Arms (N 197 -> 200)

All on the whole eligible universe of each decision (names with a sector), `CFG_BT` (1,260 sessions, every 63, 10 bps
a side), phases 0/21/42, the frozen exports of `scripts/qpo_montar` (to 16/10/2026), Ledoit-Wolf covariance, each
portfolio checked by `checar_carteira` (a violation keeps the current book).

- **MV-regras** (N 198): minimum variance with 20% a name, 30% a sector, at most 15 names (the Composite's count; the
  same two-step heuristic).
- **MV-nota** (N 199): MV-regras plus one linear constraint: the weighted mean of the names' Composite-score
  percentile (rank among the decision's names with a score; a name without a score counts 0) is at least 0.70.
  Fixed, not tuned. Infeasible = failure (the engine keeps the book).
- **Composite + MV 50/50** (N 200): half the Composite's weights and half MV-regras's at each decision
  (`estrategia_nucleo_satelite`).

Not counted: the screen's minimum variance (base) and the Composite, replayed in the same run.

## Criteria

- **Improves minimum variance** (the user's question): in every phase, CAGR above the base's, volatility at most the
  base's + 1.0 point, drawdown at most 2.0 points worse, and P[> base] >= 0.95 (PSR of the daily difference).
- **House gate** (anything beyond a fictitious account): P[> Composite] >= 0.95 and DSR >= 0.95 (N 200) in every phase.

One reading. Nothing is tuned after it; a different floor, count or mix is a new attempt.
