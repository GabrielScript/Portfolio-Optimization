# Option signals as a veto on the Conservadora (forward from put-call parity) — design and pre-registration (IC test, then attempt 54)

_2026-10-29, session 61. Written and committed before any forward, implied volatility on the forward, option signal
or IC of these signals exists. What was looked at: the lake's `b3.option_greeks` and `b3.option_daily` schemas and
`b3lake/option_greeks.py`; one coverage count (underlyings with a same-day call and put at the same strike and expiry
on the last session of each month, per year: below 10 before 2011, 12-18 in 2012-16, 26-55 in 2017-20, 83-122 from
2021); no option price, implied volatility, signal or stock return was read for this design. The existing COTAHIST
greeks were checked against opcoes.net.br on one day (DECISOES 181: median |IV gap| 0.84 pt calls, 1.55 pt puts).
User's request (PLAN §16 item 4, §18 item 1): implied-volatility level, skew and put-call volume as a veto or
meta-label on the Conservadora, never expected returns; first, the put-call parity forward per expiry as the
greeks' spot. N = 53 now._

## 1. Why, and what is already known

The fund's rules bar trading derivatives (DECISOES 176), so options can only inform the stock choice. The question:
does the option market know something about a stock's next month that the Conservadora's score (low 36-month vol,
12-month dividend yield, 12-1 momentum; Blitz and van Vliet 2018) does not? If yes, the cheapest use is a veto: skip a
Conservadora pick the option market flags.

Evidence (US): steep put skew predicts lower stock returns (Xing, Zhang and Zhao 2010, JFQA); a high put share of
option volume predicts lower returns, mostly at a one-week horizon and from open-buy volume we do not have (Pan and
Poteshman 2006, RFS); implied volatility level behaves like total volatility (low-vol anomaly; Ang, Hodrick, Xing and
Zhang 2006), which the Conservadora already uses. Brazilian evidence is thin, and B3 single-stock options are liquid in
a few dozen names (the coverage count above).

**Why the forward.** Today's greeks use Black-Scholes on the stock's same-day close with no dividends (q = 0): the
implied volatilities near ex-dates are biased, and a put and a call at the same strike give different volatilities.
The forward implied by put-call parity, `F = K + (C - P) e^{rt}`, carries the market's own expected dividends and
borrow cost; Black-76 on `F` removes both biases. Skew in particular needs it: a dividend before expiry looks like skew
when priced off the spot.

## 2. Lake work (b3-datalake, built test-first in a worktree before any signal exists)

1. **Forward per (underlying, date, expiry)** from `b3.option_daily` (is_exercise = 0, the options' own rows), for
   expiries with 5 or more business days left. Pairs = strikes where a call and a put both have a row that day. Each
   leg's price = the mid of the closing best bid and best ask when 0 < bid <= ask **for both legs**, otherwise both
   legs' closes (the closing quotes are nearer in time to the stock's close than a last trade; the source is
   recorded). `F_k = K + (C - P) e^{r t}`, with r and t exactly as today's greeks (rate = 252 ln(1 + CDI of the day),
   t = business days / 252). Pairs with K / S outside [0.8, 1.2] (S = the spot close used today) are not used.
   `forward` = the median of `F_k`; `pairs` = their count; no pair = null. New table `b3.option_forwards` (date,
   underlying_issuer, underlying_isin, expiry, du, rate, spot, forward, pairs, source).
2. **Black-76 on the forward** in `b3.option_greeks` (new nullable columns `forward`, `iv_fwd`, `delta_fwd`; a
   migration adds them, then every year is rebuilt, one job at a time): call = e^{-rt}[F N(d1) - K N(d2)], put by
   parity, d1 = [ln(F/K) + sigma^2 t / 2] / (sigma sqrt t); the same bisection bracket as today; `delta_fwd` = N(d1) for
   calls and N(d1) - 1 for puts (forward delta). Rows without a forward keep the old columns and have null `_fwd`.
3. **Checks (descriptive, no gate):** implied dividend-plus-borrow yield `r - ln(F/S)/t` by year (median and 5-95%),
   expected roughly 0-10% a year; the opcoes.net.br IV gap recomputed with `iv_fwd` beside the old one.
4. **Monthly signal export** `option_signals_monthly.parquet` (sha256 printed and written in the app's panel builder),
   one row per (issuer, month m = first day), computed only from sessions of month m-1:
   - Daily, per underlying: the **reference expiry** = the one with a forward whose business days left are nearest
     to 21 within [10, 60] (ties: the earlier expiry).
   - `iv_atm(d)` = the mean of the `iv_fwd` of the call and the put (those that exist) at the strike nearest to F
     with K / F in [0.95, 1.05], on the reference expiry.
   - `skew(d)` = `iv_fwd` of the put with K / F nearest to 0.95 within [0.80, 0.97] minus `iv_fwd` of the call with
     K / F nearest to 1.00 within [0.95, 1.05], same expiry (Xing-Zhang-Zhao's definition, on the forward).
   - Monthly: `iv_atm` and `skew` = the medians of the daily values over the last 5 sessions of m-1 that have them
     (at least 3, else missing); `pc_ratio` = put contracts / (put + call contracts) summed over all expiries in the
     last 5 sessions of m-1 (missing when the sum is 0); reported only: `rv21` (annualized realized vol of the stock's
     last 21 daily total returns of m-1) and `n_days`.

## 3. The signals and their direction (fixed now)

Each is ranked so that higher = expected better, per the literature:
- **S1 implied-vol level:** `-iv_atm` (low implied volatility ranks high).
- **S2 skew:** `-skew` (a flat skew ranks high).
- **S3 put-call volume:** `-pc_ratio` (a low put share ranks high).
Reported only, not gated: `iv_atm - rv21` (Bali and Hovakimian 2009's spread, sign as theirs) and each signal's rank
correlation with the Conservadora score and with market cap.

A security takes its issuer's signal (both PETR3 and PETR4 take PETR's), as attempt 52's `nota_do_painel`.

## 4. Stage A: IC test (read once)

Monthly Spearman IC of each signal against the month's total return in the month's universe, exactly as
`scripts/cda_ic.py` (`core.poder.ic_por_mes`, `resumo_do_ic`; frozen inputs `docs/dados/2026-10-22/`, prices cut at
16/10/2026), **2008-01 to 2026-09, months with at least 20 issuers having the signal** (each signal its own months;
the count of months read and dropped is printed). Three signals, so the t threshold is Bonferroni-adjusted.

- **Gate, per signal, both:** (a) mean IC >= 0.02 and t >= 2.4; (b) the mean monthly **partial** Spearman IC of the
  signal given the Conservadora score, `(r_sy - r_sc r_yc) / sqrt((1 - r_sc^2)(1 - r_yc^2))` on the names with the
  signal, the score and the return, >= 0.02 with t >= 2.4 (the veto must add something the Conservadora does not
  already know).
- Reported: IC and partial IC by subperiod (2008-15, 2016-20, 2021-26), months with IC > 0, issuers per month, and the
  share of the Conservadora's 15 picks that have each signal (a veto that touches no pick does nothing).
- **Fail (all three):** stop; no attempt row; option signals stay display-only at most. **Pass:** stage B only for the
  passing signal with the largest partial-IC t (one attempt, not three), after its own commit adding the trial row
  (N 53 -> 54; the top-funds test of DECISOES 186 stopped at its stage A and never took row 54).

## 5. Stage B: attempt 54 (read once, only after stage A passes)

`Conservadora-veto`: at each decision, the Conservadora score with every candidate whose signal is in the **bottom
quintile** of that month's names with the signal removed (names without the signal are not vetoed), then
`core.sinais.carteira_da_nota` as the Conservadora (15 names, 4 per sector, one class per issuer, equal weights, the
fund's fixed rules; refusal keeps the portfolio). Engine, phases, data cut, CDI, plumbing check and gate exactly as
attempts 51-52: P[> Conservadora], P[> BOVA11] and DSR >= 0.95 in all 3 phases. Months without the signal panel:
no veto (the plain Conservadora).

- **Candidate:** pre-register a forward paper log against the same-day Conservadora.
- **Not a candidate:** the line is closed; no other quintile, signal or combination.

## 6. Expectation, written before

S1 mostly repeats the Conservadora's low-vol half, so its partial IC should be near 0 (gate b fails). S2 has the best
evidence, but before 2017 fewer than 20 issuers have it, so most of its months are 2017-2026 (about 115). S3 uses total
volume at a monthly horizon, where Pan and Poteshman's effect was weak even with open-buy volume. Expected: no signal
passes both gates; S2 the likeliest. Whatever the outcome: no other window, threshold, expiry rule or rerun.

## 7. Files (to be written after this commit)

Lake (worktree): `b3lake/option_forward.py` (pairs, forward, Black-76), migration `0013`, the rebuild of
`option_greeks`, the `option-signals` export; tests first. App: `scripts/opcoes_painel.py` (frozen panel in
`docs/dados/<build date>-opcoes/` with `SHA256SUMS`; prints coverage only), `core/opcoes.py` (partial IC, the veto;
tests first), `scripts/opcoes_ic.py` (stage A, read once: `docs/opcoes-ic-<date>.txt`). Stage B harness only after the
gate.
