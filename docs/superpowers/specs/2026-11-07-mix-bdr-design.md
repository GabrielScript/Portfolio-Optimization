# Fixed BR/BDR mixes (15 names, 1 to 5 US BDRs), B3 — design and pre-registration

_2026-11-07. Written and committed before any number of these arms exists. User's questions (2026-11-07): why the
portfolio holds no US BDRs, and whether 14/1, 13/2, 12/3, 11/4 or 10/5 stocks/BDRs would have done worse than the
Composite. DECISOES 141 let an optimizer choose the US share freely (helped Min. Vol, hurt Kelly); fixed counts were
never tested. Branch `feat/mix-bdr` from `feat/periferia` (`6fb23be`). Sections 1 to 5 are fixed from the commit on._

## 1. What the data allow

US BDRs opened to retail investors in 10/2020 (CVM); before that almost none passes the app's liquidity floor
(DECISOES 141), and the Composite engine needs 1,260 NaN-free sessions of a name, so a BDR inside the Composite engine
would only enter around 2025. Hence a two-sleeve design read on **2021-01-04 to 2026-10-16**, the one period with
tradable BDRs: about 5.8 years and one US tech bull market, so the test has little power and says so.

**Prior, written now:** none of the five arms passes the gate; returns rise with k in 2021-2026 because the US leg
had a strong run (DECISOES 141: Min. Vol gained from it), but not by enough for P[> Composite] >= 0.95 in 5.8 years.

## 2. The arms `BDR-k`, k = 1 to 5

- **BR sleeve:** the Composite (attempt 51's score and rule, `estrategia_por_nota`) with **15 - k names**, 4 per
  sector, equal weights, its own engine run (`walk_forward_dinamico`, `CFG_BT`, frozen `docs/dados/2026-10-22/`).
- **US sleeve:** at each of the BR sleeve's rebalance dates, the **k names with the highest 12-1 momentum** (total
  return from 252 to 21 sessions before the decision) among the US set of DECISOES 141 (30 US-company BDRs, IVVB11,
  NASD11) that are eligible that month: traded in >= 90% of the previous 12 months' sessions with a median daily
  volume >= R$ 1 mi, and with the 252 sessions of prices. Fewer than k eligible: the empty slots go to the BR sleeve.
  Equal weights, held to the next rebalance. Prices: the lake's gold `tr_close` (splits adjusted, **no BDR
  dividends**: the US leg is understated by about its dividend yield, at most ~0.3 p.p. a year of the whole at k = 5),
  read live, its slice hashed; AVGO34 2021-09-17 dropped (bad print, DECISOES 141).
- **Mix:** at each rebalance the weights reset to (15 - k)/15 BR and k/15 US; both sleeves drift in between; 10 bps a
  side on the US sleeve's traded share and on the reset between sleeves.
- **Tax:** the reading is pre-tax, like every engine reading. A BDR gain pays 15% with no R$ 20 mil exemption, so the
  tax can only lower an arm: an arm that fails pre-tax fails net; one that passed would need a net-of-tax reading first.

## 3. Gate

Phases 0/21/42 of the BR sleeve's engine, each read from its first rebalance on or after 2021-01-04 to 16/10/2026,
against the Composite (15 names) and the Conservadora on the same dates (the Composite and the Conservadora must first
reproduce 21.4/19.9/19.1% and 17.41/16.41/16.90% over the full period). **Gate, every phase, all >= 0.95:** P[> Composite],
P[> Conservadora], P[> BOVA11], DSR. Reported, not gates: CAGR, MDD, volatility, the year table, IVVB11 over the same
window, the US sleeve's latest picks.

**N.** Five configurations, five rows in this branch's `docs/tentativas.csv` in this commit. The DSR uses the union of
every branch's trials, **N = 100** (95 + 5).

## 4. Output

`docs/mix-bdr-2026-11-07.txt` and a DECISOES entry. Nothing goes to the app.

## Addendum (2026-11-07, before any arm's number; the first run stopped at BR-13's first rebalance)

The fixed limits are the whole portfolio's (30% per sector, 20% per name and issuer). The BR sleeve holds (15 - k)/15
of the portfolio, so 4 names of one sector inside a 13-name sleeve are 4/13 = 30.8% of the sleeve but 4/15 = 26.7% of
the whole, and the engine refused it. The sleeve is therefore checked with the whole portfolio's limits scaled to
it: sector, name and issuer caps times 15 / (15 - k); the volatility cap is unchanged. The US sleeve has no sector
rule (as pre-registered): at k = 5 it can hold up to 5/15 = 33% in one US sector, above the 30% the app allows; this
is reported with the picks, not corrected.
