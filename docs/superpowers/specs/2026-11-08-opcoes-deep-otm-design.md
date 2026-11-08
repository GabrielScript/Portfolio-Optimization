# A second fictitious options account: deep out of the money (pre-registration)

_2026-11-08. User (08/11): "Pode seguir com a conta deep OTM" (buying deep out-of-the-money options, dry, only as a
simulation). Written and committed before the account ran once and before any option return was read. What was
looked at: the 06/11 options account dry run (it failed: an ask of zero, fixed in 56af121) and the liquidity of the
lake's recent chain by moneyness (2026-09-24 to 2026-11-06, below). No price after a cycle date exists in that chain,
so nothing about outcomes was seen. The rules are the author's calls (autonomous mode, CLAUDE.md), kept for the user
to review._

## What it is

- A new fictitious account, "Opções deep OTM", R$ 10,000, beside the options account of DECISOES 259 ("Opções") and
  the Selic. It never touches real money and does not change the Composite, the real buy of 19/12 or its tiers
  (DECISOES 284 read only the "Composto + Sentinela" account).
- **Same everything as "Opções" except the strike distance**: same cycle (first B3 session of each month, at its
  close), same budget (10% of the account a cycle, half per idea), same names (idea A: the stretched stock near its
  252-session high gets a put; idea B: the volatile 12-1 winner gets a call), same expiry rule (nearest 21 business
  days within [10, 45]), same costs (0.03% of the premium, the ask pays the spread), IR 15% on each expiry session's
  net result above the losses carried, held to expiry, cash at the daily Selic. The difference between the two curves
  is the strike distance alone.
- **Strikes:** the put at about 0.85 x the stock's close, the call at about 1.15 x, inside the band 0.80-0.90 (put)
  and 1.10-1.20 (call). An option outside the band is never bought, whatever is nearest.

## The contract and the size (shared with "Opções", amendment of 08/11 to its rule 6)

- Candidates: the idea's stock, the option type, traded that day (trades > 0), a real two-sided closing quote
  (bid > 0 and bid <= ask <= 2 x bid), strike inside the account's band (target +- 0.05 x the close here; +- 0.03 x
  for "Opções", that is 0.92-0.98 and 1.02-1.08), business days left in [10, 45].
- Expiry: among the candidates, the one nearest 21 business days (ties: the earlier). Strike: on that expiry, the
  candidate with the **largest traded volume that day** (`volume_brl`); ties: nearest the target, then the lower
  strike. Why not the nearest strike: on 06/11 the near strikes of VAMO3's calls had one-sided quotes and "nearest
  1.05" bought a strike 22% away; the most traded strike in a band is the price a buyer could actually get.
- **Size:** the largest lot of 100 that fits the idea's half of the budget **and** the option's traded volume that
  day (the account never buys more than the whole market traded in that option that session). If not even one lot
  fits, no order and the reason is recorded.

## Liquidity, measured before (the only data read)

Underlyings with at least R$ 1 million a day of options over 21 sessions, 10-45 business days left, 2026-09-24 to
2026-11-06 (about 77 names a day):

| Band | Underlying-days with a traded option in the band | ... with a real two-sided quote | Median day volume of the most traded | Most traded >= R$ 500 |
|---|---|---|---|---|
| put 0.92-0.98 | 2,170 | 44% | R$ 37,184 | 96% |
| call 1.02-1.08 | 2,187 | 53% | R$ 79,139 | 98% |
| put 0.80-0.90 | 2,128 | 40% | R$ 7,507 | 84% |
| call 1.10-1.20 | 2,101 | 47% | R$ 29,377 | 92% |

So a deep account at about R$ 500 per idea can fill most months on the chosen name, at the closing ask.

## Files, screen, schedule

- `simulador/opcoes_deep.csv` (ledger, the columns of `opcoes.csv`) and `simulador/opcoes_deep_marcas.csv` (daily
  unit values), append-only, written and committed by the nightly upkeep right after the options account.
- First cycle: the first session close after the code is live, **09/11/2026**; then the first session of each month.
- Screen: its own section under "Conta de opções", same layout (value, Selic, returns, drawdown, open options,
  orders), and the warning that deep out-of-the-money buying loses on average.

## Expectation and the one reading (fixed now)

- Expected: below the Selic and below "Opções". Deep out-of-the-money options expire worthless most months, so the
  account bleeds about 10% in a month when both expire, and its result rests on a few large wins. That is the point
  of measuring it: how much it bleeds and how large and rare the wins are on B3's real quotes.
- **When:** after the 12th cycle's options expire (cycles 09/11/2026 to 01/10/2027; about the end of October 2027).
- **What:** the account against the Selic and against "Opções" (return, max drawdown), the share of cycles with no
  order and why, the number of cycles whose option ended in the money and the largest single gain.
- **Decision rule:** nothing goes to real money from this reading alone; keeping, changing (a new pre-registration)
  or dropping it is the user's call on that reading. Twelve cycles of a lottery-like payoff are weak evidence, and the
  reading says so. N of the research count does not change (a forward account, no historical selection).
- Advisor review of the order logic (CLAUDE.md): the advisor was unavailable on 08/11 (the user turned it off after a
  usage error); to be asked when it is back, before the reading.
