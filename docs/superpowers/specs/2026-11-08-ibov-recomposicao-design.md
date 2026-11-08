# Ibovespa rebalance flows: buy the entrants before the passive funds must (pre-registration)

_2026-11-08. User (08/11): of the five ideas for a simulated market (MARL, MAS, mean field games, causal inference),
start with the most concrete one, the forced buying of the Ibovespa's new names by index funds, "pode prosseguir".
Written and committed before any return around a rebalance was computed. What was read before writing: the lake's
CVM index-fund holdings (which funds track the index, months covered, the entrants and leavers they imply, below) and
B3's pages on the preview calendar. No price, return or volume around any event was looked at. The rules are the
author's calls (autonomous mode, CLAUDE.md), kept for the user to review._

## The idea and why it could pay

- The Ibovespa is rebuilt every four months (portfolios in force from the first Monday of January, May and
  September). B3 publishes three previews in the old portfolio's last month: the 1st on its first session, the 2nd
  on the session after the 15th, the 3rd on its penultimate session. Funds and ETFs that copy the index (BOVA11 and
  others) must hold the new names by the close of the last session before the new portfolio starts, whatever the
  price. That demand is forced, dated and public in advance.
- Causal framing: entry is decided by B3's published rule (negotiability, presence, no penny stocks), not by the
  name's next returns, so the rebalance is a natural experiment on forced demand. A small buyer can stand in front of
  it; large funds cannot, so capacity is not our problem.
- Long only (no short selling in this app): the test buys entrants; leavers are reported, never traded.

## Data (built by code, frozen before the reading)

1. **Final compositions, by month**, from the lake's CVM fund holdings (`fund_stock_holdings`): BOVA11
   (10.406.511/0001-61, 2008-11 to 2018-08) and the fully public index fund 41.901.871/0001-63 (2021-07 to 2026-09);
   the 2018-09 to 2021-06 gap from the other index funds whose holdings match the index (a name counts only if two
   funds hold it). Cleaning, fixed now: renames and mergers mapped with the lake's `share_events` / `securities`
   (CCRO3 -> MOTV3, BRFS3 + MRFG3 -> MBRF3, ...), subscription rights and receipts dropped (tickers ending in 1, 2,
   9, 12-15), and a name is an entrant only if it is absent from the old portfolio and present in both of the two
   month-ends after the rebalance (filters partial, confidential months).
2. **Previews, 2020 onwards**, from B3's dated news pages ("1ª/2ª/3ª prévia da carteira do Ibovespa B3") and Bora
   Investir, fetched with Firecrawl: date, names entering, names leaving. Checked against item 1 (a preview entrant
   that never enters is kept: a real trader would have bought it).
3. **Validation before the reading:** the 2024-05 (VIVA3 in), 2025-05 (DIRR3 and SMFT3 in, AMOB3 out) and every
   other 2020+ rebalance of item 1 against item 2; the share that agrees is reported. Prices, Selic, costs: the app's
   usual total-return closes, `core/execucao` costs (B3 fees plus half spread and square-root impact), closing-call
   execution.

## Rules (two, both counted: N 102 -> 104)

- **R1, the 3rd preview (2009-2026).** Buy each entrant at the close of the 3rd preview's session, equal weights;
  sell at the close of the last session before the new portfolio starts (the session whose closing call the index
  funds trade in). Before 2020 the 3rd preview list is taken as the final list (the previews are not archived): a
  small look-ahead, reported with the 2020+ share of 3rd previews that equal the final list.
- **R2, the 1st preview (2020-2026).** Buy each 1st-preview entrant at the close of the 1st preview's session; sell
  as in R1. About one month held.
- Money not in an event earns the Selic. Costs on both sides. IR 15% reported separately (the R$ 20 thousand
  monthly exemption makes it zero at our size).

## Measures and gate (fixed now)

- Per event: return over the holding window minus BOVA11's over the same window (abnormal return), net of costs.
- **Gate, per rule:** mean net abnormal return > 0 with t >= 2.24 (two rules, Bonferroni at 5%), clustered by
  rebalance date (events on the same date are not independent). R1 also needs a positive mean in at least two of
  its three blocks (2009-2014, 2015-2020, 2021-2026). Reported beside it: the median, the share of winning events,
  the worst event, and the result without the look-ahead years (R1 on 2020+ only).
- **If a rule passes:** a fictitious account "Recomposição" (R$ 10,000 in the Selic, entering only on events) beside
  the others, read once after six rebalances (2028-09); nothing goes to real money from the backtest alone.
- **If both fail:** closed; other windows or thresholds would be new counted trials.

## Expectation, written before

- Small or none. In the US the S&P 500 inclusion effect has shrunk to near zero as more traders anticipate it;
  B3's passive share is smaller and growing. Entrants are often names that have already risen (overlap with the
  Composite's momentum), so part of any effect is not new. Likely result: R1 fails; R2 has too few events
  (about 35) to pass t >= 2.24 unless the effect is large.

## Build order

1. The compositions table with its cleaning and tests (renames, rights, two-month rule), then the event list.
2. The previews scraper for 2020+ (Firecrawl), frozen to `docs/dados/ibov-recomposicao/`.
3. The validation report (no returns).
4. The event study, read once: `docs/ibov-recomposicao-<date>.txt`, and `docs/tentativas.csv` +2.
