# Pre-registration: news/social log and CVM IPE filings as a veto (2026-10-23)

Written on the day the log started (lake `b3lake.sentiment`, task "b3lake sentimento", first run 23/10/2026 19:00),
before any score exists. DECISOES 123. Collection only: nothing reaches the optimizer, the paper or the app. N in
`docs/tentativas.csv` rises by one for each test below when, and only when, it runs.

## What is collected (lake design: `b3-datalake/docs/specs/2026-10-23-sentiment-log-design.md`)

- Items: 9 publisher RSS feeds every 2 hours; 5 Firecrawl news searches, r/investimentos, r/farialimabets and 2
  Threads searches at most twice a day. Each item keeps `fetched_at` (UTC), the point-in-time field.
- CVM IPE: every filing since 2003; `data_entrega` is the public date; rows found by the daily refresh carry
  `first_seen_at`, rows of the first load carry `backfilled = true`.
- Jev scores, model pinned (`jev-1.13.0` on TypeSafe, `typesafe/jev-1.13` on OpenRouter), questions fixed in
  `b3lake/jevscore.py`: items `items-v1` (relevant, market_tone, topic, company_tone, intensity, election) and
  filings `filings-v1` (filing_tone, filing_kind, filing_intensity). The state holds the text only (no URL, author or
  company name). A changed word is a new version; versions never mix in a test.
- Scoring waits for a valid key (none on 23/10/2026); a score more than 2 days after `seen_at` is marked late and
  reported separately, since a later model may know the outcome.

## Test A: material filings as a veto on the Conservadora (backtestable, 2010-2026)

- Score every backfilled material filing (`Fato Relevante`, `Comunicado ao Mercado`) of the Conservadora's candidate
  names with `filings-v1`. Mapping: `codigo_cvm` to issuer by the lake's issuer-CVM table.
- Veto: at each rebalance, drop a chosen name that has a filing with `filing_tone = bad` (probability >= 0.5) and
  `filing_intensity` >= 2 (Major) delivered in the 21 sessions before the decision date; the next name by score
  takes its place (same sector cap).
- Engine, phases, costs and IR: those of `scripts/avaliar_conservadora.py` on a frozen lake export.
- Pass: P[with veto > without] >= 0.95 in all 3 phases and a lower MDD in at least 2 of them. Otherwise the veto is
  not used.
- Known risk: Jev may have read about these filings' outcomes. The state withholds names and dates, which only
  lowers the risk. A pass is confirmed only if the forward filings (`backfilled = false`) agree in sign at the
  252-session reading.

## Test B: the forward news/social index (not before 252 sessions of log, about October 2027)

- Daily index: the mean of P(positive) - P(negative) of `market_tone` over the day's items with `relevant` >= 0.5,
  by `fetched_at` in Brasília time; items fetched after 17:00 count for the next session.
- Gate (the S2 gate, DECISOES 45): the Spearman IC between the index and the next 5 sessions' BOVA11 return >= 0.02
  with a HAC t >= 2 (5-session overlap). A second gate on volatility: the share of items with `intensity` >= 2
  against the next 5 sessions' realized volatility, same thresholds.
- Passing a gate earns only the right to pre-register a veto or meta-label (never an expected return) as a new test.

## What would make the log worthless (and is checked at each reading)

Gaps of more than a day without items, a source answering errors for more than a week, or a scored share below 90% of
items. The watchdog (`scripts/vigia.ps1`) flags a log with no successful run in 6 hours.
