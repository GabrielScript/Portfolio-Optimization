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

## Addendum, 2026-10-23 (later the same day): shown on the screen, at the user's request

The user asked to see the timing, news and sentiment on the political (election) and macro scenarios for every stock
in the app (DECISOES 124). The text above said "nothing reaches ... the app"; that changes for display only:

- The app shows the daily counts and Jev means (`lago/sentimento_diario.parquet`), the material CVM filings of its
  stocks (`lago/fatos_relevantes.parquet`) and, on the PC with the lake's disk, the headlines (title and link only).
  The deployed app gets numbers and public CVM filings, never the raw text (DECISOES 123).
- Nothing reaches the optimizer, the paper routines, the Fundo Alepha plan or the risk rules. The screen says the
  tone is untested.
- Tests A and B, their rules, data and dates are unchanged; N stays until one of them runs. Watching the forward log
  cannot leak into Test A (backfilled filings) and Test B's outcome (future returns) is unknown when the screen shows
  a day's tone.

## Amendment, 2026-11-17: Test A on the Composite, judged by a Liquid d1 model

Written before any backfilled filing is scored (no score of a backfilled filing exists on this date). The user asked
on 17/11 for a CVM-based probability per stock that tilts the Composite ("allocate more above 65-70%, trim below");
step 1 agreed with the user is to run Test A first. What changes, and only this:

- **Strategy.** The Composite (`core.composto.estrategia_por_nota`, attempt 51; the official strategy since the
  Conservadora left the main stack, DECISOES 261) replaces the Conservadora. Engine, phases 0/21/42, costs and the
  frozen export are those of `scripts/curvas_pos_idade.py` (export `docs/dados/2026-10-22/`, last session
  16/10/2026; the Composite must reproduce 21.79 / 20.34 / 19.22% before any veto is read, DECISOES 360).
- **Judge.** Jev is retired (DECISOES 269). The model is chosen before any backfilled score, on the 60 hand-labelled
  filings of DECISOES 270 (same sample, labels and materiality question), run again on 17/11 with d1.1, d1.2 and
  d1.2-ultra (d1.2 released 16/11/2026): highest accuracy@0.5 on all 60, ties broken by the lower Brier. The model id
  returned by the API is pinned and used for every filing; a changed id stops the run.
- **Questions and state.** `filings-v1` unchanged (`b3lake/jevscore.py`: filing_tone, filing_kind, filing_intensity),
  state = the lake's `filing_request` (category, subject, type, species; no company name, no date). Identical states
  are asked once and the answer is shared.
- **Filings.** `categoria` in (Fato Relevante, Comunicado ao Mercado), `data_entrega` from 2009-12-01 to 2026-10-16,
  of every issuer that is a Composite candidate in any month of the export (ticker to `codigo_cvm` by the lake's
  issuer table). A filing that maps to no ticker is dropped and counted.
- **Veto (the 23/10 rule, timing made explicit).** At each rebalance the decision date t is the training window's
  last session (the engine trades at t's close). A chosen name is dropped when it has a filing with P(bad) >= 0.5 and
  `filing_intensity` score >= 2.0 delivered in the 21 sessions before t (t itself excluded: a filing dated t may come
  after the close). Its score is removed for that decision, so `carteira_da_nota` takes the next name by score under
  the same sector cap and the same rules.
- **Pass (unchanged).** P[with veto > without] >= 0.95 in all 3 phases and a lower MDD in at least 2 of them. N in
  `docs/tentativas.csv` rises by one when it runs. A pass earns a forward ghost account only, never real money.
- **Descriptive, no gate (to calibrate the user's thresholds).** Excess return over the equal-weighted eligible
  universe for sessions 1-21 and 1-63 after each filing (entry at the close of the first session after
  `data_entrega`), by tone x intensity bucket and by P(bad) and P(good) bands (< 0.35, 0.35-0.65, 0.65-0.70, >= 0.70),
  with counts and t-statistics clustered by date, for 2010-2019, 2020-2025 and 2026 apart. It informs a later,
  separately pre-registered tilt arm; it changes nothing in Test A.
- **Known risk (unchanged, restated).** The judge may know these filings' outcomes; the state withholds names and
  dates, which only lowers the risk. The 2026 filings are reported apart, and a pass is confirmed only if the forward
  filings (`backfilled = false`) agree in sign at the 252-session reading.
