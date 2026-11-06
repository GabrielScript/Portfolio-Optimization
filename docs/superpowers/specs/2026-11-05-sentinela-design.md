# Sentinela: a DeepSeek watch agent over the Composite (pre-registration)

_2026-11-05. User's direction of 05/11 (DECISOES 261, 262): a DeepSeek agent, not a strategy, watching the Composite's
names during B3 sessions with every official source the app already has plus IBGE SIDRA, Tesouro Nacional and the
B3 lending rates; low effort to scan, max effort only when something seriously compromises a stock; it may act by
itself at confidence >= 70%, **only in a fictitious R$ 10,000 account**; no web search or fetch (API cost); near-zero
hallucination is the top requirement. Written and committed before the agent ran once and before any alert or
return was read. The rules below are the author's calls (autonomous mode, CLAUDE.md), kept for the user to review._

## What it is and what it is not

- It **watches**: it reads what is new about each Composite name and says whether anything seriously compromises it.
- It **acts only in fiction**: a new fictitious account "Composite + Sentinela" (R$ 10,000) that copies the Composite
  and lets the agent cut or sell a name; the plain Composite runs beside it. The difference between the two curves is
  what the agent adds or costs (an A/B).
- It **never touches real money**. The real R$ 500 of 19/12/2026 follows the Composite plan; the agent's alerts go to
  the user by email and the user decides.
- It does **not** forecast returns and is not a strategy. It is a guard against known bad news being missed.
- It has **no web tools**: no DeepSeek Search/Fetch, no Firecrawl, no social posts. Everything it reads comes from the
  app's own official sources, fetched by code.

## Sources (items)

Each source is read by code (no model) and turned into **items**: `id` (source + stable hash), `fonte` (class),
`data`, `ticker` (or `macro`), `texto` (verbatim, at most 600 characters, numbers as published).

| Class | Source | Changes | Item |
|---|---|---|---|
| `cvm` | CVM IPE open data (`fatos_relevantes`) | weekly (Sundays) | each new filing of a Composite name |
| `cvm` | CVM daily filings (RAD/ENET), **next priority after this build** | daily | same, without the week's delay |
| `mercado` | lake daily prices (`equity_daily`) | nightly | close move beyond 2.5 sigma of the name's 63-session daily moves, volume beyond 3x its 21-session median |
| `mercado` | B3 BDI lending (`bdi_BTBLendingOpenPosition`, `bdi_BTBLoanBalance`) | nightly | share lent out or rate jump beyond 3x its 21-session median; **only after the rate unit is checked** |
| `bcb` | Selic, Copom statement/minutes, Focus, PTAX, IBC-Br, credit (`core/bcb`, `core/macro_bcb`) | daily to monthly | a new value or a new document |
| `ibge` | IBGE SIDRA: IPCA, GDP, unemployment (PNAD), industry (PIM) | monthly/quarterly | a new release; endpoints confirmed with Firecrawl before the build |
| `tesouro` | Tesouro Nacional: primary result, gross debt | monthly | a new release; endpoints confirmed with Firecrawl before the build |
| `anbima` | ANBIMA indices already read by the app (never weights, DECISOES 260 answer) | daily | confirmation only, never an item by itself |

Macro items (`bcb`, `ibge`, `tesouro`) are shared by every name, but only on the run where they are new.

## Schedule (changed 05/11 after the user's 2h question)

- **Polling (code only, free):** every hour from 08h to 19h on B3 sessions (B3's own calendar), each source is
  asked for new items. Seen item ids are kept in `simulador/sentinela_vistos.csv`.
- **Model call only when a name has new items.** No new item, no call: the number of calls follows the number of new
  facts, not the number of runs. Checking every 1-2 h therefore costs no API and the user's 4h cadence becomes a
  ceiling on latency, not a cost driver.
- **Why not a fixed model call every 2h:** the sources do not change intraday (CVM open data weekly, prices and BDI
  nightly, IBGE and Tesouro monthly); fixed calls would mostly see nothing new, add emails (alert fatigue) and more
  chances to act (overtrading). The real speed gain is the daily CVM source above.
- **Budget caps (hard, in code):** at most 40 low-effort calls and 6 max-effort calls per day; over the cap the items
  wait for the next session and the email says so.

## Stage 1: scan (DeepSeek-V4.1-Flash-1017, low effort)

- **One stock per call.** The pack: the ticker, its sector, its weight in the Composite, at most 8 new items (newest
  first; the rest wait for the next call), and at most 5 lines of context (last close, 21-session return, the name's
  last Composite record). Target under 3,000 input tokens. Fixed system prompt first so DeepSeek's prefix cache hits
  (`deepseek-cache-economy`). The persona prompt sets the tone only; quality comes from the data and the rules.
- **Fixed JSON answer:** `{"ticker", "veredito": "sem_evento" | "atencao" | "grave", "citacoes": [{"id", "trecho"}],
  "resumo"}`. `sem_evento` is always allowed and is the expected answer most of the time.
- **Checked in code** (reusing `core/assessor`'s normalization and verbatim check): every cited `id` must be in the
  pack, every `trecho` must appear verbatim in that item's text, every number in `resumo` must appear in a cited
  `trecho`. Any failure: the answer is discarded, logged as `descartado` with the reason, and treated as `sem_evento`.
  `grave` without at least one `cvm`, `mercado`, `bcb`, `ibge` or `tesouro` citation is downgraded to `atencao`.
- `atencao` is logged and goes into the day's digest email; it never moves money.

## Jev in shadow mode (stage 1)

- Each item of a model call is also sent to Jev (TypeSafe, RLCD) as two yes/no judgments with calibrated
  probabilities: "Does this item materially affect <company>'s value?" and "Is it new information, not a repeat?".
- **Shadow only:** Jev's answers are logged (`simulador/sentinela_jev.csv`) and change nothing. If `TYPESAFE_API_KEY`
  is missing or Jev fails, the agent runs without it.
- At the reading (below) Jev is compared with DeepSeek's stage 1 and with what the stocks did. It may become the router
  (deciding which names go to stage 2, saving max-effort calls) only if it cuts calls without missing a `grave` that
  stage 2 confirmed. It never decides on money. Its calibration is about the judgment, not a return forecast.

## Stage 2: decision (DeepSeek-V4.1-Flash-1017, max effort)

- Only for a `grave` that passed the checks. Same pack plus the stage 1 citations and at most 10 of the name's last
  filings (`core/assessor.montar_pacote` limits).
- **Fixed JSON answer:** `{"ticker", "acao": "manter" | "reduzir_metade" | "vender", "confianca": 0..1,
  "citacoes": [...], "motivo"}`, checked as in stage 1. A failed check means `manter`.
- **Acts only if** `acao` is not `manter`, `confianca >= 0.70` and the checks passed. The 0.70 threshold is fixed
  here and is not tuned before the reading.
- DeepSeek's stated confidence is not calibrated (models tend to state high numbers); every confidence is logged with
  what the stock did after 5 and 20 sessions so its calibration can be measured at the reading.

## The fictitious account "Composite + Sentinela"

- Starts with R$ 10,000 on the first session the agent is live, holding the Composite's current weights at that day's
  close (same costs as the simulator).
- An agent sale or cut executes at the **next session's close** after the decision (no intraday price in the lake);
  the cash earns the daily Selic (SGS 11), as the simulator's cash does.
- No buying back by the agent. At the Composite's next record (every 63 sessions, `core/paper.PASSO`) the account
  returns to the Composite's new weights.
- The plain Composite account starts the same day with the same R$ 10,000 for a clean comparison.
- Screen: one section with both curves, the Selic, the open alerts and the agent's orders.
- **Advisor review before the build** (CLAUDE.md: order logic).

## Email and log

- Every `grave` and every action: an email at once (`core/alerta_email`), with the ticker, verdict, confidence, the
  verbatim citations and their sources. `atencao`: one digest email at 19h if any.
- Log `simulador/sentinela_log.csv`: time, ticker, stage, verdict, action, confidence, item ids, checks, tokens, cost,
  the day's close; filled later with the 5- and 20-session returns.

## Reading (once, fixed now)

- **When:** the first B3 session on or after 2027-01-08 (about 6 weeks live), or 8 weeks if fewer than 5 `grave`
  alerts by then.
- **What is read, in this order:** (1) the share of answers discarded by the checks (target under 5%; above 10% the
  agent is stopped and the prompt and pack are reviewed); (2) the 20-session return of names flagged `grave` against
  the other Composite names; (3) "Composite + Sentinela" minus plain Composite, return and max drawdown;
  (4) calibration of DeepSeek's confidence and of Jev's probabilities (reliability by decile, Brier score).
- **Decision rule:** the agent's alerts keep going to the user whatever the result; its actions are proposed for real
  money only if (2) and (3) are both on its side. With so few events this reading is weak evidence and is reported as
  such; N of the research count does not change (it is a guard, not a strategy).

## Build order

1. Confirm the IBGE SIDRA and Tesouro Nacional endpoints with Firecrawl; check the BDI lending rate unit.
2. `core/sentinela/` (items per source, seen store, pack, checks, stages, account) with tests first, then the hourly
   task, then the screen. Target: live by 2026-11-30 so it runs 2-3 weeks before the real money of 19/12.
3. Next: daily CVM filings (RAD/ENET).

## Amendments (2026-11-05 night, before any build or run; DECISOES 264-266)

1. **Fiscal items come from BCB SGS**, class `bcb`: 4649 (primary result, consolidated public sector, monthly R$ mi),
   5793 (primary result 12m % GDP), 4513 (net debt % GDP), 13762 (general government gross debt % GDP). Tesouro
   Nacional's own API (apiapex) was down on 05/11; its RTN XLSX (CKAN `resultado-do-tesouro-nacional`) is optional,
   class `tesouro`.
2. **IBGE items** via `https://apisidra.ibge.gov.br/values/...`: GDP 5932 v6564 c11255/90707 (q/q), unemployment
   **6381** (moving quarter, monthly; 4099 is quarterly only), industry 8888 v11602 c544/129314 (y/y). **IPCA comes
   from BCB only** (already in the app); SIDRA's IPCA is not used, so one issuer per indicator and no fact twice.
3. **BDI lending rates**: B3 publishes no unit; read as a decimal fraction a year (inferred). Items use only relative
   jumps (x3 the name's 21-session median), which do not depend on the unit, so they are no longer gated on the unit.
4. **Checks**: the Sentinela has its own official class set `{cvm, mercado, bcb, ibge, tesouro}` passed as a
   parameter; it reuses `core/assessor`'s normalization and verbatim matching only. `core/assessor`'s
   `CLASSES_OFICIAIS` and `_url_oficial` are not changed (the adviser's behaviour stays as it is).
5. **Execution timing**: a decision taken on session T before 17h executes at **T's close** (no lookahead: the
   decision precedes the close); a decision after 17h, or on a non-session, executes at the next session's close.
6. **One action per name per day.** After `reduzir_metade` only `vender` is allowed for that name until the
   Composite's next record. A new `grave` on a name already sold is logged and emailed, with no action.
7. The order-logic review asked by CLAUDE.md was done by the author; the user asked on 05/11 to go on without the
   advisor.

## Amendment 2 (2026-11-05 night, before any build; DECISOES 269)

8. **The shadow judge is Liquid AI's d1-ultra, not Jev.** Same request shape (`state` + typed `noul`/`choice`/`score`
   questions, `typesafe-sdk` with `base_url="https://api.liquid.ai"`, key `LIQUID_API_KEY`), US$ 0.042 per million input
   tokens, output free. The vendor reports it beats Jev 1.14 on 112 evaluations with better calibration; that is the
   vendor's own benchmark, so before d1-ultra may route anything, it is checked on our own items: 60 Portuguese CVM
   filings of the Composite's names, labelled by hand (material / not material), Brier and accuracy against Jev 1.14
   on the same items. Still shadow only, still never decides on money.

## Amendment 3 (2026-11-05 night, after the judges check of amendment 2; DECISOES 270)

9. **The shadow judge is d1.1 (Liquid API direct), not d1-ultra.** On the 60 filings of amendment 2, by its own
   criteria (accuracy and Brier): d1.1 93% / 0.082, d1-ultra 82% / 0.121, Jev 1.14 80% / 0.145 (`docs/juizes-2026-11-05.md`).
   d1-ultra caught one more true event (17 of 18 vs 15) but raised 10 false alarms against 1. During the shadow period
   the log also keeps d1-ultra's answer for the same items, so recall can be re-read on real events before any routing.

## Amendment 4 (2026-11-05 night, before any build; DECISOES 271) — proper scoring rules decide the agent's autonomy

10. Every stage-2 confidence and every judge probability is scored with proper scoring rules (Brier and log score)
    against what the stock did after 20 sessions (event = a fall beyond 1.5 sigma of the name's 63-session moves for
    `vender`/`reduzir_metade`). At the reading: if DeepSeek's decisions stated at >= 0.70 were right less than 55% of
    the time, or its Brier is worse than the constant base rate's, the agent loses its autonomy (alerts only, no
    fictitious orders) until a new pre-registered spec. Fixed now, not tuned later.
11. Real orders (19/12 onward) go to B3's closing call auction ("ordem no fechamento"), so the real fill matches the
    close price the backtests and the fictitious accounts use.

## Amendment 5 (2026-11-05 night, before any build; DECISOES 272) — which sources reach which model

12. **DeepSeek (stage 1 and 2)** reads the per-stock pack: company items (CVM filings; price/volume anomalies from the
    lake; BDI lending jumps) plus macro items only on the run where they are new (BCB: Selic, Copom, Focus, PTAX,
    IBC-Br, credit, IPCA, fiscal SGS 4649/5793/4513/13762; IBGE SIDRA: GDP 5932, unemployment 6381, industry 8888;
    Tesouro RTN optional). ANBIMA only confirms, never an item.
13. **The judges (d1.1, with d1-ultra logged)** read one company item per call (its verbatim text, the company's name
    and sector), never macro items: macro news is not about one company, and asking it once per name would multiply
    calls and answers that mean the same thing. Two noul questions per item, as in the judges check.
