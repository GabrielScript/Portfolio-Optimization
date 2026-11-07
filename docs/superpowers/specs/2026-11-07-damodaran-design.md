# Damodaran: an intrinsic-value agent over the Composite and the 12/3 (pre-registration)

_2026-11-07. User's direction of 07/11 (DECISOES 306): add Aswath Damodaran's way of valuing companies as one more
layer of reasoning, working with DeepSeek and Liquid AI d1.1 / d1-ultra on the data the app already has, acting at
confidence >= 70% on hold / buy / sell of the Composite's and the 12/3's stocks, BDRs and ETFs; feed it with his
YouTube content since 2023 (ScrapeCreators) and the site data already in `D:\data_damodaran`. Agreed with the user the
same day: a **fictitious account first**, not real money; a curated core of his content, not 500 videos in a prompt.
Written and committed before the agent ran once and before any valuation or return was read. The rules below are
the author's calls (autonomous mode, CLAUDE.md), kept for the user to review._

## What it is and what it is not

- It **values**: for each name the Composite or the 12/3 holds, it writes a short story, turns it into a few numbers
  (growth, margin, reinvestment, risk), lets code compute the value per share, and compares it with the price.
- It **acts only in fiction**: two new fictitious accounts, "Composto + Damodaran" and "12/3 + Damodaran" (R$ 10,000
  each), copy their base portfolios and let the agent sell, halve, keep or add to a name. The plain "Composto" and
  "12/3" accounts already run beside them: the difference between the curves is what the agent adds or costs.
- It **never touches real money**. The real buy of 19/12/2026 follows the Composite plan (DECISOES 284); nothing here
  changes it. Real use only after the reading below, and only if the user decides.
- It does **not** pick new names. The universe and the picks stay the Composite's and the 12/3's rules (DECISOES 299;
  the user's rule: the solver never sets the universe). It reweights or drops inside them.
- It does **not** trade news. Damodaran revalues on new numbers, not on headlines; the Sentinela keeps the news.
- **No web tools**, as the Sentinela: everything it reads is fetched by code from the app's own sources.
- It is **not backtestable**: the models were trained on what happened, and his videos since 2023 comment on it.
  Only a forward record counts. That is why no historical reading is run for it.

## Why a curated core and not 500 videos (the content judgment, 07/11)

The full channel since 2023 was fetched into `D:\data_damodaran\youtube` (427 videos, 426 with captions, about 4.2
million words, ScrapeCreators, 448 credits). Judgment (`docs/damodaran-acervo-2026-11-07.md`):

- About 80% is his yearly NYU courses (Valuation MBA, Valuation undergraduate, Corporate Finance, Investment
  Philosophies) repeated each spring; the 2025 edition carries the method.
- Brazil appears as worked examples of country risk (Embraer and Ambev, numbers from 2004) and of local-currency
  risk-free rates; there are no current views on our names. The value is the **method**, not opinions.
- DeepSeek already knows his framework from training. What it lacks is **discipline and current inputs**: the
  rules applied the same way every time, and this year's country risk premium, equity risk premium and industry betas.
- 888 thousand words in the curated core (108 videos: the country risk editions, the Data Updates 2023-2026, the 2025
  course's sessions on risk-free rates, risk premiums, cost of capital, cash flows, growth, terminal value, financial
  service firms, cyclical and commodity firms, declining firms, the life cycle, narrative and pricing; list in
  `D:\data_damodaran\youtube\selecao.json`) is still about 1.2 million tokens: no prompt holds it.

So the agent reads a **method card** (below) distilled from the core, plus his **current data** (below).

## The method card (`core/damodaran/ficha.md`)

- At most 40 rules, at most 4,000 tokens in all, written in English by the author from the curated core.
- **Each rule cites one verbatim passage** (at most 300 characters) and its video id. A test checks that every passage
  appears in that video's transcript (whitespace normalized), the same verbatim discipline as the Sentinela.
- Fixed at the first run and versioned; changing it is a spec amendment (the prompt prefix stays cached,
  `deepseek-cache-economy`).
- Covers: risk-free rate in BRL and in USD; equity risk premium and country risk premium (and a company's exposure to
  country risk by where its revenue comes from); bottom-up betas by industry; cost of capital; FCFF for
  non-financial firms; equity (excess return) valuation for banks and insurers; cyclical and commodity firms on
  normalized numbers; declining and distressed firms; the life cycle; stories bounded by the numbers; terminal growth
  never above the risk-free rate; pricing (multiples) only as a cross-check.

## Inputs (per name, built by code)

| Input | Source | Note |
|---|---|---|
| Statements, last 5 years and last 4 quarters | lake CVM DFP/ITR (`core/valor` reads) for B3 names; lake EDGAR `edgar_facts_first_filed` for BDRs | only what was filed before the run date |
| Price, shares, market cap | lake `equity_daily` / `market_caps` | last close before the run |
| Sector and the name's weight | the base account's record | |
| Brazil country risk premium, default spread, rating | `D:\data_damodaran\...\ctrypremJuly26.xlsx` and later editions, the newest dated before the run | Brazil July 2026: Ba1, spread 1.99%, CRP 3.10%, ERP 7.30% |
| Mature-market (US) implied ERP | `histimpl.xls` / his monthly ERP file, newest before the run | |
| Industry betas, unlevered | `betaemerg.xls` (B3 names), `betas.xls` (US names), newest January edition | a fixed map from our sectors to his industries, written before the first run |
| BRL risk-free rate | the Tesouro prefixed bond nearest 10 years minus Brazil's default spread (his method); fallback: US 10y x (1 + expected IPCA, Focus) / (1 + US expected inflation) | **source confirmed before the build** |
| USD risk-free rate | US 10-year Treasury (lake FRED) | |

## Stage 1: story to numbers (DeepSeek-V4.1-Flash-1017, max effort)

- **One name per call**, fixed system prompt first (method card, persona line, the investor profile line of
  DECISOES 287), then the name's pack. Target under 8,000 input tokens.
- **Fixed JSON answer:** `{"ticker", "historia" (at most 600 characters), "modelo": "fcff" | "fcfe_bancos",
  "crescimento_receita_5a", "margem_operacional_alvo", "vendas_sobre_capital", "beta_desalavancado",
  "exposicao_risco_pais", "citacoes": [{"id", "trecho"}]}`. Every parameter has a fixed range checked in code
  (for example growth -20% to +40% a year, target margin within the name's own 10-year range plus or minus 10 points);
  out of range, missing or a number not traceable to a cited passage: discarded and asked again once with the reason
  (as the Sentinela, DECISOES 304), then the name is "manter".
- **Code computes the value** (never the model): a fixed ten-year FCFF model (his "Ginzu" shape) for non-financial
  firms and an excess-return equity model for banks and insurers, terminal growth = min(risk-free, 4% BRL / 2.5% USD),
  cost of capital from the inputs above. Output: value per share and the gap to the price.

## Stage 2: the verdict (DeepSeek-V4.1-Flash-1017, max effort)

- Same pack plus the stage-1 numbers and the computed value. **Fixed JSON:** `{"ticker", "acao": "vender" |
  "reduzir_metade" | "manter" | "aumentar", "confianca": 0..1, "motivo", "citacoes"}`, checked as in stage 1.
- **Acts only if** the action is not "manter", `confianca >= 0.70`, the checks passed **and** both judges below
  agree. The 0.70 threshold is fixed here and not tuned before the reading.

## The judges (Liquid AI d1.1 and d1-ultra)

- Each stage-2 answer goes to d1.1 and d1-ultra as two yes/no judgments with probabilities: "Do the story's numbers
  follow from the cited data?" and "Does the action follow from the value against the price?" (d1.1 won the
  60-filing check, DECISOES 270, 302).
- **Gate:** an action needs d1.1's two answers "yes" at p >= 0.5. d1-ultra is logged in shadow only (its own reading
  at the same date). If a judge fails or is not reachable, the name is "manter".

## The fictitious accounts

- "Composto + Damodaran" starts with R$ 10,000 on the first run's next session holding the Composite account's
  weights; "12/3 + Damodaran" the 12/3 account's. Same costs and the BDR tax of the simulator (DECISOES 299).
- **Actions:** `vender` sells the name; `reduzir_metade` halves it; `aumentar` raises it to 1.5 times its weight,
  never above the caps (20% a name, 30% a sector, `core/risco.LIMITES`). Money from sales goes pro rata to the names
  kept, within the caps; what the caps block stays in cash at the daily Selic. An `aumentar` is funded pro rata from
  the names kept. Orders execute at the **next session's close** after the run (no intraday price in the lake).
- At each base record (the Composite every 63 sessions; the 12/3 follows it) the accounts return to the base weights,
  and the next run applies its actions again.
- **Advisor review before the build** (CLAUDE.md: order logic).

## Schedule and budget

- **Monthly**: the first B3 session of each month, 18:30 (after the close reaches the price files), plus a run on the
  first session after a name's new DFP or ITR filing (that name only). First run: Tuesday 01/12/2026.
- Calls: about 18 names x 2 stages a month plus earnings reruns; hard cap 60 DeepSeek calls and 60 judge calls a
  month (in code). Estimated cost under US$ 0.20 a month at the 07/11 prices (to be checked on the pricing page).
- Task "alepha damodaran" on the base Python launcher (`scripts/com_python_base.py`, DECISOES 282).

## Log, screen and email

- `simulador/damodaran_log.csv`: run time, ticker, account, stage, story, parameters, value per share, price, gap,
  action, confidence, judges' answers, checks, tokens, cost; filled later with the 21- and 63-session returns.
- Screen: a section with both pairs of curves (base and + Damodaran), the last valuations (value, price, gap) and the
  orders. Email only when an action is taken (the user: no "nothing new" emails).

## Reading (once, fixed now)

- **When:** the first B3 session on or after **2027-06-01** (six monthly runs).
- **What is read, in this order:** (1) the share of answers discarded by the checks (above 10%: stop and review the
  card and the pack); (2) the 63-session return of names the agent sold or halved against the names it kept, and of
  names it added to; (3) each "+ Damodaran" account minus its base, return and max drawdown; (4) calibration of the
  stated confidence (reliability by decile, Brier score) and of d1.1's and d1-ultra's probabilities.
- **Decision rule:** the agent's valuations stay visible whatever the result; its actions are proposed for real money
  only if (2) and (3) are on its side in both accounts. Six months is weak evidence and is reported as such. N of the
  research count does not change (a forward layer, no historical selection; like the Top-5 and the 12/3).

## Build order

1. Confirm the BRL risk-free source; write the sector-to-industry map; the Damodaran data readers (dated editions).
2. The method card with its verbatim test (test first, it must fail on a made-up passage).
3. The valuation code (FCFF and excess return), test first against one of his published spreadsheets' numbers.
4. Packs, prompts, checks and judges (reusing `core/sentinela` normalization and citation checks).
5. The accounts in the simulator, the log, the screen, the email; advisor review of the order logic.
6. A real end-to-end run in a scratch folder on 3 names, then the scheduled task.
