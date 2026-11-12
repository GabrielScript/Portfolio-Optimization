# 📊 Aleph Quant — Alocação quantitativa em ações da B3

> **Autor:** Gabriel Estrela Lopes

**2026-11-04: the app is named "Aleph Quant"** (DECISOES 250); **B3 only since 2026-11-03** (DECISOES 240). The app covers B3
stocks, ETFs and BDRs (the US BDRs and ETFs listed on B3 included) with CVM, BCB and ANBIMA data. The US stocks
screen and the US stock pipelines (Tiingo tasks, `alepha-quant-eua`) are on standby: code and data kept, nothing
deleted, nothing on the screen. Older sections below keep the old name where they record history.

## Version 1.0 (2026-10-23)

**Alepha** is the name of a personal decision tool. It is not an investment fund offered to third parties and
not investment advice (CVM Resolution 175/2022 covers funds; Resolution 20/2021 covers securities analysts).

- **Real-money default: the Conservadora** (Blitz & van Vliet, 2018: the 15 names of low 36-month volatility, high
  12-month dividend yield and 12-1 momentum, equal weights, at most 4 per sector), the only strategy that passed the
  pre-registered final proof after costs and IR (DECISOES 82). The user chose it on 2026-10-23 for the aggressive
  investor; BOVA11 + satellite and the study stay as options on the screen.
- **Backtest without survivorship bias, 2010-2026, three phases** (`docs/conservadora-vol-2026-10-23.txt`): CAGR
  16.4-17.4% vs 6.7-7.2% for BOVA11; annualized volatility 18.0-18.3% vs 23.4%; downside volatility 12.8-12.9% vs
  16.5-16.6%; maximum drawdown about −40% vs −50%. After costs and IR with R$ 10 thousand
  (`docs/execucao-2026-10-20.txt`): P[exit value > BOVA11] 0.981 / 0.995 / 0.988.
- **The live proof is still to come.** Paper-v2 registers the Conservadora every 63 sessions since 19/10/2026 (task
  "paper v2", weekdays at 08:45). Operational reading at 126 sessions (~27/04/2027); decision at 252 (~25/10/2027):
  the after-tax exit value must be at least BOVA11's (main criterion), and a secondary criterion pre-registered on
  2026-10-23 asks for a volatility no higher than BOVA11's. One year cannot separate alpha from luck: the paper proves
  the operation (data, routine, rules).
- **Orders from the current positions (S4, `core/rebalanceio.py`):** the plan takes the holdings to the registered
  portfolio. It splits buys and sells into standard lots plus fractional, and prices each order with the lake's
  liquidity (B3 fees + half spread + square-root impact). It computes the realized gain at average cost and the
  month's IR, where the R$ 20 thousand exemption is a cliff, so each sale shows its marginal IR. The no-trade zone was
  pre-registered and did not pass (`docs/zona-2026-10-23.txt`): the screen shows it as information and sends every
  order.
- **After v1.0 (git tag `v1.0`), automated since 2026-10-23:** the task "paper manutencao" (weekdays at 10:00,
  `scripts/manutencao.py`) waits for the lake and the paper routines, then saves the paper-v2 readings to
  `paper-v2/leituras/` (the pre-registered ones at 126 and 252 sessions, and once a month a progress reading that
  decides nothing), rebuilds `lago/liquidez.parquet` when the lake closes a month, commits exactly those files and
  pushes `main` fast-forward (never forced), which also publishes the paper routines' commits to the deployed app.
  Saved readings raise a Windows notification. The 12h watchdog ("paper vigia") warns when any routine, the upkeep
  included, did not run or failed, and when the app checkout is not on `main`. By hand: nothing, unless a
  notification asks (a `git switch main` or a `git pull`). Further research (meta-labeling, a Monte Carlo bench, a
  news/filings veto, an election-window study) is optional and each item needs its own pre-registration.

## Version 1.1 (2026-10-24): final

**Alepha v1.1** (git tag `v1.1`) is the finished optimizer: v1.0 plus the changes below. Nothing is left to
build for the strategy itself; what remains is the live proof (paper-v2, decision at 252 sessions, ~25/10/2027),
which runs and reports by itself.

- **Start with little money:** below ~R$ 1,870 the Alepha plan keeps the BOVA11 you hold and buys the
  Conservadora's stocks one share at a time (fractional market), most underweight first, each stock at most 20% of
  everything you hold. The screen says which names wait for more capital and that the portfolio is far from the
  tested one until then.
- **Prices one session old at most** on the deployed app (`lago/precos_recentes.parquet`, committed by the daily
  upkeep) plus the 15-minute-delayed quote for the plan. Fix of 2026-10-24: Yahoo's 1-minute bars stop at 16:54,
  before B3's closing auction, so once the lake has a session's close the plan uses that official close
  (`precos_do_app.com_fechamento_oficial`). Between the auction and the 22:30 upkeep the plan still shows the last
  pre-auction trade (usually within 1% of the close).
- **On the screen:** the Conservadora's backtest by period and calendar year (bad years included), and a market panel
  with the election timing (the pre-registered study: election weeks x1.43 more volatile, p 0.0013, direction not
  tested), the news tone of the forward log scored by Jev, the CVM material filings, the Focus survey, and a table of
  every stock, BDR and ETF the lake exports (about 1,545 traded in the last 15 days: 374 stocks and units, 907 BDRs,
  264 ETFs; filter by type) with risk, multiples and dividend yield. Names and sectors of BDRs and ETFs come from
  B3's registry (`lago/ativos.parquet`); BDRs are price-only and have no CVM statements. Information only: none of
  it changes the plan.
- **Sidebar:** "Todos os ativos (181)" ticked by default; the "Alepha" group holds the fields that change the
  plan (amount, plan, cost), the "Estudo" group the ones that change only the study below it.
- **Upkeep three times a day** (Mon-Fri 10:00, 16:00 and 22:30, the last after the nightly lake run; 16:00 since
  2026-10-25, for a mid-day headline publish): paper-v2 readings, liquidity, recent prices and the market panel's
  files, committed and pushed, so the deployed app shows the day's close the same evening.

### After v1.1 (2026-10-24): classic strategies compared

Research only; the Alepha plan (the Conservadora) does not change. `scripts/avaliar_classicas.py` (pre-registered,
`core/classicas.py`, output `docs/classicas-2026-10-24.txt`) runs Piotroski's F-score, Greenblatt's Magic Formula and
Carlisle's Acquirer's Multiple against the Conservadora on the same engine: lake universe without survivorship bias,
CVM statements point-in-time (the lake now also exports the gross profit, DRE 3.03), 15 names, 4 per sector, every
63 sessions, 10 bps, first decision 01/2014.

| 2014–10/2026 (3 start dates) | CAGR | Max drawdown | Vol | P[> BOVA11] |
|---|---|---|---|---|
| Piotroski F-score | 13.3–19.3% | −54% to −57% | 24% | 0.64–0.93 |
| Acquirer's Multiple | 16.0–18.1% | −54% to −56% | 24% | 0.83–0.89 |
| Magic Formula | 12.7–17.1% | −52% to −54% | 24% | 0.60–0.84 |
| **Conservadora (Alepha)** | **15.7–19.8%** | **−40% to −42%** | **19%** | 0.78–0.94 |
| BOVA11 | 11.7–12.7% | −47% | 24% | — |

All three are valid signals on the B3 (monthly IC 0.046–0.059, t ≈ 5), with returns like the Conservadora's but
12–15 p.p. deeper drawdowns; none beats it or BOVA11 with confidence. Over this shorter window the Conservadora's
edge is lower risk more than higher return; its pre-registered proof (2012 start) stands.

### After v1.1 (2026-10-24): a light per stock of the plan

Display only: the weights, the orders and the paper routines never read it. Under the Alepha plan, each of the 15
stocks gets 🔴/🟡/🟢 from its CVM material filings ("Fato Relevante", "Comunicado ao Mercado") of the last 21
sessions, each scored by Jev (`filings-v1`, every 2 hours, in the lake): red = the most likely tone is bad for the
shareholders and P(strong effect on the price) >= 0.5; green = the same with a good tone; yellow = no strong filing,
or none scored yet; red beats green. A stock that turned red after the registered portfolio gets a warning with the
approximate date of the next rebalance (63 lake sessions; paper-v2 decides). The scores reach the deployed app as
numbers in `lago/fatos_relevantes.parquet` (`scripts/painel_do_app.py`, the daily upkeep). Filings delivered before
23/10/2026 have no score (backfilled), so the lights are yellow until new filings arrive. The news/social column
shows "—" until the log's items carry a ticker (`items-v2`). The light was never tested as a sell rule.

### After v1.1 (2026-10-24): headlines on the deployed app

This repository stays public and keeps no headline text. The daily upkeep (`scripts/manutencao.py`, weekdays
10:00, 16:00 and 22:30) publishes the 50 newest relevant headlines of the forward log (title, link, Jev topic and
tone; never the summary text) as `manchetes.json` to the private repository `GabrielScript/alepha-manchetes`,
through a clone next to the projects folder and the owner's stored git login. The deployed app downloads that file
with a fine-grained GitHub token (Contents: read-only on that one repository) kept in Streamlit secrets
(`core/manchetes_online.py`):

```toml
[manchetes]
token = "github_pat_..."
```

The panel shows when the headlines were published. Without the token, or with GitHub down, the page shows only the
numbers, as before. On the lake's own PC the app keeps reading the headlines live from D:. The file is published
three times a day (the 16:00 run was added on 2026-10-25 at the user's request); the lake itself collects and
scores the news every 2 hours. A late file holds true headlines with its publish time on screen, and nothing decides
on them (display only).
`.streamlit/secrets.toml` is in `.gitignore`.

### Version 1.2 (2026-10-25, git tag `v1.2`): the DeepSeek adviser

Under the market panel, "Assessor do Alepha (DeepSeek)" gives one opinion per click on the registered plan's
stocks, for the extremely aggressive investor since 2026-10-25 (`core/assessor.py`, `ui/assessor.py`; design in
`docs/superpowers/specs/2026-10-25-assessor-deepseek-design.md`). DeepSeek V4.1 Flash (`deepseek-flash`, thinking
mode) reads a numbered evidence pack (BCB and Focus, the plan, each stock's price, risk and multiples, CVM filings
and lights, headlines) and may call tools: stock metrics, filings, BCB series, DeepSeek Search and Fetch, Firecrawl
news, ScrapeCreators Threads posts. Every claim must cite one evidence and quote it; the app keeps a claim only when
the quote is in that evidence and the claim's numbers are in the quote, and a stance needs three BCB-only reasons
against the thesis, written before it. Display only: the weights and orders never read it. Keys in Streamlit
secrets (or the environment variables `DEEPSEEK_API_KEY`, `FIRECRAWL_API_KEY`, `SCRAPECREATORS_API_KEY` locally);
without the DeepSeek key the section says it is unavailable, and a missing search key only disables that tool:

```toml
[deepseek]
api_key = "sk-..."
[firecrawl]
api_key = "fc-..."
[scrapecreators]
api_key = "..."
```

One opinion cost US$ 0.014 to 0.031 off-peak in four real runs (3 to 5 requests, 69% to 94% of the input from the
cache); the same pack, plan and day is cached for 6 hours for every visitor, so a second click costs nothing, and at
most 8 paid opinions a day run per app process (the public app spends the owner's key). Model and web text is
shown with Markdown escaped.

- **Investor profile (2026-10-25, the user's choice):** extremely aggressive: high volatility, deep drawdowns and
  concentrated, asymmetric bets are accepted when the potential return pays for them; only a permanent loss of
  capital (fraud, insolvency, value destruction) is rejected. The model is told not to prefer risk for its own sake
  and that the plan itself is low-volatility and unlevered. First live opinion with it: US$ 0.0185, no claim dropped.
- **Flash vs Pro (DECISOES 137, same evidence pack, two runs each):** Flash US$ 0.016-0.030 and 2-3 minutes, 1 and
  7 claims dropped, 6 tool calls a run; Pro (`deepseek-v4-pro`) US$ 0.074-0.106 and 4-6 minutes, no claim dropped
  but no tool call at all (no news, posts or web). Run-to-run ranking correlation 0.85 (Flash) and 0.88 (Pro);
  0.60-0.73 across models; PETR4 first in all four. Flash stays the default; `pedir_parecer(..., modelo=MODELO_PRO)`
  switches.
- **Retrieval:** a numbered evidence pack plus tools, not a vector or graph store: the data is small and tabular,
  exact lookup is what the quote check needs, and the fixed pack is what the prompt cache matches.
- **Not a backtest:** the model was trained on the past it would be tested on (look-ahead), so its value can only be
  measured forward; a pre-registered weekly log of its opinions is proposed, not started.

### After v1.2 (2026-10-27): optional US leg

- **"Incluir EUA" (sidebar, off by default):** adds 30 BDRs of US companies (Apple, Microsoft, Nvidia, Alphabet,
  Amazon, Meta, Tesla, Berkshire, JPMorgan and others) plus IVVB11 (S&P 500) and NASD11 (Nasdaq-100) to the
  optimizer's universe, each US group under its own 30% sector cap (`ativos_eua.py`). Chosen as the US companies
  above the app's liquidity floor on 23/10/2026 (of the ~820 BDRs listed on the B3, ~790 trade too little).
- **Data:** the lake's COTAHIST prices with splits adjusted; the lake has **no BDR dividends**, so US returns are
  price-only (in reais, the dollar inside). A bad AVGO34 print (17-20/09/2021) is marked as unknown return.
- **Tax:** BDR gains pay 15% with no R$ 20 mil monthly exemption and their dividends go through the carnê-leão; the
  app shows this as a caption. The real-money plans (Conservadora, registered satellite) stay Brazilian stocks only.
- **Why off by default:** in a descriptive backtest (29/04/2014-23/10/2026), the US leg improved Min. Vol (CAGR
  11.3% -> 13.9%, vol 15.9% -> 14.0%) and hurt Kelly (18.5% -> 17.1%); no mix beat IVVB11 alone (20.8%).

### After v1.2 (2026-10-27): quantum portfolio test (QPO), not adopted

- **What was tested (pre-registered, `docs/superpowers/specs/2026-10-27-qpo-design.md`):** at each quarterly
  rebalance, pick 15 of the 20 best Conservadora names by a QUBO (Conservadora score as Grinold-Kahn alpha,
  Ledoit-Wolf risk, growth-optimal risk aversion), weight them by the existing max Sharpe, unlevered. The QUBO is
  solved two ways: exactly (enumeration) and by QAOA (Dicke start + XY-ring mixer, p = 3, 150 Adam steps,
  4,096 shots). Same engine, costs and frozen data as the Conservadora (2010-2026, 3 start offsets).
- **Result (`docs/qpo-2026-10-27.txt`):** CAGR, by start offset:

  | Arm | 0 | 21 | 42 |
  |---|---|---|---|
  | QPO-exact | 19.7% | 16.8% | 16.0% |
  | QPO-QAOA | 19.4% | 16.5% | 16.7% |
  | Conservadora (15, equal weights) | 17.4% | 16.4% | 16.9% |
  | BOVA11 | 7.2% | 6.8% | 6.7% |

  Both QPO arms beat BOVA11 (P 0.96-0.99) but not the Conservadora (P 0.80/0.50/0.32 for QPO-exact), and the
  random-15 control beat QPO-exact in one offset: **not a candidate**. QAOA matched the exact optimum in 77.8% of
  the 198 rebalances (bar 80%), mean approximation ratio 0.996.
- **Why the quantum part cannot add return here:** with 20 candidates the exact optimum takes ~1 s classically, so
  a simulated QAOA can at best equal it; any edge would have to come from the QUBO's inputs, and the inputs did
  not beat the plain Conservadora. None of the papers in `Artigos_QPO` reports an out-of-sample, cost-net edge
  either.
- **Engineering kept:** the QAOA ansatz never leaves the 15-of-20 subspace, so `qpo_kaggle/qaoa_subspace.py`
  simulates its 15,504 amplitudes exactly in torch (equal to PennyLane to 1e-12), ~10-20 s per rebalance on a
  laptop CPU instead of 1-2 min on a Kaggle T4. PennyLane (`qpo_kaggle/qpo_qaoa.py`) stays as the reference.
- **Year by year against every reference (`docs/qpo-anos-2026-10-27.txt`, descriptive, no gate):** QPO-exact
  against the Conservadora, BOVA11, IVVB11 (S&P 500 in reais) and the four most traded crypto ETFs (HASH11,
  BITH11, ETHE11, QBTC11, from the lake, since 2021). It beat all seven in only 5, 4 and 4 of the 17 years (by
  offset). 2025: 23.1/13.4/15.0% against BOVA11 34.7% and the Conservadora 31.5/27.5/33.5%. 2026 to 16/10:
  15.8/19.5/18.4% against BOVA11 28.0%. It beat IVVB11 and every crypto ETF in 2025 and 2026 (the ETFs fell 15-24%
  each year), but the crypto ETFs rose 80-175% in 2023 and 2024 and IVVB11 58% in 2024. No stock picker wins every
  year against the best of seven unrelated assets without hindsight. Tuning the QUBO until it did would be fitting
  the past, so it was not tried.
- **Code review (session 54):** the QAOA reference no longer reseeds numpy/torch globally or changes torch's
  default dtype on import (float64 only inside the solver). The subspace solver refuses sizes the sampler cannot
  take (C(n, k) > 2^24, e.g. n = 28, k = 15) before allocating. A resumed solve refuses results from another QUBO,
  backend, step count or shot count. The QUBO helpers have one source (`qpo_kaggle/qubo_math.py`). None of this
  changes the recorded results.
- **Full universe, session 55 (DECISOES 153-158; `core/qubo_universo.py`):** the first reading chose 15 of the
  Conservadora's own top 20, so the QUBO was an overlay on its ranking. Attempt 47 (pre-registered, N 47) let the same
  QUBO choose 15 of every eligible name of the lake's point-in-time universe (55-171 per date), at most 4 per sector,
  solved by simulated annealing (two independent runs per rebalance; on the 198 recorded 20-name QUBOs both the
  annealer and a SCIP MIQP reproduce the exact optimum, and at full size the runs agreed on 100% of rebalances).
  Attempt 48 (pre-registered before 47 was read) added value (Magic Formula + Trending Value) to the score.

  | Phase 0 / 21 / 42 | P[> Conservadora] | P[> BOVA11] | DSR | Years beating both (of 17) |
  |---|---|---|---|---|
  | 47: full universe | 0.61 / 0.50 / 0.32 | 0.98 / 0.97 / 0.95 | 0.95 / 0.87 / 0.72 | 7 / 6 / 6 |
  | 48: + value | 0.85 / 0.56 / 0.19 | 0.995 / 0.97 / 0.92 | 0.99 / 0.90 / 0.55 | 8 / 7 / 8 |
  | 49: + value + profit growth | 0.80 / 0.50 / 0.43 | 0.99 / 0.96 / 0.96 | 0.99 / 0.85 / 0.77 | 10 / 7 / 7 |

  None is a candidate (the gate needs 0.95 against both in every phase). Attempt 49 ran only because profit growth
  first passed its own IC test (IC 0.033, t 3.63, 2012-2026, `docs/crescimento-lucro-2026-10-28.txt`). 48 looked
  strong in phase 0 only, so its edge depends on the rebalance date. The optimizer is not the bottleneck: the
  expected returns are. Better solvers or quantum hardware would find the same portfolios. **The QPO line is
  finished** (DECISOES 162): the code stays, and profit growth moves on as a signal for the Conservadora's work.

### After v1.2 (2026-10-28): the composite and its forward paper log (paper-v3)

- **Attempt 51 (pre-registered, `docs/superpowers/specs/2026-10-28-composto-crescimento-design.md`):** the
  Conservadora's own rule (15 names, 4 per sector, equal weights, fixed rules) ordered by the mean of three scores that
  each passed their own IC test: the Conservadora score, value (Magic Formula + Trending Value) and profit growth.
  CAGR 21.4 / 19.9 / 19.1% against the Conservadora's 17.4 / 16.4 / 16.9% (phases 0 / 21 / 42), but P[> Conservadora]
  0.968 / 0.955 / **0.864**: it missed the 0.95 bar in one phase, so it is **not adopted** and its backtest line is
  closed (`docs/composto-crescimento-2026-10-28.txt`). It is the closest of the 51 trials.
- **Paper-v3 (pre-registered before any code, `docs/superpowers/specs/2026-10-28-paper-v3-composto-design.md`):** the
  same rule, unchanged, recorded on data nobody has seen yet. Every 63 sessions, `scripts/paper_v3.py` decides the
  composite **and** a Conservadora control on the same close from the lake's live files (it runs the lake's
  `universe` and `fundamentals` first and does not decide without the month's market values), and appends both to
  `paper-v3/composto.csv` and `paper-v3/conservadora.csv` with the input files' hashes. No money moves. Decision at
  252 sessions: the composite passes if its exit value after costs and income tax is at least the Conservadora's.
  One year cannot separate skill from luck, so a pass is evidence, not adoption. Task "paper v3", weekdays 10:30;
  the 12:00 watchdog checks it. Paper-v2 is unchanged. The DSR N stays 51: the same configuration on new data is
  not a new trial (`docs/tentativas.csv`).

### After v1.2 (2026-11-06): the Sentinela, a DeepSeek watch agent in a fictitious account

- **What it is (spec `docs/superpowers/specs/2026-11-05-sentinela-design.md`, DECISOES 262-280):** not a strategy. Every
  hour from 08:20 to 19:20 on B3 sessions, code reads the official sources (CVM filings, also the same-day ones from
  RAD/ENETWeb; the lake's prices and volumes; B3 lending; BCB SGS; IBGE SIDRA; Tesouro RTN) into short verbatim items.
  Only a Composite name with new items gets a DeepSeek-V4.1-Flash scan at low effort; every citation is checked
  against the item's text in code (a failed check discards the answer). A checked `grave` gets one max-effort decision,
  which acts only at confidence >= 0.70 and only in the fictitious account "Composite + Sentinela" (R$ 10,000 from the
  06/11/2026 close), beside the plain Composite. Liquid's d1.1 and d1-ultra judge each company item in shadow. Caps:
  40 scans and 6 decisions a day. Every call is logged with tokens and cost; Brier and log scores are derived from the
  log; emails go out for each `grave` and a 19h digest, only when there is news (no "nothing new" emails). The one
  reading (first session on or after 2027-01-08) decides whether it keeps its autonomy (rule in
  `core.sentinela.pontuacao.autonomia`).
- **Since 07/11 (spec amendments 7-8, DECISOES 303-304):** each new Fato Relevante of the Composite's names is
  downloaded once from RAD and its PDF text (first 3,000 characters, `pypdf`) follows the subject line, so the scan
  sees what changed, not only the title; Comunicados stay subject-only. An answer the code checks discard (or one not
  JSON or cut) is asked again once with the reason; the rules do not change and network failures are not retried.
- **Run it:** `scripts/sentinela.py` (`--semear` before the first live run, `--forcar --sem-email --pasta DIR` for a
  test). Task "alepha sentinela" through `scripts/com_python_base.py` (base Python + the venv's packages).
- **Also in this round:** the Conservadora left the main line (charts and headline; it stays in the tables); the 28
  formulations tested against the Composite are on the screen; `scripts/aporte_real.py` prints the ticket of the real
  buy of 19/12 (closing-auction orders, the user places them). Since 06/11 the amount follows the fictitious "Composto +
  Sentinela" balance: under R$ 11,000 nothing, then R$ 500, R$ 750 from 12,500, R$ 1,000 from 13,000; it also prints
  the agent's alpha against "Composto (base)" (DECISOES 283-284). The paid adviser opinion needs an access code; d1.1
  scores the sentiment log beside Jev (lake `b3lake.d1score`).

Dashboard em Python/Streamlit para otimizar carteiras de ações da B3 com foco no **investidor arrojado**.
Usa Markowitz/Kelly com estimadores robustos (Ledoit-Wolf + shrinkage estilo Jorion), alocação hierárquica
(HRP) e um **walk-forward fora da amostra, líquido de custos**, para que toda estratégia seja comparada nas
mesmas condições.

---

## 🚀 Funcionalidades

### 🎯 Objetivos de otimização
- **Núcleo Mín. Vol + satélite Kelly (padrão do estudo, só ações)** — (1 − f)·w_MínVol + f·w_Kelly, com o satélite f de
  0 a 50% (padrão 30%). Foi a mistura aprovada no teste pré-registrado sem viés de sobrevivência
  (`docs/universo-dinamico-2026-10-18.txt`): em 10 anos, Sharpe 0,35 × 0,36 do Kelly com drawdown de −46% × −63%.
- **Crescimento máximo (Kelly)** — `max μ'w − (λ/2)·w'Σw`, com aversão λ ajustável (padrão λ = 2). Opção: no mesmo
  teste, perdeu Sharpe para a Mínima variância em 5 anos.
- **Máximo Sharpe** — resolvido de forma exata (transformação de Charnes-Cooper, problema convexo).
- **Mínima variância.**
- **HRP** — Hierarchical Risk Parity (López de Prado); ignora o retorno esperado. Na tela, é projetado nos tetos
  antes de virar plano; nas comparações do walk-forward, fica puro.
- **Máximo retorno com teto de volatilidade** — mantido como legado: gera soluções de canto que concentram
  o erro de estimação.

### ⚙️ Restrições e parâmetros
- **Regras de risco fixas** (`core/risco.py`), conferidas na carteira de hoje e nos pesos efetivos do plano: sem
  alavancagem (pesos ≥ 0, soma ≤ 100%), até 20% por ação e por emissor, 30% por setor (papel sem setor também
  recusa) e 40% a.a. de volatilidade ex-ante. A tela só aperta: os sliders param nesses valores. Carteira fora das
  regras não vira plano. Stop relativo do satélite (razão satélite/BOVA11 −25% desde o pico) e pausa de um pregão
  depois de perda diária de 5% já existem como funções; valem quando houver posições registradas no paper trading
  (`core/registro.py`). A rotina diária que registra o satélite e aplica as duas regras (`scripts/paper_trading.py`,
  pré-registrada, com a decisão do item 3 do critério de pronto fixada antes) roda agendada de segunda a sexta às
  8h30. O 1º registro, no fechamento de 16/10/2026 (`paper/carteiras.csv`), começou o relógio de 6 a 12 meses.
- **As mesmas regras no motor do walk-forward com deslistadas** (`estrategia_otimizada_nas_regras`, `Congelamento`,
  recusa = mantém a carteira), testadas num pré-registro (`docs/regras-no-motor-2026-10-18.txt`): 0 recusas em 60
  rebalanceios; o stop relativo disparou nos dois horizontes (05/2025 e 12/2025); o padrão com regras teve Sharpe
  −0,06 (−0,13 sem) em 5 anos e 0,28 (0,35 sem) em 10. O teto de setor custa à Mín. Vol em 10 anos (0,19 × 0,28).
- **Robustez fora da amostra, 2010–2026** (`scripts/avaliar_robustez.py`, pré-registrado; saída em
  `docs/robustez-2026-10-19.txt`): janelas móveis de 3 e 5 anos, subperíodos, P[> BOVA11] também em log-retorno,
  DSR com as tentativas de `docs/tentativas.csv` e MinTRL, em 3 fases de rebalanceio. O satélite com regras teve IR
  de 0,32 a 0,37 contra o BOVA11 (P 0,90–0,93, MinTRL de 20 a 27 anos); o plano do dinheiro real, com o stop
  relativo, de 0,04 a 0,18 (o stop dispara em 2 a 3 anos na maioria das janelas de 5 anos); o Núcleo Mín. Vol 70/30
  chegou a P 0,95–0,96 no período inteiro, mas só por causa de 2010–2014 e com DSR abaixo de 0,95: não cumpre o
  critério de pronto. A janela de estimação de 10 anos não passou (P[R10 > R5] de 0,20 a 0,48). Detalhes no
  ROTEIRO §4.11.
- **S2.1, Conservadora de Blitz e van Vliet** (`scripts/avaliar_conservadora.py`, pré-registrado; saída em
  `docs/conservadora-2026-10-20.txt`): baixa vol + dividend yield + momentum 12−1, 15 nomes com pesos iguais. IC
  mensal de 0,081 (t 5,9) em 2008–2026 e 0,059 em 1998–2007; no walk-forward 2010–2026, 16,3–17,3% a.a. contra
  6,7–7,2% do BOVA11 e 13,0–14,3% do satélite Kelly, P[> BOVA11] 0,99 e DSR 0,80–0,88. Aprovada e candidata à prova
  final, mas ainda sem IR, com DSR abaixo de 0,95 e sem paper trading: não cumpre o critério de pronto. ROTEIRO §4.7.
- **S2.2 e S2.3, Fórmula Mágica de Greenblatt e Trending Value de O'Shaughnessy** (`scripts/avaliar_valor.py`,
  pré-registrado; saída em `docs/valor-2026-10-20.txt`), com os fundamentos da CVM ponto no tempo (data de entrega)
  do lago: IC mensal de 0,059 nas duas (t 5,4 e 5,5) em 2012–2026, as duas aprovadas como sinal; no walk-forward
  2012–2026, 10,2–12,1% e 11,1–13,5% a.a. contra 8,0–9,2% do BOVA11 e 16,7–18,5% da Conservadora nas mesmas datas,
  P[> BOVA11] 0,59–0,85: nenhuma candidata à prova final. ROTEIRO §4.7.
- **S3, Black-Litterman com views** (`scripts/avaliar_views.py`, pré-registrado; saída em `docs/views-2026-10-20.txt`):
  o Kelly atual com o μ trocado pelo posterior de um prior de mercado e das views da Conservadora e do valor, com a
  confiança pelo IC ponto no tempo. Retorno igual ao do Kelly (10,7–14,4% a.a., MDD −61 a −63%), P[> BOVA11]
  0,82–0,95: não candidata. ROTEIRO §4.8.
- **Alvo de vol com EGARCH** (`scripts/avaliar_vol_alvo.py`, pré-registrado; saída em `docs/vol-alvo-2026-10-20.txt`):
  a exposição de cada estratégia do S2 cai quando a vol prevista por um EGARCH passa da vol de 2 anos, e o resto
  fica no CDI. Corta 7 a 13 p.p. do MDD (Conservadora de −41% para −32/−33%), mas P[com alvo > original] de 0,14 a
  0,89: reprovado nas quatro, ficam as originais. ROTEIRO §4.13.
- **Prova final com execução realista (S4)** (`scripts/avaliar_execucao.py`, pré-registrada; saída em
  `docs/execucao-2026-10-20.txt`): a Conservadora refeita em reais com emolumentos da B3, meio spread e impacto pela
  raiz do tamanho da ordem, e com o IR de pessoa física (15%, isenção de vendas até R$ 20 mil no mês, preço médio,
  prejuízo acumulado), contra o BOVA11 que paga 15% do ganho na venda. Com R$ 10 mil, de 2012 a 2026: 16,1–17,8% a.a.
  líquidos de custo e de IR contra 7,2–8,4% do BOVA11, P[> BOVA11] da saída 0,981–0,995: **aprovada**, candidata ao
  paper-v2. Com R$ 1 milhão o custo sobe para 1,1–1,2% a.a. e o P cai para 0,94–0,98. Não cumpre ainda o critério de
  pronto (falta o paper trading). ROTEIRO §4.14.
- **Paper-v2 da Conservadora** (`scripts/paper_v2.py`, pré-registrado, tag `paper-v2`): registro em
  `paper-v2/carteiras.csv`, separado do paper-v1. A carteira do dia é a que o motor decidiria (conferido contra ele);
  rebalanceio a cada 63 pregões, com o `universe` do lago rodado no dia devido. 1º registro no fechamento de
  19/10/2026 (15 nomes, pesos iguais); leituras em 126 e 252 pregões com custo e IR de R$ 10 mil contra o BOVA11
  (`--leitura`). Tarefa "paper v2" do Windows, de segunda a sexta às 8h45 (`scripts/rodar_paper.py paper_v2`), conferida
  pelo vigia das 12h; o `universe` também roda toda noite na rotina do lago. ROTEIRO §4.10.
- **Núcleo BOVA11 + IVVB11** (`scripts/avaliar_nucleo.py`, pré-registrado; saída em `docs/nucleo-2026-10-20.txt`):
  metade BOVA11, metade IVVB11, de 2014 a 2026. 17,6–18,0% a.a. contra 11,9–12,0% do BOVA11, com vol de 15,7% contra
  23,8% e MDD de −34% contra −47%, mas P[> BOVA11] em log de 0,87–0,89 (pede 0,95) num regime de câmbio só: não
  candidato, o núcleo segue BOVA11. ROTEIRO §4.12.
- **HRP com a distância do paper** (`scripts/avaliar_hrp_distancia.py`, pré-registrado; saída em
  `docs/hrp-distancia-2026-10-20.txt`): López de Prado (2016) clusteriza a distância euclidiana entre as colunas da
  matriz de distâncias (d~), e o app clusteriza a própria d. Com d~, o HRP empata (P[d~ > d] 0,50–0,71, retorno total
  até 2 p.p. menor nas fases 0 e 21) e o HRP-Árvore reprova (P 0,03–0,39): o app fica com a d. A opção do paper está
  em `core/hrp_paper.py`, fora dos caminhos vigiados do paper.
- **Small caps below the universe with R$ 10 thousand** (pre-registered, `scripts/avaliar_iliquidez.py`, DECISOES 99):
  the 15 least liquid tradable names, quarterly, midpoint returns, forced exits, costs and the income-tax exemption,
  against SMAL11. P 0.52-0.65 (0.95 needed), CAGR -5% to -7% a year against SMAL11's +6% to +12%: spreads cost ~15%
  of equity a year and there was no premium even before costs. Closed.
- **Stocks x bonds core** (pre-registered, `scripts/avaliar_acoes_titulos.py`, output `docs/acoes-titulos-2026-10-21.txt`,
  DECISOES 111): BOVA11 50/50 with the IMA-B (net of IMAB11's fee; ANBIMA history files, research use only, not in
  git) or with the CDI, from 12/2008. CAGR 11.1–11.8% and 10.6–11.0% vs BOVA11's 9.4–10.4%, MDD −29% and −24% vs
  −50%, but P[> BOVA11] in log 0.67–0.72 and 0.57–0.65 (0.95 needed): neither is a candidate, the core stays BOVA11.
  The split between stocks and fixed income is the investor's risk choice, not a backtest result.
- **Macro filters on the core** (pre-registered, `scripts/avaliar_filtros_macro.py`, output
  `docs/filtros-macro-2026-10-22.txt`, DECISOES 112): BOVA11 switched to the CDI by an equity-risk-premium veto
  (market E/P minus the 10-year real rate from the Tesouro), a credit veto (IDA-DI minus CDI over 63 sessions) or a
  logistic meta-label on both, each against its own average exposure. No candidate: P log [> control] 0.09–0.11, 0.02–0.21
  and 0.30–0.96 (the meta-label only in one phase). The first run's premium was wrong (the lake priced GOLL at ~R$ 40
  trillion); rerun once on corrected market values (`docs/dados/2026-10-22-cvm/`, same N).
- **Conservadora on the corrected lake** (descriptive, `scripts/rerun_conservadora.py`, output
  `docs/conservadora-2026-10-22.txt`, DECISOES 115): the S2.1 script unchanged on the exports with the CODBDI 58 and
  RPMG3 fixes, same period and N. Still approved: IC 0.083 (t 6.0), P[> BOVA11] 0.991–0.994, CAGR 16.4–17.4%.
- **Paired test, Núcleo Mín. Vol 70/30 + Kelly 30% vs HRP, BOVA11 and IVVB11** (descriptive, not counted in the DSR's
  N; DECISOES 90): vs BOVA11 +8.6–8.9 p.p./yr and P 0.95–0.99 as a single test, but deflated for 32 trials 0.51–0.73,
  60–68% of the excess from 2010–2014 and behind BOVA11 over the last 5 years; vs HRP P log 0.93–0.98 (≥ 0.95 only in
  phase 0 under the conservative methods); vs IVVB11 (since 2014) −3 to −5 p.p./yr. Probable, not established; no
  income-tax test and no paper trading. The screen's PSR/DSR test "beats the CDI", deflated only by the strategies on
  screen.
- Teto de volatilidade anual (restrição cônica), peso máximo por ativo e número máximo de ativos
  (cardinalidade em duas etapas).
- Universo: lista líquida, filtro por setores ou todos os ativos de `assets.py`.
- Histórico de 3 a 10 anos, custo de transação em bps por lado e shrinkage de μ ligável/desligável.
- **Reamostragem de μ (bagging de Michaud)**, opcional e só para o Kelly: média de 100 carteiras calculadas
  em reamostras do histórico (bootstrap circular em blocos de 21 pregões), projetada nos limites. Espalha o
  peso entre mais ativos, mas vem **desligada**: na avaliação pré-registrada (`scripts/avaliar_reamostragem.py`,
  walk-forward de 5 e 10 anos), o Sharpe ficou abaixo do Kelly simples e o drawdown não melhorou. É lenta
  (~1 min a mais no walk-forward de 5 anos com a lista líquida).
- **Growth Score fora da tela.** O código continua em `core/growth.py`, mas o app não o usa: ele lê
  fundamentos e momentum de hoje (look-ahead se aplicado ao passado) e nunca foi testado no walk-forward.
  Assim, a carteira de hoje sai do mesmo método que o walk-forward avalia.

### 📈 Análises
- **Carteira de hoje:** cards de retorno/risco, fronteira eficiente (com o ponto 1/N), composição, tabela
  com setores, contribuição para o risco e matriz de correlação. **Teto por setor** (até 30%, regra fixa) na
  carteira, no plano de compra e no walk-forward; o HRP da tela é projetado nele.
- **Alepha plan (the default since v1.0):** the Conservadora registered in `paper-v2/carteiras.csv`, 100% in
  stocks, from the positions typed on the screen (ticker, quantity, average price) and the cash to invest; month's
  sales outside the plan and the carried loss enter the IR. The plan is refused if the final weights break a fixed
  risk rule; below the minimum capital for the 15 names with 2 shares each (shown on the screen) some names stay out.
- **Plano de compra "BOVA11 + satélite"** (option): a regra do paper-v1, BOVA11 com as cotas inteiras mais
  perto de 70% do valor e, no satélite, a carteira registrada no paper trading (`paper/carteiras.csv`), sem
  reotimizar, antes da carteira do estudo (um estudo inválido não esconde o plano). O satélite fica em caixa quando
  algum papel registrado está sem cotação ou sem setor, quando passaria de 50% do valor, quando não cabem papéis
  bastantes com 2 unidades no teto, quando as quantidades inteiras furariam uma regra de risco ou quando o plano
  ficaria mais longe dos pesos registrados que o próprio caixa; depois do stop relativo do paper, tudo vai ao BOVA11
  (com o registro de 16/10/2026: R$ 300 e R$ 1.300 → 1 e 4 BOVA11 + caixa; R$ 3.000 → 10 BOVA11 + 8 papéis). Aviso
  do dinheiro real antes do plano.
- **Plano de compra do estudo** (escolha "Estudo"): cotação do dia (Yahoo, atraso de ~15 min; sem negócio, o último
  fechamento) e quantas ações comprar de cada uma da carteira otimizada, a partir de carteira zerada e de qualquer valor a partir de R$ 100. Quantidades
  inteiras com **mínimo de 2 ações por posição**; o custo estimado sai do valor antes da divisão; nenhuma
  posição passa do peso máximo. **Investe o valor todo:** antes de otimizar, tira as ações cujas 2 unidades
  passam do peso máximo × valor (com aviso); depois de comprar o piso de cada alvo, segue comprando a ação
  mais abaixo do alvo até nada caber no caixa e no teto (uma posição pode passar um pouco do alvo). Divide
  cada quantidade em lote padrão (múltiplos de 100) e fracionário (ticker com F) e mostra o peso efetivo, o
  valor investido e a sobra em caixa. A sobra não leva um setor além do teto, e o peso efetivo por setor
  aparece abaixo da tabela. Avisa quais ações ficaram de fora e por quê (2 ações passam do teto por ativo,
  passariam do teto do setor ou não cabem no caixa). Ação sem cotação sai do plano com aviso, e o valor dela vai para as outras só até os tetos (o resto fica em caixa). As
  cotações ficam guardadas por 5 min; o botão "Atualizar cotações" busca de novo só as da carteira na tela.
- **Walk-forward fora da amostra:** rebalanceamento trimestral com treino rolante na **mesma janela de
  estimação da carteira de hoje** e teste de N anos (sliders "Janela de estimação" e "Teste fora da amostra").
  Ele baixa janela + teste anos de preços, então ações com histórico mais curto ficam fora dele (a legenda
  lista quais) e continuam na carteira de hoje. As referências de mercado (BOVA11, SMAL11, DIVO11 e IVVB11) usam a mesma
  janela e as mesmas datas; a que não tem histórico para a janela aparece na legenda. Compara o objetivo
  escolhido (e a versão reamostrada, se ligada), 1/N, HRP e HRP-Árvore contra elas e o CDI; o alfa é contra o BOVA11. Mostra CAGR,
  volatilidade, Sharpe e Sortino (sobre o CDI vigente em cada data), Calmar, Omega, **PSR e DSR** (probabilidade
  de o Sharpe verdadeiro ser positivo e a mesma probabilidade descontada pelo número de estratégias comparadas;
  Bailey e López de Prado), VaR/CVaR, drawdown máximo, alfa por regressão com estatística t, pesos ao longo do
  tempo e métricas rolantes.

### 🏦 Dados
- **Preços diários:** lago de dados da B3 (projeto irmão `b3-datalake`), arquivo `lago/precos_diarios.parquet`:
  índice de retorno total por papel a partir do COTAHIST oficial e dos eventos da B3 e da CVM (cisões, bonificações,
  JCP líquido do IR), terminando no último fechamento; ações e units negociadas nas últimas duas semanas mais BOVA11,
  SMAL11, DIVO11 e IVVB11 (este desde 29/04/2014), desde 2005. Dia sem negócio fica sem preço (o Yahoo repetia o anterior), então ação pouco
  negociada cai na exigência de cobertura ≥ 95% da janela. A tela mostra até que pregão vão os preços.
- **Cotação do dia:** Yahoo Finance (`yfinance`), só para o plano de compra.
- **Banco Central (SGS):** CDI diário (série 12), Selic meta point-in-time (série 432) e PTAX (série 1, só para o painel macro),
  baixados em janelas de até 10 anos, com nova tentativa em falhas transitórias.
- **Daily CDI from the lake (since 2026-10-20):** the `b3-datalake` nightly routine keeps SGS 12 since 1986 and
  exports `data/b3lake/cdi_daily.parquet`; `core/cdi_lago.py` reads it and asks the SGS only for the days after the
  last night's run. Without the file (Streamlit Cloud: `data/` is not versioned) the app downloads from the SGS as before.
- **Lago de dados da B3 (projeto irmão `b3-datalake`):** exports em parquet gerados a partir do COTAHIST
  oficial (1986–hoje) e dos eventos da B3 e da CVM: retorno total ajustado desde 07/1994 de 3.948 papéis,
  **incluindo deslistados e em recuperação judicial** (troca de ticker, como ITAU4 → ITUB4, continua a mesma
  série), painel mensal de liquidez/volatilidade/microestrutura e índices próprios. Os preços diários do app
  já vêm dele. O lago também publica o **universo ponto no tempo**: em cada mês, as ações e units negociadas em
  ≥ 90% dos pregões dos 12 meses anteriores, com volume diário mediano ≥ R$ 1 mi corrigido pelo IPCA conhecido
  na data, um papel por emissor (417 papéis desde 2005, 186 deles hoje deslistados). O `core/backtest.py` já tem
  o walk-forward com esse universo (`walk_forward_dinamico`); a avaliação pré-registrada rodou em 2026-10-18
  (`docs/universo-dinamico-2026-10-18.txt`) e o app ainda usa a lista de hoje. Uma rotina diária no Windows (Agendador de Tarefas, seg–sex
  às 21h45 com novas tentativas até 23h45 e às 8h do dia seguinte) traz o pregão do dia para o lago, confere
  com portões automáticos (troca de ticker sem revisão, arquivo truncado, retorno ≥ 50% sem revisão, gold
  reproduzível) e só atualiza os exports quando todos passam; quando algo falha, avisa no Windows.
- **Setor por emissor, com as deslistadas** (`data/b3lake/sectors.parquet`, lido por `core.lago.ler_setores`):
  `SETOR_ATIV` do cadastro da CVM mapeado nos 10 setores do app; cobre 97% dos emissores do universo. É a taxonomia
  da CVM, mais grossa que a da B3 do `assets.py` (149 de 178 concordam): o motor dinâmico usa a do lago para vivas e
  mortas, e a tela continua com o `assets.py`.
- **Proventos por papel e data ex** (`data/b3lake/dividends.parquet`): dinheiro pago por ação (JCP líquido do IR),
  com o yield sobre o fechamento do dia ex, para os sinais de payout do S2.
- **Correção de 2026-10-18:** a B3 lista entre os proventos em dinheiro o custo atribuído das bonificações, que o
  lago contava de novo no dia do evento de ações (AXIA3 22/12/2025 +22% em vez de −3%; CYRE3 e RENT3 também). Os
  preços do app foram republicados sem isso.
- **Em desenvolvimento (só no `core/`, ainda sem tela):** painel macro mensal para cenários — Focus
  (expectativas do BCB), curva prefixada do Tesouro Direto e fatores de surpresa macro.

---

## 🛠️ Tecnologias

- **[Streamlit](https://streamlit.io/)** — interface.
- **[CVXPY](https://www.cvxpy.org/)** (solvers CLARABEL/SCS) — otimização convexa.
- **[scikit-learn](https://scikit-learn.org/)** — covariância de Ledoit-Wolf.
- **[SciPy](https://scipy.org/)** — clusterização hierárquica (HRP) e estatística.
- **[Pandas](https://pandas.pydata.org/) & [NumPy](https://numpy.org/)** — dados.
- **[yfinance](https://pypi.org/project/yfinance/)** e **[API SGS/Olinda do BCB](https://dadosabertos.bcb.gov.br/)** — fontes.
- **[Plotly](https://plotly.com/)** — gráficos.

---

## 📂 Estrutura do Projeto

```
Portfolio-Optimization/
├── app.py                    # Orquestrador Streamlit (camada fina)
├── assets.py                 # Universo de ativos e setores da B3
├── ativos_tela.py            # Market panel's full list: every stock, BDR and ETF of the lake, named and typed
├── visualizations.py         # Gráficos Plotly
├── core/                     # Lógica pura, sem Streamlit
│   ├── config.py             #   Objetivos, configurações, Selic → taxa aritmética
│   ├── lago.py               #   Preços diários do lago da B3 (lago/precos_diarios.parquet)
│   ├── market_data.py        #   Preços (lago), calendário B3, CDI, Selic, cotação do dia (Yahoo)
│   ├── cdi_lago.py           #   Daily CDI from the lake's copy, topped up from the SGS (outside the paper guards)
│   ├── ordens.py             #   Plano de compra: pesos + cotações → quantidades inteiras
│   ├── risco.py              #   Regras de risco fixas: aprova ou recusa a carteira (não muda pesos)
│   ├── registro.py           #   Registro do paper trading (CSV que só acrescenta linhas)
│   ├── paper.py              #   Paper trading: curva do registro, custos, decisão do dia, leituras de 6 e 12 meses
│   ├── bcb.py                #   Cliente SGS do Banco Central (janelas, cache, retry)
│   ├── estimation.py         #   Retornos, Ledoit-Wolf, Jorion, tilt growth
│   ├── optimization.py       #   Kelly, Max-Sharpe, Min-Vol, HRP, fronteira, projeção nos limites
│   ├── reamostragem.py       #   Carteira reamostrada (bagging, bootstrap em blocos)
│   ├── hierarchical.py       #   HRP e HRP-Árvore
│   ├── hrp_paper.py          #   HRP com a distância d~ do paper original (pesquisa, fora da tela)
│   ├── metrics.py            #   Sharpe, Sortino, Calmar, VaR/CVaR, alfa, drawdown, PSR/DSR
│   ├── robustez.py           #   MinTRL, P[> referência] em log-retorno, janelas móveis
│   ├── backtest.py           #   Walk-forward com custos e CDI point-in-time (universo ponto no tempo, regras no motor)
│   ├── poder.py              #   Poder dos portões do S2 por simulação (IC plantado no painel mensal)
│   ├── sinais.py             #   Sinais do S2: Conservadora (vol 36m, DY 12m, momentum 12−1) e a carteira dela
│   ├── valor.py              #   Sinais do S2 com fundamentos da CVM: Fórmula Mágica e Trending Value
│   ├── vol_alvo.py           #   Alvo de vol com EGARCH sobre uma curva fora da amostra (pesquisa)
│   ├── views.py              #   Black-Litterman com views dos sinais do S2 (S3)
│   ├── impostos.py           #   IR de pessoa física em ações: apuração mensal, isenção e prejuízo (S4)
│   ├── execucao.py           #   Curva refeita em reais com custo por liquidez, IR e valor de saída (S4)
│   ├── rebalanceio.py        #   Orders from the current positions: lots, cost, month IR, no-trade zone (S4, v1.0)
│   ├── volatilidade.py       #   Annualized and downside volatility (rerun and paper-v2's secondary criterion)
│   ├── periodos.py           #   The Conservadora's frozen results by period and calendar year
│   ├── precos_do_app.py      #   Committed price file + the recent sessions (deployed app)
│   ├── sentimento.py         #   News tone (Jev), CVM filings, election timing: information only
│   ├── macro.py              #   Focus survey summary: information only
│   ├── indicadores.py        #   Per-stock multiples, dividend yield and risk: information only
│   ├── growth.py             #   Growth Score (Yahoo/CVM), fora da tela até ser validado
│   ├── focus.py              #   Expectativas Focus (em desenvolvimento)
│   ├── tesouro.py            #   Curva prefixada do Tesouro (em desenvolvimento)
│   └── fatores_macro.py      #   Painel e fatores macro (em desenvolvimento)
├── ui/
│   ├── servicos.py           # Cache do Streamlit com chaves explícitas
│   ├── alepha.py             # Alepha plan blocks: positions, month IR, orders table, CVM note
│   ├── painel.py             # Market panel: election timing, news, macro and every stock, BDR and ETF
│   └── secoes.py             # Seções e textos da interface
├── scripts/
│   ├── conferir_painel_macro.py  # Conferência ao vivo do painel macro (usa rede)
│   ├── avaliar_reamostragem.py   # Avaliação pré-registrada da reamostragem (usa rede, ~30 min)
│   ├── avaliar_janelas.py        # Teste pré-registrado das janelas de estimação (usa rede)
│   ├── avaliar_universo_dinamico.py  # Teste pré-registrado com o universo ponto no tempo do lago (rodado)
│   ├── avaliar_regras_no_motor.py    # Regras de risco no motor, pré-registrado (rodado)
│   ├── poder_dos_portoes.py          # Poder dos portões do S2, pré-registrado (rodado)
│   ├── avaliar_robustez.py           # Robustez 2010–2026 e janela de 10 anos, pré-registrado (rodado)
│   ├── avaliar_conservadora.py       # S2.1, Conservadora de Blitz e van Vliet, pré-registrado (rodado)
│   ├── avaliar_nucleo.py             # Núcleo BOVA11 + IVVB11, pré-registrado (rodado)
│   ├── avaliar_valor.py              # S2.2 e S2.3, Fórmula Mágica e Trending Value, pré-registrado (rodado)
│   ├── avaliar_vol_alvo.py           # Alvo de vol com EGARCH, pré-registrado (rodado)
│   ├── avaliar_views.py              # S3, Black-Litterman com views, pré-registrado (rodado)
│   ├── avaliar_execucao.py           # Prova final com execução realista (S4), pré-registrada (rodada)
│   ├── avaliar_hrp_distancia.py      # HRP com a distância d~ do paper, pré-registrado (rodado)
│   ├── avaliar_iliquidez.py          # Small caps below the universe, pre-registered (run, closed)
│   ├── avaliar_acoes_titulos.py      # Stocks x bonds core (IMA-B or CDI), pre-registered (run)
│   ├── avaliar_filtros_macro.py      # Premium/credit vetoes and meta-label on BOVA11, pre-registered (run)
│   ├── rerun_conservadora.py         # S2.1 rerun on the corrected lake, with both volatilities (descriptive)
│   ├── avaliar_zona.py               # No-trade zone with vs without, pre-registered (run: information only)
│   ├── liquidez_do_app.py            # The app's liquidity snapshot (lago/liquidez.parquet), rebuilt by the upkeep
│   ├── manutencao.py                 # Daily upkeep: paper-v2 readings, liquidity, recent prices, panel, push
│   ├── painel_do_app.py              # The market panel's files from the lake's log, filings and statements
│   ├── curvas_conservadora.py        # The frozen rerun's daily curves (results by period and year)
│   ├── avisar.ps1                    # Windows notification of the watchdog and the upkeep
│   ├── vigia.ps1                     # 12h watchdog of the lake, paper, paper-v2 and upkeep routines
│   ├── paper_v2.py                   # Paper-v2 da Conservadora: rotina e leitura, pré-registrado
│   └── paper_trading.py              # Rotina diária do paper trading (S5), pré-registrada (agendada seg-sex 8h30)
├── docs/
│   ├── ROTEIRO.md                # Roteiro: critério de pronto, fases, riscos e decisões pendentes
│   ├── janelas-tiro1-*.txt       # Saídas do tiro 1 do teste das janelas
│   ├── universo-dinamico-2026-10-18.txt  # Saída do S1-P.b (sem viés de sobrevivência)
│   ├── regras-no-motor-2026-10-18.txt    # Saída das regras de risco no motor
│   ├── poder-dos-portoes-2026-10-18.txt  # Saída do poder dos portões do S2
│   ├── robustez-2026-10-19.txt   # Saída do teste de robustez
│   ├── conservadora-2026-10-20.txt  # Saída do S2.1 (Conservadora)
│   ├── nucleo-2026-10-20.txt     # Saída do teste do núcleo BOVA11 + IVVB11
│   ├── valor-2026-10-20.txt      # Saída do S2.2 e do S2.3 (Fórmula Mágica e Trending Value)
│   ├── vol-alvo-2026-10-20.txt   # Saída do teste do alvo de vol com EGARCH
│   ├── views-2026-10-20.txt      # Saída do S3 (Black-Litterman com views)
│   ├── execucao-2026-10-20.txt   # Saída da prova final com custo por liquidez e IR (S4)
│   ├── hrp-distancia-2026-10-20.txt  # Saída do teste do HRP com a distância d~ do paper
│   ├── acoes-titulos-2026-10-21.txt  # Output of the stocks x bonds core
│   ├── filtros-macro-2026-10-2*.txt  # Outputs of the macro filters (10-21 invalid premium; 10-22 corrected)
│   ├── conservadora-2026-10-22.txt   # Conservadora on the corrected lake
│   ├── conservadora-vol-2026-10-23.txt  # The same, with annualized and downside volatility
│   ├── zona-2026-10-23.txt       # Output of the no-trade zone test
│   ├── tentativas.csv            # Configurações já avaliadas fora da amostra (o N do DSR)
│   └── dados/                    # Exports do lago fixados pelos pré-registros (com SHA256SUMS)
├── lago/
│   ├── precos_diarios.parquet    # Preços diários de retorno total publicados pelo b3-datalake
│   ├── liquidez.parquet          # Last closed month's half spread, daily vol and median volume (order costs)
│   ├── precos_recentes.parquet   # The sessions after the committed price file (daily upkeep)
│   ├── sentimento_diario.parquet # News tone per day (numbers only)
│   ├── fatos_relevantes.parquet  # CVM material filings of the app's stocks, last 90 days
│   └── fundamentos.parquet       # Per-stock multiples and 12-month dividend yield
├── tests/                    # Testes com pytest, sem rede (o app inteiro roda via AppTest com dados sintéticos)
└── requirements.txt
```

---

## 🚀 Como Executar

### Pré-requisitos
- Python **3.11** ou superior (NumPy 2 e CVXPY 1.9).

### Passo a passo

1. **Clone o repositório:**
   ```bash
   git clone https://github.com/GabrielScript/Portfolio-Optimization.git
   cd Portfolio-Optimization
   ```

2. **Crie e ative um ambiente virtual** (recomendado — evita conflito com um NumPy 1.x global):
   ```bash
   python -m venv .venv
   # Windows
   .\.venv\Scripts\activate
   # Linux/Mac
   source .venv/bin/activate
   ```

3. **Instale as dependências:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Execute a aplicação:**
   ```bash
   streamlit run app.py
   ```
   O dashboard abre em `http://localhost:8501`. A primeira execução baixa preços e fundamentos e pode
   levar um ou dois minutos.

5. **Rode os testes** (opcional; precisa do `pytest`):
   ```bash
   pip install pytest
   python -m pytest tests -q
   ```

---

## ⚠️ Limitações

- **Viés de sobrevivência.** O walk-forward do app usa a lista atual de tickers; empresas que quebraram ou
  fecharam capital não aparecem, o que favorece as estratégias contra o BOVA11 (o índice não tem o viés). O universo ponto no tempo, com as
  deslistadas, já existe (lago + `walk_forward_dinamico`). Na avaliação pré-registrada sem o viés
  (`docs/universo-dinamico-2026-10-18.txt`), o Kelly λ=2 teve Sharpe −0,31 em 5 anos e 0,36 em 10 (MDD −63%),
  e o núcleo Mín. Vol 70% + satélite Kelly 30% teve 0,35 em 10 anos com MDD −46%: ele virou o padrão da tela.
  Refeito com os preços corrigidos (bonificação) no pré-registro das regras, as duas decisões não mudaram.
- **Setor das deslistadas é o de hoje.** O `SETOR_ATIV` da CVM não tem data de vigência: empresa que mudou de
  ramo fica com o setor atual na história toda; 10 holdings sem setor principal ficam sem setor.
- **Migração de classe sem continuidade.** Quem migrou de PN para ON no Novo Mercado (PCAR4 → PCAR3 em
  2020, SUZB5 → SUZB3) começa no lago sem a história da classe antiga e fica fora de janelas longas.
- Os preços diários vêm do lago desde 2026-10 (antes, Yahoo: errava cisões e usava JCP bruto; registrou
  +112% num dia em que o retorno total real foi +14,1%, cisão do Assaí). Medições publicadas antes disso
  usaram os dados do Yahoo.
- **Sem vantagem comprovada sobre o BOVA11.** A última medição fora da amostra deu empate estatístico; use o
  BOVA11 como núcleo e o otimizador, no máximo, como satélite pequeno. Os horizontes de 5 e 10 anos terminam no
  mesmo dia (são uma história só: ganho forte em 2016–2021, perda em 2021–2026); com tracking error de 16–19%,
  provar o alfa contra o BOVA11 com PSR ≥ 0,95 pediria ~14–30 anos de histórico. A coluna P[> BOVA11] usa a média
  aritmética dos retornos diários, não a riqueza final, e favorece as curvas mais voláteis.
- **O plano do estudo só compra ações.** No modo "Estudo", com valores pequenos (R$ 300) o plano para em "Aumente
  o valor a investir" porque cabem menos de 5 papéis, e o filtro por preço reotimiza num universo menor. O padrão da
  tela é o plano do Alepha (a Conservadora registrada no paper-v2), que não reotimiza.
- **Paper trading próprio, sem plataforma de corretora.** A rotina usa os fechamentos oficiais do lago, o mesmo código
  do walk-forward e um registro versionado que só cresce (auditável; uma conta demo pode ser zerada). Não mede a
  execução: spread, slippage, ordens parciais e a liquidez do fracionário ficam fora (custo fixo de 10 bps por lado).
- **Sem alavancagem.** A regra fixa é 1,0×: com o empréstimo a CDI + spread perto do retorno histórico do BOVA11, a
  alavancagem custaria mais do que rende enquanto o alfa não estiver provado.
- **S&P 500 só como referência.** O IVVB11 (S&P 500 em reais, sem hedge) está no lago desde 2014: 20,1% a.a. ×
  12,4% do BOVA11 em 10 anos e 9,2% × 15,2% em 5, correlação mensal −0,18. O núcleo metade BOVA11, metade IVVB11
  não passou no teste pré-registrado (ROTEIRO §4.12).
- **IR is modeled in the final proof and in the Alepha plan, not in the study's walk-forward** (15% of the
  month's gain; stock sales up to R$ 20 thousand a month are exempt; ETFs are not). Simplifications: an ETF loss does
  not offset a stock gain in the plan, and the exemption counts only the sales typed on the screen plus the plan's.
  The lake already nets the JCP withholding (15% until 2025 and 17.5% since 01/01/2026, LC 224/2025).
- **Janelas curtas.** Com 3–5 anos de histórico, alfas e diferenças de Sharpe raramente são
  estatisticamente significativos; o app mostra a estatística t para lembrar disso. A análise de poder
  (`docs/poder-dos-portoes-2026-10-18.txt`) mediu isso para os sinais do S2: o portão "IC médio mensal ≥ 0,02 e
  t ≥ 2" aprova 87% dos sinais com IC 0,03 e 1% dos sinais sem valor nos 225 meses de 2008 a 2026; em 5 anos, com IC
  que varia de mês a mês, nenhum portão passa de 47%. Por isso os sinais do S2 são julgados na janela inteira.

- **The lake has had one lean audit, not a full review.** Session 35 ran 3 read-only reviewers (adjustments, metrics, data);
  its 4 findings were fixed test-first (DECISOES 96-97) and a Yahoo cross-check of 24 liquid stock-years matched
  (median gap 0.16 p.p.). COTAHIST ingestion and illiquid/delisted names against CVM have not been reviewed yet
  (DECISOES 100). Known leftovers: 59 FRE contradictions, BDRs/FIIs/ETFs without payouts, adjustment only from 1994-07,
  no delisting terminal return.

O que falta e em que ordem: [`docs/ROTEIRO.md`](docs/ROTEIRO.md).

---

## 📝 Isenção de Responsabilidade (Disclaimer)

Esta é uma ferramenta de **apoio à decisão**, sem vantagem comprovada sobre o BOVA11 até o fim da validação em paper trading.
"Alepha" is the name of this personal tool, not a fund offered to third parties (CVM Resolution 175/2022).

> ⚠️ **Não representa recomendação de investimento.** As decisões financeiras devem ser tomadas com base em análise própria ou orientação profissional. Retornos passados não garantem retornos futuros.

---

© 2026 Gabriel Estrela Lopes

## Session 58 notes (2026-10-29)

- Market values with the lake's 1000x share-count slips fixed are frozen in `docs/dados/2026-10-29-cvm`;
  `scripts/sensibilidade_caps.py` shows the value score and attempt 51 unchanged (`docs/sensibilidade-caps-2026-10-29.txt`).
- Quantum Portfolio Optimization is closed and not adopted (DECISOES 144-150).
- Next: NPY pre-registration, then a BR/US sleeve rotation after adding PTAX to the lake (PLAN §15).

## Session 59 notes (2026-10-29)

- Three pre-registered lines read once, none adopted: net payout yield (no gain over the dividend yield), fund
  managers' best ideas from the CVM CDA (attempt 52) and the BR/US/gold rotation with a CDI filter (attempt 53).
  Results in `docs/npy-ic-2026-10-29.txt`, `docs/cda-best-ideas-2026-10-29.txt` and `docs/h2-rotacao-2026-10-29.txt`.
- The lake now exports CVM CDA fund stock positions and the daily PTAX. N = 53 (PLAN §16).
- Options data in the lake (research only until a pre-registered test passes; the no-derivatives rule was lifted on 30/10, DECISOES 191): implied volatility and greeks
  of every traded B3 option since 1994 computed from COTAHIST, B3's daily Price Report from 2018 (futures, options
  on futures, open interest) and a daily snapshot of opcoes.net.br chains to cross-check the greeks.
- Session 63 (30/10/2026): the no-derivatives rule is lifted; options and futures enter only if a pre-registered
  test shows they help (DECISOES 191, 194). Next: a fictitious R$ 10,000 simulator (R$ 9,000 Composite, R$ 1,000 in
  CDI for options) with a dashboard vs BOVA11, IVVB11 and CDI (DECISOES 192); every buy or sell needs an official
  source, never news or sentiment alone (193). US data without survivorship bias: free route researched (195).
- Quantum Portfolio Optimization: finalized, not adopted (see the QPO section above); no further QPO attempt.

## Session 60 notes (2026-10-29)

- Options and derivatives data finished in the lake (research only): B3 Price Reports 2018-2026 (47.97 M rows),
  1.72 M instruments with their terms, and the first full day of opcoes.net.br chains (245 underlyings; its implied
  volatility matches ours within ~1 point). B3's own files had five defects, each handled and tested.
- The lake now keeps CVM's daily fund report (quota, NAV, flows and holders per fund and month, 2000 on, closed funds
  included).
- Attempt 54 pre-registered before any data was read: copy the 15 equity funds with the best 12-month return
  (`docs/superpowers/specs/2026-10-29-top-fundos-copia-design.md`, branch `feat/top-fundos`). It must beat attempt
  52's fund signal and the same score built from all funds, or the line closes. Not run yet; N = 53.
- Leverage 1.5x chosen by the user; it goes into the plan screen once the broker's margin rate is known. The paper
  logs stay unlevered (PLAN §17).

## Session 61 notes (2026-10-29)

- Attempt 54 read once and not passed: copying the 15 equity funds with the best 12-month return gives a monthly IC
  of 0.029 (t 3.79), under attempt 52's 0.032, and no gain over the same score built from all ranked funds (0.001,
  t 0.18). The score leans to large caps (rank correlation 0.37 with market value). The copycat line is closed and
  N stays 53 (`docs/top-fundos-ic-2026-10-29.txt`; coverage in `docs/top-fundos-painel-2026-10-29.txt`).
- Leverage: the broker rate the user found (0.45% a day) is the fee for covering a margin call, not a loan rate, and
  would cost far more than 1.5x could earn. The plan screen keeps 1.0x until an annual borrowing rate (a stock
  forward, "termo", or the broker's leveraged-investment product) is known.
- Next: one pre-registered IC test of option signals as a veto on the Conservadora (PLAN §18).

## Session 62 notes (2026-10-30)

- The Alepha screen's backtest table is now live: the registered Conservadora and the attempt-51 composite rerun on
  the lake to the last close (`scripts/curvas_ao_vivo.py`, descriptive, no rule changed), next to BOVA11, BITH11 and
  QBTC11. Each year since 2010 in the measure picked (return, excess over BOVA11 or the CDI, volatility, drawdown,
  worst month, Sharpe, Sortino, Calmar, beta), a whole-period risk summary (alpha against BOVA11 with its t, CVaR,
  best and worst year) and the rebalance phase to choose. The daily upkeep reruns it about once a week.
- To 29/10/2026 (phase 0): Conservadora 16.1% a year since 02/2010, composite 19.8%, BOVA11 6.0%; in 2026 to date
  -8.7%, -11.0% and +5.4%. The composite is still only a forward candidate (paper-v3).
- The upkeep now also saves paper-v3's readings at 126 and 252 sessions.
- The three funds asked about (Capyvara FIA, AMI FIF Ações, Trend Bolsa Americana Alavancado), invested and copied
  from their public CVM holdings in all 7 combinations (`docs/tres-fundos-2026-10-30.txt`): high past returns, but
  picked after the fact; Capyvara's return does not come from its public stock book (index futures, trading inside
  the month), AMI is almost all one BDR (AURA33), and Trend holds S&P 500 futures, nothing to copy. None enters the
  plan.
- Leverage stays 1.0x. Next: the options veto test (spec approved).
## Session 64 notes (2026-10-31)

- Fictitious-account simulator built on a branch, not in `main` yet: first buy at the close of 03/11, R$ 9,000 in the composite and
  R$ 1,000 in the CDI, against BOVA11, IVVB11 and the CDI. Real money stays on the Conservadora until paper-v3 decides.
- US stocks and ETFs, on branches: a lake with the same layers as the B3 one (raw files on the local disk only, bronze, ClickHouse,
  gold), joined with FRED, French and EDGAR. Only derived results will reach this repository and the deployed app.
- Audit so far: universe vs French's market -0.3% a year, correlation 0.987; the review is not closed. Known limit: few dead companies
  in the source; a free Tiingo download is filling them at 500 symbols a month.
- The Alepha strategies on US stocks are pre-registered (N stays 53) and have not run, so there is no US result yet.

## US stocks (2026-10-31)

The screen "Ações americanas" shows, from derived files in `docs/dados/us-*`: the Alepha strategies in R$ against SPY, ETFs and
Brazilian funds, the gate, today's 15 names, each universe stock's indicators and the US macro. Information only: **no strategy
passes the gate** (`docs/us-estrategias-2026-10-31.txt`; best is the Composite, P at most 0.78, DSR at most 0.22), with 310 of 1,505
dead names so far. Real money stays on the Conservadora. Provider prices never reach git or the app; data, ClickHouse tables and
audit are in the lake's README.

```bash
python -X utf8 scripts/us_avaliar.py > docs/us-estrategias-<date>.txt   # about 30 minutes, after `us-gold` in the lake
python -X utf8 scripts/us_indicadores.py
```

Risk reads monthly returns, since dead names have no daily history. Banks keep P/E and P/B here, unlike the B3 panel, because
EDGAR's equity is a bank's equity. Empty market values are names whose cap the checks reject (16% today, BRK.B among them).

**Still being fixed (2026-11-01, not merged):** five months where the site left a split in the prices move on a lake branch (UAVS 2024-10 goes from +2,727% to -43.5%), and the dead-name list grows from 1,505 to 2,114 names, including Sears, Endo and Seagen. At 500 names a month on Tiingo's free plan the download ends on 01/02/2027. Until then the US numbers may be optimistic, and US readings so far ran on the old panel. 7,133 dead names are not on Tiingo's list at all. DECISOES 229.

## The book's strategies and the hybrid QPO (2026-11-01, in progress)

Testing the formulations of Palomar's book that had not run, on the B3 and on US stocks, with the app's caps and the same gate as every other attempt. **The code is on the branches `feat/qml`, `feat/us-todas` and `feat/us-lote2`, not on this one yet; the app does not show these results until they merge.**

- **Finished on the B3, nothing passes:** the first batch (HRP-projetado, IVarP, IVolP, MDivP, MDecP, RPP, HERC, Mean-CVaR, Mean-CDaR, MVSK, Robust-MV) and the hybrid quantum forecast and QPO. The best new arm, HRP-projetado, makes 16.9-17.6% a year against the Composite's 19.1-21.4%.
- **Running or waiting:** the second batch (exact Kelly with and without the volatility cap, quintiles, box robust mean, subset resampling and bagging, downside risk, MVSK tilting from the Composite, root utility, a return constraint), 14 arms, pre-registered, reading on the B3 under way. Everything on US stocks waits for the frozen data.
- **What the tests say so far:** the forecast matters more than the optimizer. Markowitz with the Composite's signal ties the Composite (18.9-20.3%); with historical means it makes 13-14% with a 62% drawdown.
- **US data:** the Tiingo batch stopped at the free plan's 500 symbols a month, now with a clean stop. Dead names from Tiingo: 978 of 1,505 candidates. Five missed splits were repaired and the gold now builds byte for byte the same.

Past results, before tax, no promise for the future. Details: `DECISOES.md` 226-234.

## Where things stand (2026-11-03)

- **B3, 2010-2026:** the Composite makes 17.4-19.5% a year against the CDI's 9.8% and BOVA11's 5-6%; over the last 5 years 14.7-16.6% against the CDI's 12.7%. The Conservadora makes 14.8-15.8%, and over the last 5 years it only ties the CDI. No formulation from the book beats the Composite; a second batch of 14 has run and has not been read.
- **US stocks, 11/2019-09/2026:** the Composite makes 17.0-20.2% against SPY's 15.5% (22.6% in reais against the CDI's 9.9%).
- **Alepha Quant US (disruptive stocks):** the lake now holds market caps for multi-class and foreign filers, R&D and capex (`exports/us_eua`). The pre-registered funnel (score, QUBO to 15, HRP with a momentum tilt) makes 20.4-27.4% a year against QQQ's 21% and SPY's 15.5%, with 2022 at -26% and drawdowns of 33-43%. Code in the separate repository `alepha-quant-eua`; nothing on the screen yet.
- **None of these passes the gate** (DSR at most 0.89 on the B3 and 0.22 on the US, against 0.95). Real money stays on the Conservadora.
- **Operations:** after the laptop crashed on 02/11, the lake's two scheduled tasks run on the base Python 3.14 interpreter with the venv's packages, because the venv interpreters were blocked from writing outside the project folders.

Past results, before tax, no promise for the future. Details: `DECISOES.md` 236-239.

## Session notes (2026-11-04/05)

- The app is "Aleph Quant", with the user's eagle above the title. The buy plan, positions and monthly IR blocks left the
  page; a section explains what Aleph Quant is (B3 stocks, ETFs and BDRs; Composite and Conservadora; a decision tool for
  real money). The Selic replaced the CDI on screen and in the simulator's cash.
- Market panel: Banco Central indicators (IBC-Br, direct investment, credit granted, its rate, defaults, the last Copom
  minutes) with a "?" each, also read by the DeepSeek adviser; CAGR, Sharpe, Sortino, Calmar, CVaR, alpha, its t and
  information ratio per name; filings with Jev's tone in Portuguese; headlines linked by their title. CVM filings arrive
  weekly (the CVM's open data is published on Sundays).
- DeepSeek adviser: depth picked on screen, "Máxima" (max, default), "Rápida" (high in 3 parallel parts, ~75 s) and
  "Expressa" (high in 5 parts, ~65 s). Max gives the best-checked opinion but its time varies (133-218 s measured on
  05/11); generate it before a presentation and the same click shows it from the cache.
- The simulator's DeepSeek portfolio judges the names of both rule portfolios, next to an equal-weight control.

## Session notes (2026-11-05 evening)

- Fictitious options account ("Conta de opções (fictícia)", `core/simulador_opcoes.py`): R$ 10,000 apart from the
  stocks accounts; on each month's first session it buys a put near 0.95 x spot on the stretched winner near its high
  and a call near 1.05 x spot on the most volatile third's top momentum, 10% of the account, held to expiry, Selic on
  the rest. Only buying (loss capped at the premiums), only two-sided quotes. The upkeep's "opções" step appends
  `simulador/opcoes.csv` and `simulador/opcoes_marcas.csv` from the lake's `option_chain_recent.parquet`.
- Options veto (spec `docs/superpowers/specs/2026-10-29-opcoes-veto-design.md`): read once on 05/11, no signal passed
  (`docs/opcoes-ic-2026-11-05.txt`); options do not enter the stock choice.
- New lake data (not used by the app yet): B3 official indices, daily index compositions, 16 BDI tables (lending,
  open positions, forward, IOPV, margin scenarios...).
- Next: a DeepSeek "sentinel" that watches the Composite's names during the session and alerts by email (DECISOES 261).

## Session notes (2026-11-05 night)

- Sentinela (DeepSeek watch agent) pre-registered before any code: `docs/superpowers/specs/2026-11-05-sentinela-design.md`
  with five amendments. Sources polled by code every hour (no API cost), the model called only on new items; hard
  caps of 40 low + 6 max calls a day; acts only in a fictitious "Composite + Sentinela" account at >= 70%; its
  autonomy is decided at the reading (first session on or after 2027-01-08) by proper scoring rules (Brier, log).
- New sources confirmed: IBGE SIDRA (GDP, unemployment, industry), BCB fiscal series (primary result, net and gross
  debt), B3 lending jumps, price/volume anomalies; Tesouro's API was down (its monthly XLSX is optional).
- Decision judge: on 60 hand-labelled CVM filings, Liquid AI's d1.1 beat d1-ultra and Jev 1.14 (93% / 82% / 80%
  accuracy; Brier 0.082 / 0.121 / 0.145; about 1.1 s each) (`docs/juizes-2026-11-05.md`). d1.1 replaces Jev in new
  work; Jev stays only in the pre-registered sentiment log.
- Research closed: the book's 25 formulations (second batch read 05/11) and three optimizers fed the Composite's
  alpha: none passes; half Kelly and robust Markowitz tie the Composite with smaller drawdowns (branches
  `feat/us-lote2`, `feat/lote3-composto`). The Composite stays the real-money plan (R$ 500 on 19/12, closing auction).
- Out of scope by the user: FIIs/FIAGROs; US work shelved (not deleted). No ML/RL/multi-agent model decides weights.

## Session notes (2026-11-06)

- The Sentinela went live: hourly 08:20-19:20 on B3 sessions (task "alepha sentinela"), 0 DeepSeek calls on its first
  day (nothing new after the 00:20 seeding). Its fictitious "Composto + Sentinela" account starts at the 06/11 close
  beside "Composto (base)". Both of its instructions now state the extremely aggressive investor's profile
  (DECISOES 287).
- The live app restarted twice through a comment in `requirements.txt` (the Cloud had the new `app.py` with old `ui`
  modules in memory, DECISOES 281). The backtest title names the Composite alone, with BOVA11 as the yardstick.
- DeepSeek spend: US$ 5 of credit went in 3 days, against small logged costs (about US$ 0.04 per max opinion, under
  US$ 0.001 per scan). The public app's button now needs an access code (`[deepseek] codigo` in the Cloud secrets, set
  by the user and checked live), and the cap is 3 a day (DECISOES 283).
- The real buy of 19/12 is sized by the Sentinela account's balance: no buy under R$ 11,000; R$ 500 from 11,000;
  R$ 750 from 12,500; R$ 1,000 from 13,000. The ticket also prints the agent's alpha against "Composto (base)"
  (DECISOES 284).
- d1.1 (Liquid AI) scores the sentiment log beside Jev, with Jev's own questions, in `sentiment/scores_d1/` (lake
  72df5de). Jev's log is untouched (DECISOES 284).
- Scheduled tasks: paper v1, v2 and v3 had not run since 03/11, because the venv's python cannot write outside the
  project. Every app task now runs the base-Python launcher, which also passes the venv's packages to child processes.
  Paper v2 and v3 refresh the lake through the lake's own base runner on their due days (`core/python_do_lago.py`).
  All eight tasks ended 06/11 with rc 0 (DECISOES 282, 285, 286). The block is not Windows Security, and its cause is
  still unknown.
- The CI's failure e-mails of 06/11 came from `openpyxl` missing from `requirements.txt`. That is fixed, and the
  actions no longer run on Node 20. B3 hours checked 06/11: continuous trading in stocks ends at 17:55, and their
  closing call runs 17:55-18:00.

## Session notes (2026-11-07)

- Research, none adopted (DECISOES 295-300; DSR union N 102): eight papers on new architectures and quantum portfolios
  (`docs/novas-arquiteturas-2026-11-07.md`; KAN the best fit, quantum closed again); KAN no edge; periphery tilt fails;
  fixed BR/BDR mixes (15 names with 1 to 5 US BDRs) beat the Composite in 2021-2026 but fail the gate; HERC with CDaR
  (Raffinot 2018, matched to skfolio to 1e-6) alone gives 11-12% a year and on the Composite's 15 names 15-19%,
  against the Composite's 19-21%: equal weights inside the Composite stay. The four branches are pushed, not merged.
- New ghost account "12/3" in the simulator from 09/11/2026: the Composite with 12 names plus the 3 US BDRs of highest
  12-1 momentum, 1/15 each, net of the 15% tax on BDR gains (`core/mista.py`, DECISOES 299).
- New ghost account "Fermiônica" in the simulator from 12/11/2026: 15 names among every candidate with a Composite score,
  chosen by score and by the log-determinant of their return correlation (theta 0.5), 4 per sector, 1/15 each
  (`core/fermionica_fantasma.py`, `core/fermionica.py`). The best of six determinantal arms, which failed the gate
  (DECISOES 344): followed forward next to the Composite, no claim before 252 sessions (DECISOES 348).
- Refiled CVM statements (DECISOES 349): among the rows received before the decision month, the latest receipt of an
  (issuer, account, period end) counts (`core.valor.documentos_conhecidos`); two of the same day that disagree are
  left out. One decision of 66 changes in two of the three phases; the Composite baseline is 21.4 / 20.0 / 19.0%.
- Research, none adopted (DECISOES 350; DSR union N 109): the Composite and the determinantal and von Neumann
  selections at 5, 10, 20 and 30 names and theta 0.25 to 1, a selection of free size (grand-canonical) and weights
  of largest von Neumann entropy (`core.diversidade.eic_weights`). Fifteen names is the peak; no arm passes the gate
  (`docs/fermionica-grade-2026-11-11.txt`).
- Research, none adopted (DECISOES 352; DSR union N 117): Inovemetria, five innovation and diffusion signals of the
  full universe (`core/inovemetria/`): growth-curve forecast (Bass, logistic, Gompertz), share taken in the sector,
  intangible and gross investment, INPI patents and price convexity. Patents and convexity pass the IC gate (0.024,
  t 3.6; 0.029, t 2.6); no portfolio arm beats the Composite, alone, inside its score or inside the determinantal
  selection with a patent-field kernel (`docs/inovemetria-ic-2026-11-12.txt`, `docs/inovemetria-2026-11-12.txt`).
  A map of the candidates is in `docs/inovemetria-mapa-2026-11-12.csv`.
- Lake nightly (DECISOES 351): B3's open date "31/12/9999" is read as no date (`b3lake.b3events._br_date`); it had stopped
  the gold insert on 11/11. The app shows only what the upkeep publishes after the lake releases its lock.
- The Sentinela: the 06/11 seeding had marked the week's results filings as seen; a one-off run read them on 07/11
  (BBSE3, PSSA3 and LEVE3 `atencao`, no `grave`). It now reads each new Fato Relevante's text and asks a discarded
  answer again once (above). GPT-OSS-120B on Groq was checked on the 60 hand-labelled CVM filings: 67% right against
  d1.1's 93% and DeepSeek Flash's 82%, with the same latency as d1.1; it replaces nothing (`docs/juizes-2026-11-07.md`).

## Session notes (2026-11-08)

- Damodaran agent shelved before any build and out of Aleph Quant (DECISOES 307; branch `feat/damodaran` kept).
- New ghost account "Opções deep OTM" from 09/11 beside "Opções" (DECISOES 308, spec
  `docs/superpowers/specs/2026-11-08-opcoes-deep-otm-design.md`): puts at 85% and calls at 115% of spot, most traded
  strike within a 5% band, size capped by the day's option volume. The first options account had never traded (zero-ask
  division, then a strike too far and too large); both fixed test-first. Reading after the 12th cycle (about Oct 2027).
- The real-buy ticket gave B3's closing call as 16h55; it is 17h55-18h since 03/11 (309). Drex Max fails the 60-filing
  judges check, judges stay d1.1 and d1-ultra (310, `docs/juizes-2026-11-08.md`). No reranker for the Sentinela's packs (311).
- Ibovespa rebalance flows (buy the entrants before index funds must): both pre-registered rules fail (312, N 104).
  R1 mean -0.30% per date (t -0.72); R2 +6.3% per date but t 1.50 under the 2.24 gate on 14 dates. Closed; code and data
  on `feat/ibov-recomposicao`, not merged.
- Answers, no code: no verified quantum-finance alpha at 08/11/2026 (Lean, math-olympiad or Wolfram do not change that);
  the app's game-theory piece is the Sentinela's proper scoring (DECISOES 271); what to add at R$30k+ is capacity
  reading of the Composite and a monthly-sales < R$20k rule in the ticket (offered, not asked).

## Session notes (2026-11-09/10)

- Jev (TypeSafe) is back in-session with no config change, answering `typesafe/jev-1.14-20261014` through the
  OpenRouter key. The owed gate of DECISOES 320 was re-run: per file it was 7/7 `escalate` with 0 contradicted, and the
  no-look-ahead claim read "unsupported" only because each file was judged alone; the whole change in ONE `files` call
  (7 code files, ~82k chars, no truncation) gave composite 0.645, `safe_to_apply` 0.20, `escalate`, limiting rubric
  `test_gap`, and both claims verified (no look-ahead at 0.61, 18 tests at 1.0), 0 contradicted (DECISOES 321).
- BDR facts corrected (DECISOES 322): `ativos_eua.ATIVOS_EUA` has **32** curated US BDRs; the app panel
  (`lago/precos_diarios.parquet`) has **923** BDR columns; the research export `data/b3lake/universe_prices.parquet`
  (used by the QML arm) has **none**. B3 BDR prices exist from **03/09/2013** for 16 of the 32; all 32 only from
  21/06/2022. The earlier "no BDRs" answer was true of one file and false of the project's data.
- G2 "Composite Quantum" was pre-registered and committed before any number (`87d19bc`), then run (`f198310`;
  DECISOES 323). Gate: t(alpha vs Composite) >= 2 and the 95% interval above zero in all three phases.
  Daily signal lookup: t **1.84 / 0.87 / 0.94** (gains 1.4% in phase 0, loses in 21/42 - the staleness hypothesis is
  dead). BDR carrier: t **1.76 / 1.48 / 1.58**; best risk profile measured (MDD -33 to -35% vs -42%, vol 16.3-16.7% vs
  19.2-19.6%) and it beats the Composite on CAGR and Sharpe in all three phases, but no significance. Both fail.
- Why t is hard: t = IR x sqrt(years). At 16.7 years, t = 2 needs IR >= 0.49 against 0.44 / 0.26 / 0.37 today;
  MinTRL is 65-1,215 years. Reducing volatility does not move t (alpha and its standard error scale together), and
  rebalancing frequency does not either (the alpha regression already uses daily returns). Only a better information
  ratio or more time does.
- The Stage-2 forward holdout is **open**: R$ 10,000 fictitious, rules frozen, 63-session rebalance, 9 names bought at
  the 09/11 close, mark R$ 9,990.00 after cost (DECISOES 324; `simulador/qml_holdout/`). Ledger helpers tested
  (7 tests); the bug that blocked it - prices passed where the engine hands the strategy a filtered **return** window -
  is fixed. Judgement in 12 months, or earlier only if the drawdown leaves the -38% envelope.
- On the 09/11 rise: 06/11 R$ 9,978.59 -> 09/11 R$ 10,131.62 (+1.53%) is the **Composite alone**. "Composto (base)" and
  "Composto + Sentinela" are identical because no Sentinela sale has been accepted yet.
- Tooling, for the next session: the QML tests and runs need `ai-hedge-fund/.venv-qpo` (torch 2.14.1+cpu,
  pennylane 0.45.1); the project `.venv` has no torch. Work in `../po-qml` (the `data/` folder is gitignored and absent
  there, so pass `--lago <main repo>/data/b3lake`). No GPU is needed: the circuit is 6 qubits and 45 parameters, and the
  daily run took 879 s on the CPU.

## Session notes (2026-11-10)

- **The DRL replication is closed, and it loses to the Composite** (DECISOES 330-335). Three papers split by method and
  weighted 40/40/20 after the user's revision; PPO/A2C/DDPG/TD3/SAC run on the full eligible B3 universe, 2010-2026,
  monthly, 10 bps. PPO + Differential Sharpe Ratio: CAGR 8.15%, Sharpe 0.43. PPO + DSR + drawdown term: 8.05%, 0.43.
  Equal weight over the whole universe: 9.75%, 0.52. Minimum variance: 9.51%, **0.91**, MDD -34.4%. The Composite:
  20.57%, 1.20. **The universe was never the constraint - the signal is.** The drawdown-penalty term changed nothing,
  and the only remaining DRL role (a construction layer over an existing signal) dies on the same ~200-decision
  arithmetic that closed the QML line.
- **A look-ahead bug was found and fixed** (DECISOES 335). `drl/ambiente.py` matrix mode fed the observation the return
  the next action earns (`[:, inicio: t + 1]` instead of `[:, inicio: t]`), caught by a DDPG cell at Sharpe 4.11 with
  300x turnover. Fixed, with a test that pins it. Two other bugs on the way: DDPG/TD3 rejected `log_std_init`
  (deterministic policies have a `qf` head), and the baselines were reset to 1.0 every January - a sawtooth that
  fabricated a crash each new year and produced the absurd MVO -59% / equal-weight -67% figures, which were retracted.
- **Volatility targeting is the one genuine improvement** (DECISOES 336). New `core/vol_target.py`, 10 tests. Exposure
  = target / realised 21-session volatility, measured to the previous session, capped at 1 (never levered), idle capital
  in the CDI, 10 bps on what moves. Target 20%: **CAGR 20.30%, Sharpe 1.20, MDD -21.4%** against the plain Composite's
  19.91% / 1.04 / -41.5%, at 96% mean exposure. Monotone across three targets and two windows. **Not live**: it was
  measured on the saved curve assuming daily adjustment, with no tax; the engine run with real rebalancing and a FIFO
  cost basis is owed, and it is an order-generation change, so it needs the advisor review.
- **LLM signal layer planned, not built** (DECISOES 338). `LLM-SIGNAL-LAYER-PLAN.md` and
  `docs/superpowers/plans/2026-11-10-llm-signal-layer.md`: reranker as triage, extraction as typed events, a
  pre-registered veto gate (drop a ticker with a `material` + `baixa` event at confidence >= 0.7 in the last 20
  sessions). Zero cost, free tiers only. Research correction: the free OpenRouter model is Nemotron **3** Ultra
  550B-A55B; there is no "Nemotron 3.5 Ultra".
- **A local 9B was benchmarked and discarded** (DECISOES 337). Qwen3.5-9B Q4_K_M on this laptop: **1.66 tokens/s** at 4
  threads, 0.84 at 8 (hyperthreading halves it). The same extraction takes 2 min 35 s locally against 1.3-2.2 s on the
  DeepSeek API - about **90x**. The model was deleted; the `llama.cpp` runtime stays in `D:\modelos` for future tests.
  This BIOS has no UMA frame buffer setting: the 2.1 GB reserved for the iGPU cannot be recovered.
