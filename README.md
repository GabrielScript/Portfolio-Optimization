# 📊 Alepha Quant — Alocação quantitativa em ações da B3

> **Autor:** Gabriel Estrela Lopes

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
- Options data in the lake (research only: the fund's rules bar trading derivatives): implied volatility and greeks
  of every traded B3 option since 1994 computed from COTAHIST, B3's daily Price Report from 2018 (futures, options
  on futures, open interest) and a daily snapshot of opcoes.net.br chains to cross-check the greeks.
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
