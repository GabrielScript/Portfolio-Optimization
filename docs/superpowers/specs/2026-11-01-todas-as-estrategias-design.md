# Every strategy, US and the books' new ones

Pre-registered on 01/11/2026, before any arm below was computed on real data. Only synthetic timing spikes ran before this commit.
The user's request: test every strategy used in the project, plus those of López de Prado's book, Palomar's book and the HRP paper.
Extends `2026-10-31-us-estrategias-design.md` and `2026-11-01-qpo-hibrido-design.md`.

## 1. What is run, on which market

Each configuration is read once per market. A strategy read on the B3 is not read there again.

- **US:** every recorded attempt that transfers (section 2), plus the new ones (section 3).
- **B3:** only the new ones (section 3), because the others already have their reading.

Every recorded attempt has a line in section 2 or an exclusion with its reason, so none is dropped in silence.

## 2. Recorded attempts on the US (`docs/tentativas.csv`, N + 0)

Same configuration on another universe, so N does not grow, as the header of that file says. BOVA11 becomes SPY, IMA-B becomes AGG
and the CDI becomes the 1-month T-bill of Ken French's file.

| Rows | Configuration |
|---|---|
| 7, 8 and the control | Kelly λ = 1, 4 and 2: 20% per name, at most 10 names, 40% volatility |
| 9, 10 | Max Sharpe, minimum variance, same caps |
| 11, 31, 32 | HRP default; HRP with the paper's distance; HRP-tree, ward |
| 12 | 1/N over the whole universe |
| 13, 14 | Minimum-variance core 70/30 and 50/50, Kelly λ = 2 satellite |
| 15, 16 | SPY core 70/30 and 50/50, Kelly λ = 2 satellite |
| 17, 18, 19 | Kelly λ = 2, minimum variance and the 70/30 core with the fixed rules |
| 20 | SPY 70% + Kelly λ = 2 with rules and the 30% relative stop |
| 1 | Resampled Kelly λ = 2: B = 100, blocks of 1 month (the B3's 21 sessions), seed 0 |
| 30 | Kelly λ = 2 with Black-Litterman views (Conservadora and value), rules |
| 34, 35 | SPY 50% + AGG 50%; SPY 50% + T-bill 50%, fixed weights, back to target every rebalance |
| 42, 44, 45 | QPO-exact on the top 20; Conservadora-MS; Topk-MS |
| 47, 48, 49 | QUBO-universo, -valor and -crescimento by annealing |
| 50 | Vencedoras-crescimento |

**Already read on the US, reference only, not read again:** 22, 24, 25, 40, 41, 51 (31/10 test), 54 to 57 (hybrid QPO).

**Excluded, and why:**

| Rows | Why |
|---|---|
| 2 | A replica of an old optimizer, not a strategy |
| 3, 4, 5, 6, 21 | Training-window variants. The US panel has 10 years: 36 months is the base. 60 months would lose the 2020 crash, and 120 is the whole panel |
| 23 | BOVA11 + IVVB11 mixes the local index with the S&P in reais: no US analogue |
| 26, 27, 28, 29 | EGARCH volatility target: a daily model, and the dead names have no daily prices |
| 33 | Below the liquidity floor, Amihud from daily data: not on the panel. The 31/10 floor sensitivity covers that direction |
| 36, 37, 38 | Vetoes built on Brazilian series (IDA-DI, NTN-B, CDI). A US version would be a different configuration, so a separate attempt |
| 39 | The no-trade zone is built on the B3's costs and income tax |
| 43 | QAOA at best equals the exact row 42; the solver bench covers it |
| 46 | Random-15 is a control for rows 42 and 43, not a strategy |
| 52 | Needs Brazilian fund holdings (CDA) |
| 53 | Rotation between the B3 and the US sleeves: not a US strategy |

## 3. New configurations (N + 11: 57 -> 68)

**Cost, stated before the result:** 11 more trials raise the DSR bar for every strategy, the Composite included.

Fixed caps for all: 20% per name, 30% per sector, 40% volatility where it applies. Parameters are the books' or fixed here, and none
is tuned afterwards.

| # | Arm | Source | Definition |
|---|---|---|---|
| 58 | HRP-projetado | the app's "HRP (hierárquico, projetado nos tetos)" | `projetar_pesos` of default HRP: at most 10 names, the caps |
| 59 | IVarP | López de Prado ch. 16, Palomar ch. 6 | w proportional to 1/variance |
| 60 | IVolP | Palomar ch. 6 | w proportional to 1/volatility |
| 61 | MDivP | Palomar ch. 6 | Max Sharpe with mu = the names' volatilities, rf 0: the most diversified |
| 62 | MDecP | Palomar ch. 6 | Minimum variance on the correlation matrix, then divided by the volatilities |
| 63 | RPP | Palomar ch. 11 | Equal risk contribution, long-only, by cyclical coordinate descent |
| 64 | HERC | Palomar ch. 12 | Ward on the correlation distance, 12 clusters, inside 1/N, across equal risk contribution. Deviation: 12 clusters instead of the gap statistic, which on 36 rows is noise |
| 65 | Mean-CVaR | Palomar ch. 10 | max w'mu - CVaR(0.90) per period, Rockafellar-Uryasev LP on the training rows. 0.90 because 0.95 on 36 rows would rest on 2 scenarios |
| 66 | Mean-CDaR | Palomar ch. 10 | max w'mu - CDaR(0.90) per period, uncompounded drawdowns, LP |
| 67 | MVSK | Palomar ch. 9 | CRRA weights with gamma = 1 (1, 1/2, 1/3, 1/4), moments from the scenarios, sequential convex steps from 1/N, at most 20 |
| 68 | Robust-MV | Palomar ch. 14 | max w'mu - kappa \|\|S^(1/2) w\|\| - (1/2) w'Sigma w, S = Sigma/T, kappa = 1, an SOCP |

Arms 59, 60, 62, 63 and 64 hold the whole universe, with weights cut to the caps by `ajustar_aos_tetos`. Arms 61, 65, 66, 67 and 68
hold at most 10 names by the same two-stage rule as Kelly.

**From the books, not run:**

- **López de Prado:** a methods book. PSR, DSR and purging are already this test's machinery; labels and meta-labels were tried on 21/10
  (row 38); its bet sizing and microstructure need intraday data; chapter 16 is HRP, run above with its IVarP baseline.
- **Palomar ch. 13, index tracking:** no alpha by construction.
- **Palomar ch. 15, pairs trading:** long-short, and the plan is long-only and unlevered.
- **Palomar ch. 16, deep learning:** `Artigos_PO/TRIAGEM.md` already rejected what it covers, and it does not fit the machine.

## 4. Estimation, the limit of this test

**US: 36 monthly rows against about 2,960 names.** No daily route exists: of the 599 companies that stopped, stockanalysis has daily
prices for 297 and Yahoo for 10, so daily estimation would favour survivors. This is a finding to read with the result:

- Ledoit-Wolf shrinks heavily (about 0.58 on the synthetic spike), so the covariance is mostly its target.
- **Jorion's shrinkage turns itself off** when `n_obs <= n_ativos + 2` (`core.estimation.shrinkage_jorion`). Kelly and max Sharpe on the
  US therefore use raw 36-month means, unlike their B3 readings.
- CVaR and CDaR at 0.90 rest on about 4 of the 36 scenarios.

**Monthly moments:** mu = 12 x mean and Sigma = 12 x Ledoit-Wolf from the monthly returns. `core.us_teste.no_grid_mensal` is not used
where mu matters: its scale is right for the covariance but inflates the mean by sqrt(21).

**Solver:** the low-rank form of `core/baixo_posto.py`, held by tests to the dense optimizer on small problems. Same optimum, so it
is not a new configuration.

**B3:** `CFG_BT`, 1,260 sessions, mu and Sigma as the engine estimates them, Jorion on, as every other B3 reading.

## 5. Engines, gates and order

- **US:** `core.us_teste`, monthly grid, every 3 months, 36 months of training, 10 bps per side, phases 0/1/2, on the exports frozen for
  the hybrid QPO (`D:/b3-datalake/exports/qml/us/exports`). The Composite and the Conservadora must reproduce the batch's
  `us-estrategias-2026-11-01.txt` before any arm is read.
- **B3:** `walk_forward_dinamico`, phases 0/21/42, same frozen folders as the hybrid QPO. The Conservadora must reproduce 17.41/16.41/
  16.90% and the Composite 21.4/19.9/19.1%.

**Gate, each arm, every phase, all >= 0.95.** B3: P[> Composite], P[> Conservadora], P[> BOVA11], DSR. US: P[> Composite],
P[> Conservadora], P[> SPY], DSR. A pass makes the arm a candidate for a forward paper log, nothing more. Nothing is tuned after a
reading. CAGR, volatility, drawdown, turnover and the year tables with CDI and Selic are reported, never gates.

**Order:** the B3 reading of section 3 after the solver bench; the US reading after the hybrid QPO's US reading, one heavy job at a time.

## 6. Prior, written before any reading

No arm passes against the Composite on either market. On the US none passes against SPY, as in the 31/10 test. Minimum variance, IVarP,
RPP and HRP will show the best risk measures and the lower returns. Kelly and max Sharpe on the US will do worst among the optimizers,
because raw 36-month means pick past winners. The books' own text expects CVaR, CDaR and HRP not to beat 1/N or GMVP consistently.

## 7. Output

`docs/todas-as-estrategias-2026-11-01.txt`, derived files in `docs/dados/`, a screen table, and a DECISOES entry. No real money moves
from this table. The user decides.
