# MARS on the B3 — design and pre-registration

_2026-11-08. Written and committed before any MARS number exists. User's request: run the method of "MARS: A
Meta-Adaptive Reinforcement Learning Framework for Risk-Aware Multi-Agent Portfolio Management" (Chen, Li, Wang,
AAAI-26; `MFG_MARL/15173-AAAI26.ChenJ-ML.pdf`) on the B3 for the aggressive investor, alone or combined with our
strategy, and see whether its reported results (DJI 2024 Sharpe 2.84, DJI 2022 -0.86% with MDD -16.8%) can be
reached. Heavy training may run on Kaggle or Modal. Branch `feat/mars` (from `feat/herc` `20bd5c1`, which holds the
213 harness). Sections 1 to 7 are fixed from the commit on; any change before the reading is an addendum committed
before any number._

## 1. What the paper does and what it does not report

MARS = an ensemble of N = 10 DDPG agents (actor, critic and a safety critic each), agent i with a risk threshold
theta_i and penalty lambda_i, plus a meta-adaptive controller (MAC) that maps the state to softmax weights over the
agents; the executed action is the weighted mean of the agents' actions passed through a rule overlay (no short,
20% position cap, cash buffer). State: cash, holdings and, per asset, price + MACD, RSI, CCI, ADX. Reward: return
minus transaction cost minus 0.5 x 30-day volatility minus 2.0 x 30-day drawdown. The safety critic regresses an
environment risk score C_env in [0, 1] = 0.4 simulated volatility + 0.3 concentration + 0.3 leverage; the actor
minimizes -Q + lambda_i ReLU(C - theta_i). The MAC minimizes -(mean Qbar / std Qbar - 0.5 mean Cbar) over a
batch. MLPs 256-128-64, gamma 0.99, seed 42, 50 stocks, two test years (2022, 2024) after 5 training and 1
validation year, Yahoo data, hyperparameters "tuned on the validation set".

Not reported, so filled here (section 4): episode count, learning rates, batch, exploration, max trade size, cash
buffer size, transaction cost, indicator windows, units of the reward penalties, how the three risk parts are scaled,
which theta goes with which lambda, MAC training schedule, and how the test years chain. No code was released.

Reading of the evidence, written now: one seed, one year per test, a 50-stock list picked today (survivorship), the
MARS-Div5 and MARS-Div15 ablations both below N = 10 (a sign of selection on the test year), and on HSI 2024 MARS
made 16.2% against the index's 24.5%. The paper's numbers are not a prior for the B3.

## 2. What is already known here

The binding limit on the B3 is the expected-return signal, not the solver (DECISOES 213, 265, 267-268, 296). Daily
and monthly refits of learned models churn the picks (KAN-15 and VQC-15 turnover 4.4 vs 3.0) and lose to the
Composite even with a comparable IC. The Composite makes 21.4/19.9/19.1% a year (phases 0/21/42, 2010 to 16/10/2026)
with MDD -41.5/-42.2/-42.8%. B3 regimes are not the paper's: BOVA11 made +5.1% in 2022 and -10.1% in 2024.

**Prior, written now.** MARS-50 passes the gate: about 2%. MARS-50+score passes: about 4%. Most likely outcome: both
arms below the Composite's CAGR (MARS-50 between 5% and 14% a year), with lower volatility than the Composite,
because the reward's drawdown penalty and the cash buffer push towards cash and CDI.

## 3. Arms (two configurations, two rows in `docs/tentativas.csv`)

| Arm | Definition |
|---|---|
| `MARS-50` | The paper's method on the year's 50 most traded B3 names (section 4), its features (price, MACD, RSI, CCI, ADX) |
| `MARS-50+score` | The same with one more feature per asset: the Composite's score (attempt 51's, `_value_score(..., com_crescimento=True)`) of the session's month, as a percentile among the 50 (missing = 0.5). The combination with our strategy: the RL policy sees our alpha signal |

Descriptive only, not trials: `MARS-Static` (the same trained agents of `MARS-50`, uniform weights instead of the
MAC, deployment only); the MAC's mean weights per year on the conservative (agents 0-2), neutral (3-6) and
aggressive (7-9) groups; the paper's table (CR, AR, SR, AVol, MDD) for the B3 test years 2022 and 2024.

## 4. Method, each choice fixed here, one configuration, no search

**Data (frozen).** Prices: `docs/dados/2026-10-22/universe_prices.parquet` (total-return index, the engine's), cut at
16/10/2026; universe: `universe_monthly.parquet` of the same folder; CDI: `docs/dados/2026-10-21-macro/
cdi_daily.parquet`. Two inputs are frozen now into `docs/dados/2026-11-08-mars/` with SHA256SUMS: daily traded value
(`volume_brl`, lake `exports/daily_liquid_panel.parquet`) and the daily high/close and low/close ratios of each
ticker (lake `bronze/cotahist`, standard lot, cash market); high and low of the total-return index = index x ratio
(ratios are same-day, so the adjustment cancels). A missing ratio is 1.

**Walk-forward.** Test years Y = 2010 ... 2026 (2026 to 16/10). Training = the five calendar years before Y-1
(for 2010 only 2005-2008, the data's start, after a 60-session warm-up), validation = Y-1, test = Y.

**Universe of year Y.** Names in the point-in-time universe of January Y, with a total-return price on at least 95%
of the training-plus-validation sessions, ranked by median daily traded value over the 252 sessions before Y; the top
50 (D = 50, the paper's). Fixed by data, not by any other strategy. A name whose price ends during the test is sold
at its last price the session before; a name not listed on a day cannot be bought.

**Features**, per asset, from the total-return index and its high/low: log(P_t / P_first training session), MACD
(12, 26) / P_t, RSI(30), CCI(30), ADX(30) (FinRL's windows; the paper builds on FinRL). Each feature is standardized
with the mean and standard deviation over the training window (all 50 assets pooled) and clipped to [-5, 5]. State
= [cash / V, holding_i / V for the 50, features] (D x 6 + 1 = 301; 351 with the score).

**Execution (the paper's overlay).** Action A in [-1, 1]^50 = target change per asset, scaled to A_i x 10% of the
portfolio value V. Rules, in order: no short (a sale is at most the holding), sales first, a buy stops at 20% of V
in that name, buys are scaled down pro rata so that cash after buys and costs stays >= 2% of V. Cost 10 bps per side
(the engine's CFG_BT). Cash earns the daily CDI (a deviation that helps MARS: cash at 0% would be unrealistic in
Brazil). Training and validation episodes start in cash; the test years chain (holdings carry over; names leaving
the 50 are sold on the first session of the new year).

**Reward.** R_t = (V_{t+1} - V_t)/V_t - cost_t / V_t - (0.5 sigma_30 + 2.0 DD_30), sigma_30 = standard deviation of
the last 30 daily portfolio returns (daily units), DD_30 = the maximum drawdown inside the last 30 values (a
positive fraction). V is already net of costs; the paper's formula subtracts the cost again and so does this one.

**Risk score.** For the aggregated action before the overlay (only the no-short clip), with post-trade weights w:
volatility = min(1, sqrt(252 w' S w) / 0.5), S = covariance of the last 60 daily returns; concentration =
clip((sum w_i^2 - 1/D) / (1 - 1/D), 0, 1); leverage = clip(sum w_i - 1, 0, 1). C_env = 0.4 vol + 0.3 conc + 0.3 lev.
The safety critic is trained on (s_t, A_t) with this target (the executed A'_t would never show leverage); the
critic on (s_t, A'_t, R_t, s_{t+1}), as Algorithm 1.

**Agents.** theta_i = linspace(0.10, 0.55, 10); lambda_i = linspace(5.5, 1.0, 10) (agent 0 the most conservative:
lowest tolerance with the largest penalty; the paper gives both ranges but not the pairing). DDPG: MLPs 256-128-64
with ReLU, tanh output for the actor; Adam, actor 1e-4, critic and safety critic 1e-3; soft target update tau 0.005;
gamma 0.99; batch 128 sampled independently per agent from the shared transitions; one update per agent per session
after the first 1,000 sessions; Gaussian exploration noise sd 0.1 on the aggregated action; 30 episodes over the
training window. No terminal flag (the episode end is a time limit).

**MAC.** MLP 256-128-64 to 10 logits, softmax. After every episode, 200 Adam steps (lr 1e-4, batch 256) on its
buffer of (s_t, Q_i(s_t, a_i), C_i(s_t, a_i)), loss -(mean Qbar / (std Qbar + 1e-6) - 0.5 mean Cbar), lambda_meta
0.5.

**Validation.** After each episode, a deterministic pass (no noise) over the validation year from cash; the snapshot
of all networks with the best validation Sharpe is the one tested (the only use of validation; nothing is tuned).

**Seeds.** 42 (the paper's), 0 and 1. The arm's curve is one third of the capital in each seed's run from the first
test session (an equal-capital split, not rebalanced). Each seed's curve is also printed.

## 5. Reading and gate

Engine, data and checks of the 213 reading (`kan_avaliar_b3.py`): the Conservadora must reproduce 17.41/16.41/16.90%
and the Composite 21.4/19.9/19.1% before a MARS arm is read. Each arm's daily value curve is aligned to each phase's
dates (it starts at the phase's first session, normalized) and read with `tabela_do_periodo` and
`psr_da_diferenca`. **Gate, every phase, all >= 0.95:** P[> Composite], P[> Conservadora], P[> BOVA11], DSR.

**N.** Two configurations, two rows. The DSR uses the conservative union of every branch's trials: 104 before this
branch (`feat/ibov-recomposicao`), so **N = 106**, printed beside the branch count.

## 6. Compute

The training is one self-contained file, `qpo_kaggle/mars.py` (torch; the 10 agents as one batched ensemble), reading
one data package `mars_pacote.npz` built locally by `scripts/mars_pacote.py`. 2 arms x 3 seeds x 17 years = 102
trainings. They run on Kaggle (CPU or GPU kernels, the qpo-solve pattern), not on the laptop; Modal is the fallback.
The outputs (daily values, MAC weights, validation picks) come back as one npz per arm and seed, hashed.

## 7. Output

`docs/mars-2026-11-08.txt` (tables per phase, year table, gate, descriptive extras) and one DECISOES entry. Nothing
goes to the app unless an arm passes, and then only as a forward paper log after the user's go.

## Addendum 1 (08/11, before any test number; the first Modal run was stopped and discarded)

Two faults surfaced in the first training run's logs (validation diagnostics only; no test year was chained, no
curve was read):

- **Validation Sharpe.** Section 4 said "the best validation Sharpe" without saying over what. With cash earning
  the CDI (a deviation of this spec, not the paper's), a policy that stays in cash has an almost constant daily
  return, so its raw Sharpe exploded (135-391 in the logs) and the checkpoint picked "all cash". The validation
  Sharpe is now the project's usual one: the daily excess over the CDI, annualized, and 0 when the excess does not
  move. In the paper cash earns nothing, so "all cash" scores 0 there too: this restores the paper's intent.
- **Overlay.** With cash already below the 2% buffer (holdings grew) and no buy, the pro-rata rule divided 0 by 0.
  Buys now shrink only when there are buys.

Both fixed test first (`tests/test_mars.py`). Every training is rerun from scratch; the first run's snapshots were
deleted from the Volume.
