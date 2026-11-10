# DRL v2: the three papers on the corrected simulator (pre-registered 2026-11-10)

Written and committed before any v2 agent number exists. Code: branch `refactor/drl-review` (commit fbfb310 and
after); review of the first version: `docs/revisao-drl-2026-11-10.md`. Smoke runs used to time the cells
(`drl/saidas/smoke-modal`, `drl/saidas/timing`, 3,024-30,240 timesteps) are not readings and are not reported.

## Setup (fixed)

- Tape: lake exports cut at 2026-11-06 (`universe_prices`, `universe_monthly`, `cdi_daily`); eligible = on the lake's
  list for the month the return is earned in, with 252 unbroken closes. Names of a window = those eligible on the first
  session of its test year (~120-220).
- Environment: compact observation (6 numbers per name: weight, 1/5/21/252-session returns, 63-session vol; the cash
  row carries the market vol20 and vol20/vol60, expanding z-scores), one decision every 21 sessions, 10 bps per unit
  traded, cash earns the CDI, softmax over the tradeable names and cash.
- Protocol (P1): test years 2012-2026, five years of training, the year before the test as validation; one agent per
  seed, the seed with the highest validation reward is the only one that sees the test year; the test years are
  stitched into one curve.
- Compute: Modal, 4 CPU and 8 GiB per cell, within the workspace's free monthly credit (about US$ 15 left on
  2026-11-10); budget for this run US$ 10, checked with `modal billing summary` between the two batches.

## Arms

| paper | strategy | seeds | timesteps (decisions) |
|---|---|---|---|
| P1 | `ppo/dsr` (Table 1 hyperparameters) | 0-4 | 150,000 |
| P2 | `ppo/dsr_drawdown` | 0-4 | 150,000 |
| P2 | `a2c/dsr_drawdown` | 0-4 | 150,000 |
| P2 | `ddpg/dsr_drawdown` | 0-2 | 30,000 |
| P3 | `sac/value` | 0-2 | 30,000 |
| P3 | `td3/value` | 0-2 | 30,000 |

Off-policy agents get fewer seeds and timesteps because a gradient step per transition costs about twenty times a PPO
step on CPU (timed: 3,000 SAC steps ~60 s against 30,240 PPO steps ~35 s); at the on-policy budget each would cost
about US$ 16. 30,000 decisions are about 500 passes over a five-year training tape. This is a limitation of the
reading, written here before it.

References in the same simulator, already run (`drl/saidas/2026-11-10-rules`): equal weight, minimum variance and
maximum Sharpe (Ledoit-Wolf, 252 sessions). The Composite enters as a reference curve from the house engine.

## Gates (fixed here)

- An agent passes when its stitched 2012-2026 curve has a higher CAGR and a higher Sharpe over the CDI than the best of
  the three rules **and** than the Composite reference over the same sessions.
- Descriptive: per-year returns, turnover, the validation reward of every seed, and the spread between seeds.
- Six new trials for the DSR count. A pass is a candidate only: any use needs its own forward holdout.
