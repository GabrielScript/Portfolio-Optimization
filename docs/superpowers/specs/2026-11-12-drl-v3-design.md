# DRL v3: the second review and four corrected arms (pre-registered 2026-11-12)

Written and committed before any v3 number exists. Branch `refactor/drl-v3`, from `refactor/drl-review` (773a436).
The first review (`docs/revisao-drl-2026-11-10.md`) fixed the simulator; this one read `drl/` again line by line, plus
`qpo_kaggle/mars.py`, `scripts/mars_pacote.py`, `qpo_kaggle/kan.py` and the training loop of `qpo_kaggle/qml_sinal.py`.
Smoke runs used to time the cells are not readings and are not reported.

## What the second review found in `drl/`

No look-ahead was found in the observation, the market features, the rules or the cost. The defects are in what the
agent is asked to learn and how:

1. **The reward's optimum is 100% cash.** The differential Sharpe ratio is fed the raw portfolio return while cash
   earns the CDI. Cash then has a return of about 10% a year at almost no volatility, so the raw-return Sharpe is
   maximized by holding nothing else. The gate measures the Sharpe over the CDI, where that book scores 0: the agent
   was trained on a different target from the one it was judged by.
2. **Cold start of the reward.** `(A, B)` start at 0, so the reward of session `k` of an episode is about
   `1 / sqrt(k / 252)` times its steady-state size: the first month weighs roughly six times a later one, in training
   and in the validation reward that picks the seed.
3. **One path, replayed.** A training episode always starts on the same row and runs the five years: about 60
   decisions, replayed about 2,500 times in 150,000 timesteps.
4. **The end of the tape is a terminal state.** `step` returns `terminated=True` at the last row, so the value of the
   last state is taken as 0 instead of bootstrapped (a truncation, in Gymnasium's terms).
5. **A policy that cannot leave equal weight.** One flat MLP maps about 900 inputs to about 151 logits; each logit
   has its own weights and sees about 60 distinct states. With a Gaussian of standard deviation 0.37 on the logits,
   the softmax stays within about 40% of 1/n. "PPO collapses to equal weight" was the design, not a finding.
6. **Raw inputs.** Returns of 0.005 and of 0.5 enter a tanh network unscaled, and the cross-section is never
   standardized.
7. **Survivors only in training.** The names of a window are those eligible on the first test session, and the five
   training years use the same columns: a name that died inside them is never seen.
8. **The report lacks alpha, beta and the t of alpha.**

Not defects, kept: each test year starts from cash (agents and rules alike); P1's seed rule; `max_sharpe` on raw
sample means (the paper's comparator).

## What it found elsewhere (reviewed, not rerun)

- `mars.py`: four choices weaken its learning: (a) the critic is trained on the executed action and differentiated
  at the raw one, which the overlay cuts by a large factor whenever cash binds; (b) the reward subtracts the cost a
  second time (the value already nets it); (c) the meta controller is trained on Q and C stored when each transition
  was collected, most of them from untrained critics; (d) a name is sold whole on its last close, known only
  afterwards. All four are written in MARS's own pre-registration (`2026-11-08-mars-design.md`, section 4) as the
  paper's algorithm: the code does what was registered, and none touches the measured curve's accounting.
- `kan.py`, `qml_sinal.prever_torch`: no defect found.

## v3 environment (all four arms)

- Tape, timing, cost and eligibility as v2: lake exports cut at 2026-11-06, one decision every 21 sessions, 10 bps
  per unit traded, cash earns the CDI, 252 unbroken closes.
- **Fully invested.** Under a Sharpe over the CDI the cash share is not identified (scaling the risky book scales
  mean and deviation alike), so the agents hold no cash unless nothing is tradeable, as the three rules do.
- **Reward `dsr_cdi`:** the differential Sharpe ratio of the session return in excess of the CDI; `(A, B)` start at
  the mean and mean square of the equal-weighted eligible market's excess return over the 252 sessions before the
  episode's first row.
- **Observation `cross`:** per name the weight held and the 1, 5, 21 and 252-session returns and the 63-session
  volatility, each standardized across the names tradeable at the decision and clipped to +/- 3; a name that is not
  tradeable carries zeros; the cash row carries the cash weight and the two market indicators.
- **Training episodes:** 12 decisions from cash, the first row drawn uniformly from the training stretch; the end of
  an episode is a truncation.
- **Training universe:** the names eligible on any session of the training or validation years or on the first test
  session. The validation and test environments trade only the names eligible on their own first session, so the
  test book is drawn from the same set as the rules'.

## Arms (four new trials, N 117 -> 121)

| arm | policy | training | seeds |
|---|---|---|---|
| `ppo/dsr_cdi` | P1's flat MLP, 64-64 tanh | PPO, P1 Table 1, 150,000 decisions | 0-4, P1's rule: the seed with the highest validation reward trades the test year |
| `ppo_shared/dsr_cdi` | one 64-64 tanh network shared by every name (input: the name's row and the cash row; output: its logit); value from the mean of the names' hidden vectors | the same | the same |
| `direct/sharpe` | the same shared network | gradient ascent through the simulator (it is differentiable in the weights): 300 Adam steps at 1e-3, 32 episodes each, loss = minus the mean annualized Sharpe over the CDI of the episodes; every 10 steps the validation year is run from cash and the best validation Sharpe is kept | 0-4, the test book is the mean of the five seeds' weights |
| `direct_score/sharpe` | the same, plus the Composite's score of the month (standardized the same way) as a seventh input | the same | the same |

The Composite score is the house's (`scripts.qubo_universo_avaliar._value_score(com_crescimento=True)`), exported per
month and name and frozen in `docs/dados/2026-11-12-drl-v3/`; the month's score is known before its first session.
Months without fundamentals carry no score (zeros after standardization): a limitation of the first windows.

## Gates (fixed here, v2's)

- An arm passes when its stitched 2012-2026 curve has a higher CAGR **and** a higher Sharpe over the CDI than the
  best of the three rules **and** than the Composite reference over the same sessions.
- The Composite reference is the current baseline curve (phase 0, after the refiled-statements fix,
  `docs/dados/inovemetria-curvas-fase0-2026-11-12.parquet` on `feat/inovemetria`), frozen with BOVA11 in the same
  folder; every row ends on its last session (2026-10-16). It comes from the house engine, another simulator.
- Reported for every row: CAGR, volatility, Sharpe and Sortino over the CDI, drawdown, Calmar, turnover, and the
  house CAPM against BOVA11 (daily excess returns, alpha x 252, beta, Newey-West t of alpha); per-year returns and,
  for the PPO arms, each seed's validation reward.
- A pass is a candidate only: any use needs its own forward holdout.

## Compute

PPO arms on Modal (4 CPU, 8 GiB per cell), inside the month's free credit (US$ 9.19 left on 2026-11-12); budget
US$ 4. Direct arms on the laptop.

## Amendment before any number (2026-11-12, after the timing smoke runs)

Timed on the laptop, no return read: a direct step takes 0.25 s (285 names x 32 episodes x 12 decisions), so the two
direct arms would take about five hours there; one PPO rollout and update of the shared policy takes 85 s, so
150,000 decisions x 5 seeds x 15 windows would cost several times the free credit.

- `ppo_shared/dsr_cdi`: **30,240 decisions (ten rollouts) and seeds 0-2**, the cut v2 made for its off-policy agents
  for the same reason. A limitation of that arm's reading, written here before it.
- All four arms run on Modal. Budget US$ 7 of the US$ 9.19, the direct arms first, `modal billing summary` checked
  before the PPO arms.
- The timing run also counted the names (no return involved): the 2012 window trades 143 names and its training
  universe has 227; 2024, 206 and 285. Defect 7 left out about a third of the names of the training years.
- Today's lake export cut at 2026-11-06 reproduces the v2 rule cells of 2012, 2019 and 2026 to the last digit, so
  the three rules are not rerun: their cells (`drl/saidas/2026-11-10-rules`) are read with the new arms.

## Second amendment, while the PPO arms were training (2026-11-12, no PPO number read)

A probe of the review found a defect of this version's own warm start: in 2006 no name has its 252 closes yet for
the first sessions, the equal-weighted "market" behind the first `(A, B)` earns exactly 0, its excess over the CDI
has a variance of about 1e-10, and the reward of an episode starting there reached 1e4 to 1e6. Fixed with a test:
the warm start counts only sessions with an eligible name and needs 63 of them, else the start is cold.

- Only the training stretch of the 2012 window reaches those rows (2006). The 2012 cells of `ppo/dsr_cdi` and
  `ppo_shared/dsr_cdi` trained on the defect are deleted unread and trained again on the fixed code.
- The direct arms never use the reward (their loss is the Sharpe of the episode), so their cells stand.
