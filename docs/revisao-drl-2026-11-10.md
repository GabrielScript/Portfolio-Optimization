# Review of the DRL package written at low effort (2026-11-10)

Scope: every module of `drl/` and its tests, read against the three papers in `drl_po/` (P1 Sood et al., PPO with the
differential Sharpe ratio; P2 Sattar et al., RMS-driven A2C/DDPG/PPO with a drawdown reward; P3 Rezaei and
Nezamabadi-Pour, A2C/DDPG/PPO/SAC/TD3 review) and the PyTorch and Python pattern guides. Branch `refactor/drl-review`.

## Defects found (severity first)

1. **Two simulators.** Agents ran in `ambiente.py`; the baselines ran in a separate loop (`comparar.py`,
   `baselines_periodo.py`) that rebalanced every session while the agents traded every 21, charged no cost for
   bringing drifted weights back, and restarted every test year from zero weights. Agent and baseline rows were never
   comparable.
2. **Ignored actions were rewarded.** With `rebalancear_a_cada=21` the environment kept daily steps and replaced 20 of
   21 actions by "hold", but still credited each ignored action with that session's reward: the agent was trained on
   actions that had no effect.
3. **Cash earned nothing.** On B3 the CDI paid 5-14% a year over 2012-2026; an agent that held cash, or the drawdown
   reward that pushes towards it, was penalized by that much.
4. **A different tape from the house engine.** `lago/precos_diarios.parquet` (1,544 columns, every BDR and ETF) with
   eligibility = 252 closes, not the lake's point-in-time list. A held name that halted and reopened lost the move
   across the gap (283 cases, worst -13.9%): `nan_to_num` set the return to 0.
5. **Look-ahead in the matrix observation** (already found and fixed in commit ef94b60); the compact observation and
   the baselines' window were not covered by a test. Now all three are, each verified by a mutation.
6. **PPO was not the paper's PPO.** SB3 defaults (gamma 0.99, GAE 0.95, clip 0.2, batch 64, 10 epochs, constant
   rate) instead of P1's Table 1 (0.9, 0.9, 0.25, 1260, 16, 3e-4 annealed to 1e-5).
7. **SAC could never run** (given the on-policy `vf` head and a one-million-transition buffer), so P3's set was
   incomplete.
8. **The validation year was never used.** The protocol carved it out, but no seed selection or stopping read it.
9. **"Ledoit-Wolf" was a fixed 10% blend**, and the MVO baseline an exponentiated-gradient loop without a check against
   the closed form.
10. **Metrics with a zero risk-free rate**, and the closing summary put a drawdown measured on yearly points (-9.2%)
    next to daily ones; the Composite's Sharpe 1.20 there is not comparable with anything in the table.
11. **Lost result files and swallowed failures.** Rows with changing columns were appended to one CSV; two result files
    no longer parse. The grid runners caught every exception and moved on.
12. **Duplication.** Four Modal scripts differed in three lines each.

## Done

- Rewrite (commit fbfb310): one `PortfolioEnv` for agents and rules (`apply` is the only path that moves money); the
  lake's total-return export, point-in-time list and CDI; a step = one decision held `hold` sessions; cost on the
  notional traded against the drifted book; P1's PPO table; SAC/TD3/DDPG with a Q head and a capped buffer; seed
  selection on the validation year; one JSON file per cell; metrics over the CDI; one Modal script.
- 52 tests (`tests/test_drl_*.py`). Seven mutations (matrix and compact look-ahead, the baselines' window, the gap,
  turnover against the target instead of the drifted book, cost on every session of a hold, eligibility read for the
  decision's month) each fail a test.

## First reading on the corrected simulator (classical rules, 2012-2026, 10 bps, over the CDI)

| | CAGR | Sharpe | MDD | turnover/yr |
|---|---|---|---|---|
| Composite (house engine, reference) | 20.6% | 0.57 | -41.5% | - |
| max Sharpe (P1's MVO, Ledoit-Wolf, 252 sessions) | 14.9% | 0.32 | -49.8% | 7.2 |
| minimum variance | 7.7% | -0.07 | -38.1% | 4.4 |
| equal weight | 5.9% | -0.04 | -51.7% | 2.0 |

The first version reported minimum variance at Sharpe 0.91 and equal weight at 9.75% CAGR: zero risk-free rate,
another tape and daily rebalancing. The agents' rerun is in `drl/saidas/` under its own run name when it exists.
