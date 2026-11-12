# MARS-R: MARS with three learning repairs (pre-registered 2026-11-12)

Written and committed before any MARS-R number exists. Branch `feat/mars-r`, from `feat/mars` (5dfd76d). Everything
not named here is `2026-11-08-mars-design.md` unchanged: the frozen package `mars_pacote.npz`, the 50 names a year,
the features, the overlay, the risk score, the ten agents, the 30 episodes, the validation rule, the seeds, the
chained test years, the reading script and the gate.

## Why

The DRL review of 2026-11-12 read `qpo_kaggle/mars.py` line by line. The code does what the MARS spec registered,
and the spec follows the paper. Three of the paper's own choices weaken what the agents can learn; the user asked
for them to be repaired and the run repeated (free GPU on Kaggle). This is our variant, not the paper.

## The three repairs (arms `MARS-50R` and `MARS-50R+score`)

1. **The critic is trained on the raw action.** Algorithm 1 trains Q on the executed action A' and improves the
   actor through Q at its raw output A. The overlay cuts A by a large factor whenever cash binds (50 buys of 10% of
   V ask for five times the portfolio), so the actor's gradient is taken where Q saw no data. Here Q is trained on
   (s, A, R, s'): the overlay is part of the environment, as in any DDPG with a clipped action.
2. **The cost is subtracted once.** R = (V' - V)/V - (0.5 sigma_30 + 2.0 DD_30); V is already net of costs.
3. **The controller reads the current critics.** Its loss uses Q_i and C_i of the sampled states computed with the
   networks as they are when it trains (at the current actors' actions), not the values stored when each
   transition was collected.

Kept: a name is sold at its last price the session before it ends (the spec's accounting convention).

## Reading and gate

`scripts/mars_avaliar_b3.py <outputs> R`: the same engine, checks, phases and gate (every phase, all >= 0.95:
P[> Composite], P[> Conservadora], P[> BOVA11], DSR). Two new rows; the DSR uses the union of every branch's
trials: 117 + 4 (DRL v3, `refactor/drl-v3`) + 2 = **N 123**. The first reading's arms are printed beside the new
ones only as description (MARS-50 13.8/13.7/13.8%, MARS-50+score 13.6/13.4/13.3%).

## Compute

Kaggle, one GPU kernel per arm (`qpo_kaggle/mars_kernel.py`): the three seeds as three processes, the 17 years in
sequence, then the chained test; the package rides as the private dataset `mars-pacote`. Estimated 3 to 6 hours
per kernel.
