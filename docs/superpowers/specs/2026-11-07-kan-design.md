# KAN forecast and KAN-15, B3 — design and pre-registration

_2026-11-07. Written and committed before any KAN forecast, IC or curve exists. User's request: test the
Kolmogorov-Arnold network (Liu et al., arXiv 2404.19756v5) on the B3 for the aggressive portfolio, after the research
note `docs/novas-arquiteturas-2026-11-07.md` (main `2bf673f`) ranked it first. Branch `feat/kan` from `feat/qml`
(`a1ed6f9`): it reuses the hybrid QPO's frozen panel, purge and reading (spec `2026-11-01-qpo-hibrido-design.md`,
sections 2, 3 and 5), so the KAN faces exactly the rows, target and gate that Ridge, the MLP, the VQC and the Composite
faced in DECISOES 213. Sections 1 to 6 are fixed from the commit on._

## 1. What is already known

DECISOES 213, same package (`D:/b3-datalake/exports/qml/b3/pacote.npz`: 45,883 rows, 205 forecast months), monthly
Spearman IC 2012-01 to 2026-07: Ridge 0.127, Composite 0.114, MLP 0.109, VQC 0.104, Fourier 0.077. The 57-parameter
MLP lost to the linear model.

**Prior, written now:** KAN-1's IC lands within +/- 0.01 of Ridge's (0.127); about 15% that KAN-1 - Ridge has t >= 2;
KAN-15 does not pass the portfolio gate (the Composite is the closest of all trials and still failed a phase).

## 2. Why the shallow KAN is a clean test (proved, note of 2026-11-07)

A [d,1] KAN computes f(x) = sum_i phi_i(x_i), phi_i(u) = w_b silu(u) + w_s sum_j c_j B_j(u), with degree-p B-splines
on a uniform grid of [a,b] extended by p knots per side. (a) By Marsden's identity the class contains every affine map
on [a,b]^d, so it nests the Ridge class on the same angles. (b) On ranked inputs it contains an equal-weight rank
average. (c) With one feature a strictly increasing phi cannot change rank IC; (d) with two or more it can. Two blind
adversarial verifiers: HOLDS. Limits: only on the grid range; the Composite proper is not nested exactly (its score is
not the equal-weight average of these six angles with missing = 0), so the Composite stays an outside benchmark. So
a KAN-1 gain over Ridge can only come from the shape of each factor's curve; KAN-2 adds interactions.

## 3. Models, each fixed here, one configuration, no search

Inputs: the package's six angles theta = pi (u - 1/2) of momentum 12-1, low volatility, dividend yield 12m, Magic
Formula, Trending Value, profit growth (missing = 0), all in (-pi/2, pi/2). The KAN layer is written in this repository
(`qpo_kaggle/kan.py`) from the paper's definition, after efficient-kan's `KANLinear` (MIT): no package download.

| Model | Definition |
|---|---|
| `KAN-1` | One layer [6,1]: grid 5 on [-pi/2, pi/2], cubic B-splines (p = 3, 8 basis functions), base silu, per-edge spline scaler; no bias (the splines' partition of unity holds the constant). Parameters 6 + 48 + 6 = 60 (MLP 57, VQC 61) |
| `KAN-2` | Two layers [6,2,1]: the first as KAN-1 with 2 outputs, the second on [-2, 2]; 120 + 20 = 140 parameters. Hidden values outside [-2, 2] pass only through silu |

Initialization, seeded: base weights U(-1/sqrt(fan_in), 1/sqrt(fan_in)), spline coefficients U(-0.1, 0.1), scalers 1.
Training: exactly the MLP's loop of 213 (`prever_torch`, `Treino()`: MSE, Adam lr 0.02, batch 1,024, 400 steps on the
first month then 50 warm-started, rows and batches seeded by seed and month), float64, seeds 0, 1, 2 averaged. No grid
update, no sparsity penalty, no pruning (none of these was given to the MLP either).

## 4. Forecast test (signal)

Window 2012-01 to 2026-07, the 213 reading's. **Gate: IC >= 0.02 and Newey-West t (2 lags) >= 2.** Reported, not
gates: paired differences KAN-1 - Ridge, KAN-1 - MLP, KAN-1 - Composite, KAN-2 - KAN-1, each with its t. A KAN edge is
claimed only if KAN-1 passes the gate and both KAN-1 - Ridge and KAN-1 - Composite have t >= 2.

Curves, descriptive only: the six learned phi_i of each seed's last-month KAN-1, on 21 points of [-pi/2, pi/2], and
each curve's non-linearity 1 - R^2 of its best line. Turning any curve into a fixed formula inside the Composite would
be a new configuration with its own pre-registration; it is not part of this one.

## 5. Portfolio arm and gate

`KAN-15`: KAN-1's forecast through `estrategia_por_nota` (15 names, 4 per sector, equal weights), the rule of VQC-15.
Engine, data and checks as the 213 reading: `walk_forward_dinamico`, `CFG_BT` (1,260 sessions, every 63, 10 bps per
side), phases 0/21/42, cut 16/10/2026, frozen `docs/dados/2026-10-22/`, `2026-10-24-cvm/`, `2026-10-21-macro/`; the
Conservadora must reproduce 17.41/16.41/16.90% and the Composite 21.4/19.9/19.1% before KAN-15 is read.
**Gate, every phase, all >= 0.95:** P[> Composite], P[> Conservadora], P[> BOVA11], DSR. `KAN-15` runs whatever the
signal test says (the user asked for the comparison); it is a candidate for a forward paper log only if KAN-1 also
passes section 4. KAN-2 gets no portfolio arm.

**N.** One new configuration (KAN-15), one row in this branch's `docs/tentativas.csv` in this commit. The branches
disagree on N (main 55, `feat/qml` 57, the US batches up to 92 plus main's Top10), so the DSR uses the conservative
union **N = 94** (93 + this one), printed beside the branch count.

## 6. Compute and output

Local CPU, `.venv-qpo` (torch 2.14.1+cpu), 3 threads; the MLP took 110 s on this package. The KAN columns are added
to the same `previsoes.npz` (the 213 columns are not touched: checked by hash before and after). Output:
`docs/kan-2026-11-07.txt` (IC table, differences, curves, portfolio table, year table, gate) and a DECISOES entry.
Nothing goes to the app.
