# QML arm of the QPO: design and pre-registration (test branch)

Date: 2026-11-09. Branch: `feat/qml-quantum-portfolio` (never merged into production without the user's word).
Status: descriptive study, not a reading; N in `docs/tentativas.csv` does not change.

## Question

The QPO's QUBO spends quantum hardware on *slicing the cake* (which 15 names, which weights). This
study moves the quantum work to *finding the recipe*: a variational quantum circuit reads each
asset's public features and returns P(up) over the next 63 sessions; the existing CVXPY engine then
allocates the money under the fixed rules. The question is whether that split beats the QUBO, the
Conservadora and, above all, the Composite.

## Architecture

- **Brain (quantum)**: PennyLane `default.qubit`, six wires, `AngleEmbedding(Y)` of six
  cross-sectionally standardized features, two `StronglyEntanglingLayers`, one `expval(Z_i)` per wire,
  a classical linear readout (six numbers -> one logit) and a learnable scale/bias, BCE on the
  relative direction label. PennyLane `TorchLayer` with the Torch interface and backpropagation;
  training on CPU (the circuit is 6 qubits, GPU is not the bottleneck). One model is trained per
  two-year block, only on labels realized before 1 January of the block.
- **Muscle (classical)**: `core.optimization.otimizar` (CVXPY) receives
  `mu_i = rf + tilt * (2 p_i - 1) * sigma_i` and enforces 20% per name, 30% per sector, 40% ex-ante
  volatility and no leverage; `core.risco.checar_carteira` checks the result again.
- **Features (all public at the decision session)**: 21- and 63-session total return, 12-1 momentum,
  21- and 63-session realized volatility, drawdown from the 252-session high.
- **Label**: sign of the forward 63-session total return minus the cross-sectional median at that
  date (relative, so the circuit never learns the market level).

## Leakage controls

1. `features_at` reads only rows up to the decision date.
2. A panel sample carries the date its label is realized. The model for year Y is trained only on
   samples whose label was realized before 1 January of Y.
3. Cross-sectional standardization is done inside each date, never with full-sample statistics.
4. The rebalance signal is the cached prediction for the decision session (or the prior session when
   a freeze moves the rebalance); the cache is keyed by date and only stores causal predictions.

## Arms (same engine, dates, universe, costs and rebalances)

| Arm | Universe | Objective | Meaning |
|---|---|---|---|
| Composite | full eligible | equal weights, 15 names | baseline the user wants to beat |
| Conservadora | full eligible | equal weights, 15 names | second baseline |
| Quântica (QUBO) | full eligible | 15 picks + max Sharpe | the current quantum arm |
| QML-Composite (Máx. Sharpe) | Composite's 15 names | max Sharpe on QML mu | does QML help inside the proven book? |
| QML-Composite (Meio-Kelly) | Composite's 15 names | Kelly λ=2 on QML mu | same, growth objective |
| QML-Meio-Kelly (Máx. Sharpe) | half-Kelly's names | max Sharpe on QML mu | the user's "Meio-Kelly-Alfa" variant |
| QML-Universo (Máx. Sharpe) | full eligible | max Sharpe on QML mu | QML as the only alpha |
| QML+Composite (λ, Máx. Sharpe) | full eligible | max Sharpe on QML mu | selection by Composite score + λ·QML conviction |
| Média-variância no Composite (Meio-Kelly) | Composite's 15 names | Kelly λ=2 on Jorion mu | classical control: names vs quantum |

Sensitivity: tilt 0.5 and 2.0 on the QML-Composite max-Sharpe arm; λ 0.25, 0.5 and 1.0 on the combined
selection. They are reported, not used to pick a production configuration.

## Metrics

Total return, CAGR, Sharpe (over the CDI), Sortino, Jensen alpha per year against BOVA11 with a
Newey-West standard error, its t statistic, maximum drawdown, volatility and mean number of names.
The same regression against the Composite is reported beside the alpha against BOVA11.

## Decision rule (written before the run)

The QML arm only becomes a candidate if, on phase 0 **and** on phases 21 and 42 (the engine's
pre-registered cuts), it beats the Composite on CAGR and does not lose on Sharpe, with t(alpha) >= 2.
A single phase is not a result. Otherwise the line is recorded as not a candidate, as every other
failed line in `DECISOES.md`.

## Limits

The circuit is small by design (six qubits); a simulator has no quantum advantage claim. The study
measures whether the hybrid adds value over the project's own classical baselines, nothing more.
