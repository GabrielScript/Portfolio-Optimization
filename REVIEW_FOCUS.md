# Review focus: the engine that measures the Composite (walk-forward, score combiner, loaders, costs, metrics)

The Composite's score and selection (`core/composto`, `core/valor`, `core/sinais`, `core/crescimento_lucro`) and the
lake's data chain were reviewed before. This branch is what turns them into the numbers every decision rests on
(Composite 21.8 / 20.3 / 19.2% a year in phases 0 / 21 / 42; 133 trials judged against it). Modules imported but not
here (`core/composto`, `core/valor`, ...) were reviewed; read them as correct.

Look first for anything that makes the backtest better than a real investor could do, in this order:

1. **Look-ahead in the walk-forward** (`core/backtest.py`, `scripts/avaliar_universo_dinamico.py`): the decision on a
   session must use data up to that session's close only; trades at the next price, not the same close used to decide;
   the monthly universe of the decision month, not a later one; `Congelamento` and the training window.
2. **The score combiner** (`core/qubo_universo.composite_score`, `core/views.nota_media_dos_escores`): normal scores
   among the month's candidates only, missing components, ties, a name scored with another month's data.
3. **App-side loaders** (`scripts/avaliar_robustez.carregar`, `scripts/avaliar_valor.carregar_balancos` / `notas`,
   `scripts/avaliar_conservadora.carregar_mensal`): point-in-time statements (`available_from`), monthly returns and
   dividends (double counting with total-return prices), delisted names, `_acoes`.
4. **Costs and rules** (`core/execucao`, `core/rebalanceio`, `core/risco`): 10 bps and liquidity cost applied on every
   trade, turnover, the 20% name and 30% sector caps, cash.
5. **Metrics** (`core/metrics`, `core/robustez`): CAGR, Sharpe over the CDI, drawdown, P[> Composite] (probabilistic
   Sharpe of the difference), DSR with the number of trials.

## Added after a call trace of the Composite's run (2026-11-13)

A profile of the Composite's phase-0 replay (CAGR 0.2179) and of paper-v3's 27/10 decision listed the unreviewed code
that actually runs. Same priorities as above:

6. **The covariance behind the risk check** (`core/estimation.estimar_momentos`, `covariancia_ledoit_wolf`):
   `core/composto.estrategia_por_nota` passes it to `core/risco.checar_carteira`, which can refuse a portfolio (the
   engine then keeps the current one). Shrinkage, the window, missing returns.
7. **The backtest's parameters** (`core/config.py`): `ConfigBacktest` (costs, rebalance rule, defaults) and its checks.
8. **Helpers on the path**: `scripts/avaliar_regras_no_motor.setores_dos_papeis` (the sector map behind the 30% cap),
   `scripts/fermionica_grade_avaliar.memoized` (a month's score cached: a wrong key would reuse another month's),
   `scripts/qpo_montar` (the frozen data folder, the last session and the run date).
9. **The gate's table** (`scripts/xquad_avaliar.metricas`, with `core/market_data`'s CDI and Selic for the risk-free
   rate): P[> Composite], DSR, alpha and Sharpe of every reading.
10. **The lake's orchestration** (`lake/src/b3lake/cli.py`, from the b3-datalake repo; its modules were reviewed
    separately): the order of bronze, silver, gold and export, and the arguments each step gets (a missing folder that
    silently skips a step such as the CVM share factors).
