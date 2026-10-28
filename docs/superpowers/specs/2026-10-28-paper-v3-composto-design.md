# Paper-v3: forward paper log of the attempt-51 composite against a same-day Conservadora — design and pre-registration

_2026-10-28, session 57. Written and committed before any code of `scripts/paper_v3.py` and before the first record.
User's request (session 57: "Yes. Let's go"), on the note of DECISOES 164. After this commit the text does not change;
changing the strategy, the rules or the criterion means a new pre-registration and the clock from zero._

## 1. Why, and what it does not reopen

Attempt 51 (DECISOES 164) was the closest of the 51 backtest trials: P[> Conservadora] 0.968 / 0.955 / 0.864, failing
only phase 42. Its backtest line stays **closed**: no retuning, no other weights, no growth-only arm. This log does not
re-read 2010-2026; it measures the **same rule, unchanged, on data that does not exist yet**, so no backtest snooping is
possible. Per the rule in the header of `docs/tentativas.csv` (the same configuration on other data does not count
again; paper-v2 added no row), **the DSR N stays 51**. Note: the session-56 note and the chat said "N + 1"; the file's
rule wins and the correction is written in DECISOES.

## 2. The two arms, decided on the same close

- **Composto** (the arm under test): `core.composto.estrategia_por_nota` with attempt 51's score, built from the same
  functions as `scripts.qubo_universo_avaliar._value_score(mensal, com_crescimento=True)`: over the names with a
  Conservadora score, the mean of the normal scores of the Conservadora score, value (mean of the normal scores of the
  Magic Formula and Trending Value) and profit growth (`core.crescimento_lucro.nota_de_crescimento`), with the ones a
  name has. 15 names, 4 per sector, equal weights, the fund's fixed rules with the training window's Ledoit-Wolf
  covariance.
- **Conservadora** (control, the live strategy, not a trial): `core.sinais.estrategia_conservadora`, exactly as paper-v2
  decides it. Paper-v2 is untouched and keeps its own clock (first record 19/10); the control is recorded here so both
  arms share every date, cost and session.
- **Inputs, live, not frozen:** the lake's app exports in `data/b3lake/`: the five files of paper-v2
  (`universe_prices`, `universe_monthly`, `sectors`, `monthly_total_returns`, `dividends`) plus `fundamentals.parquet`
  and `market_caps.parquet` (CVM statements point-in-time by receipt date; market value of each issuer at the start of
  each month). The training window and universe are paper-v2's `treino_do_dia` (1,260 returns up to D, ffill of at most
  5 sessions, the point-in-time universe of the next session's month, in file order); both arms get the same window.
  The scores are those of the month of D, as in the engine.

## 3. Calendar and the routine

- Rebalance when 63 or more sessions of `lago/precos_diarios.parquet` have passed since the last record (paper-v2's
  `rebalanceio_devido`); the first record is the first run.
- On the due day the routine runs, in this order: lake `universe` (prices and universe up to D), lake `fundamentals`
  (CVM DFP/ITR, network; `cmd_fundamentals` is not part of `daily`), then checks: `universe_prices` reaches D, and
  `market_caps` has the month of D. If either command fails or either check fails, it does not decide and tries on the
  next session. Lake running (`daily.lock`) = exit without deciding.
- Freeze: a loss of 5% or more of the **Composto's** registered curve in the last session = no orders for either arm
  (`pode_negociar`; retry next session).
- Refusal: if either arm's portfolio is refused by the fixed rules, neither is recorded and the routine retries on the
  next session (both arms keep the same dates). Attempt 51 and the Conservadora had 0 refusals in 2012-2026.
- No stop. Running again on the same close records nothing.
- Records: `paper-v3/composto.csv` and `paper-v3/conservadora.csv` (date, ticker, weight, total-return index value;
  append only, `core.registro.registrar_carteira`), one line per record in `paper-v3/proveniencia.csv` (code version,
  sha256 of the seven files, the lake manifest dates) and a copy of the seven files in `paper-v3/marcos/<close>/` (the
  lake rewrites the originals; ~9 MB per record). Local commit only on `main`; the upkeep pushes.
- Scheduled task "paper v3", weekdays at 09:15 (after "paper v2" at 08:45: the two never run the lake's `universe` at
  the same time), through `scripts/rodar_paper.py paper_v3`.
- Guard: git tag `paper-v3` at the commit of the first record; the reading prints what changed since the tag in the
  decision path (paper-v2's modules + `core/composto.py`, `core/valor.py`, `core/crescimento_lucro.py`,
  `core/qubo_universo.py`, `core/views.py`, `scripts/avaliar_valor.py`, `scripts/paper_v2.py`) and in this script's
  decision definitions.

## 4. Readings and the decision (fixed now)

- With 126 sessions after the first record: report, descriptive. With 252: the decision.
- Measure (`--leitura`): each arm's recorded sequence refeito by `core.execucao.refazer` with R$ 10 thousand, the final
  proof's costs (fees 0.030%, half spread and impact with Y = 1, liquidity of the previous month from
  `monthly_metrics.parquet`) and the individual's income tax (R$ 20 thousand monthly sales exemption, dividends in the
  base), plus BOVA11 bought on the first record's close and held (no exemption), on exit values: P[> BOVA11]
  arithmetic and log, CAGR, MDD, final exit, cost per year and tax; and P[Composto > Conservadora] (PSR of the
  difference of the exit values), descriptive.
- **Decision at 252 sessions: the composite passes if its final exit value is greater than or equal to the
  Conservadora's on the same sessions.** BOVA11 is reported, not a gate. Passing does not adopt it by itself: one year
  of alpha has a standard error the size of the tracking error, so a pass means "the composite kept up live and the
  operation works"; adoption would be a separate decision written in DECISOES (e.g., running a second year). Failing
  closes the composite for good.
- Wrong data found later (checked in the lake): the recorded line stays; the report marks the stretch and DECISOES
  records it.

## 5. Expectation, written before

First record on the close of 27/10/2026 (run by hand on 28/10; the 26/10 crash is before it, so nothing is frozen);
4 rebalances in 12 months; 0 refusals; 0 to 2 frozen sessions; turnover near 300% a year for both. The two arms share
the Conservadora score and the portfolio rule, so expect a large overlap of names (backtest turnover 3.0-3.2 vs 3.0).
The likelier outcome at 252 sessions is a small gap either way that does not separate skill from luck.
