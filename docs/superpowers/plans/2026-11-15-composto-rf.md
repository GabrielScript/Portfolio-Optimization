# The Composite's Inputs Through a Random-Features Net: Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Read once whether a Gaussian-kernel ridge (the infinite-width limit of a random-features net) on the Composite's six legs orders monthly returns beyond the Composite's three components, and run its 15-name arm only if it does.

**Architecture:** A pure numeric module (`core/random_features.py`: kernel, fit, alpha choice) under a domain module (`core/composto_rf.py`: a month's inputs and target, the rolling 60-month scores, the residual, the scores as a `NotaDoMes`). One script freezes the scores of every month 2010-01..2026-10; a second reads stage 1 (IC) from that file; stage 2 is a new `rf` family in `scripts/composto_variantes_avaliar.py`, built only if stage 1 names a candidate.

**Tech Stack:** Python 3 (project `.venv`), numpy, scipy.linalg (`cho_factor`/`cho_solve`), pandas, pytest. No new dependency (scikit-learn is not used).

**Spec:** `docs/superpowers/specs/2026-11-15-composto-rf-design.md` (the pre-registration; commit it before any code).

## Global Constraints

- Branch `feat/composto-rf`, worktree `../po-rf`, base `feat/composto-faixa` `cde5e5b`. Never commit work in progress to `main` (the upkeep pushes it at 10:00).
- Python: `$PY = ..\Portfolio-Optimization\.venv\Scripts\python.exe`; scripts run as `& $PY -X utf8 scripts/<name>.py`.
- Before writing any Python, load the skills python-design-patterns and python-patterns (user rule), then rigorous-coding.
- TDD one test at a time: write it, run it, see it fail for the right reason, then the code. A test that passes without the change, or a workaround in a test, is rejected.
- No network in any reading. Frozen inputs only: `docs/dados/2026-10-22/`, `docs/dados/2026-10-24-cvm/`.
- Never edit `core/market_data.py`, `core/__init__.py`, `core/lago.py`, `core/paper.py`, `core/bcb.py`, `assets.py` (paper guards).
- Registered values, verbatim from the spec: inputs `vol, momentum, dy, formula, trending, crescimento`; gamma = 1/6; window 60 months, minimum 36; alpha grid (0.1, 1, 10, 100, 1000), chosen each January and at the first scored month on fit M-60..M-13 / validate M-12..M-1, tie to the larger; IC period 2012-01..2026-09; gate IC >= 0.02 and t >= 2; candidate also needs residual IC > 0 and t >= 2; anchors 0.085 / 0.070 / 0.033 (tolerance 0.003); stage 2 arm `RF-kernel-15`, N 190, gate P[> Composite] >= 0.95 and DSR >= 0.95 in phases 0/21/42.
- Each reading runs once. Its output file is committed as printed; a crash before any arm number is fixed and rerun, a crash after is reported, not rerun silently.
- One heavy job at a time, never during the lake's nightly 21:45-23:45. The score panel (Task 3) is the heavy one.
- US English in code, docs and commits; no mention of TCC or "Neumann". LF line endings.
- Commit messages end with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`. Push is allowed, never `--force`.

## Review Focus

1. Look-ahead: the score at month M must not move when any return of month M or later changes. Test in Task 2.
2. A month with fewer than 36 training months has no score; in stage 2 that decision uses the Composite's score and is logged, not a crash. Tests in Task 2.
3. Names before ~2010 have no CVM legs (`formula`, `trending`, `crescimento`): they count as 0, not dropped; a name without `vol` or `momentum` is dropped. Test in Task 2.
4. A validation month whose predictions are constant gives a NaN Spearman IC: the mean skips it, and an alpha whose every month is NaN never wins. Test in Task 1.
5. Two alphas with the same validation IC: the larger wins. Test in Task 1.

---

### Task 1: Kernel ridge and the alpha choice

**Files:**
- Create: `core/random_features.py`
- Test: `tests/test_random_features.py`

**Interfaces:**
- Produces:
  - `GRADE_ALPHA: tuple[float, ...] = (0.1, 1.0, 10.0, 100.0, 1000.0)`
  - `kernel_rbf(a: np.ndarray, b: np.ndarray, gamma: float) -> np.ndarray`: `exp(-gamma * |a_i - b_j|^2)`, shape (len(a), len(b)).
  - `@dataclass(frozen=True) class KernelRidge: treino: np.ndarray; coef: np.ndarray; gamma: float` with `prever(self, x: np.ndarray) -> np.ndarray`.
  - `ajustar_kernel(x: np.ndarray, y: np.ndarray, alpha: float, gamma: float) -> KernelRidge`: coef = (K + alpha I)^-1 y, no intercept (targets are centred normal scores), `cho_factor(..., overwrite_a=True)` so only one n x n matrix lives at a time.
  - `ajustar_linear(x: np.ndarray, y: np.ndarray, alpha: float) -> np.ndarray`: ridge coefficients (X'X + alpha I)^-1 X'y, no intercept.
  - `Previsor = Callable[[np.ndarray], np.ndarray]`; `Ajuste = Callable[[np.ndarray, np.ndarray, float], Previsor]`
  - `@dataclass(frozen=True) class Bloco: x: np.ndarray; y: np.ndarray; meses: np.ndarray` (one row per name-month).
  - `ic_medio(previsto: np.ndarray, alvo: np.ndarray, meses: np.ndarray) -> float`: mean over months of the Spearman correlation, months with NaN skipped; all NaN gives `-inf`.
  - `escolher_alpha(ajuste: Ajuste, treino: Bloco, validacao: Bloco, grade: Sequence[float] = GRADE_ALPHA) -> float`

- [ ] **Step 1: Write the failing test**

```python
def test_kernel_is_one_on_the_diagonal_and_matches_the_formula():
    a = np.array([[0.0, 0.0], [1.0, 2.0]])
    k = kernel_rbf(a, a, gamma=0.5)
    assert np.allclose(np.diag(k), 1.0)
    assert np.isclose(k[0, 1], np.exp(-0.5 * 5.0))
```

- [ ] **Step 2: Run it and see it fail**

Run: `& $PY -m pytest tests/test_random_features.py -q`
Expected: FAIL, `ModuleNotFoundError: core.random_features`

- [ ] **Step 3: Implement `kernel_rbf`** with squared distances from `|a|^2 + |b|^2 - 2ab'`, clipped at 0.

- [ ] **Step 4: Run it and see it pass**

- [ ] **Step 5: One test at a time, each run red then green, in this order**

```python
def test_fit_solves_the_ridge_system():
    x = np.arange(10.0).reshape(5, 2) / 10; y = np.array([1.0, -1.0, 0.5, 0.0, 2.0])
    modelo = ajustar_kernel(x, y, alpha=0.3, gamma=1.0)
    k = kernel_rbf(x, x, 1.0)
    assert np.allclose(modelo.coef, np.linalg.solve(k + 0.3 * np.eye(5), y))
    assert np.allclose(modelo.prever(x), k @ modelo.coef)

def test_a_tiny_alpha_interpolates_and_a_huge_one_shrinks_to_zero():
    # interpolation: |prever(x) - y| < 1e-4 at alpha 1e-9; shrink: max |prever(x)| < 1e-3 at alpha 1e6

def test_linear_ridge_matches_its_closed_form():
    # ajustar_linear(x, y, 2.0) == np.linalg.solve(x.T @ x + 2 * I, x.T @ y)

def test_mean_ic_skips_a_constant_month_and_all_nan_is_minus_infinity():
    # months [1,1,1,2,2,2]: month 1 perfectly ranked (IC 1), month 2 constant prediction -> ic_medio == 1.0
    # every month constant -> ic_medio == -inf

def test_alpha_choice_takes_the_best_validation_ic():
    # an Ajuste stub whose predictor is good only for alpha 10, noise for the others -> escolher_alpha == 10.0

def test_alpha_tie_goes_to_the_larger():
    # an Ajuste stub giving the same predictor for every alpha -> escolher_alpha == 1000.0
```

- [ ] **Step 6: Commit**

```bash
git add core/random_features.py tests/test_random_features.py
git commit -m "feat: Gaussian-kernel ridge, linear ridge and the pre-registered alpha choice (random_features)"
```

---

### Task 2: The month's inputs, the rolling scores, the residual

**Files:**
- Create: `core/composto_rf.py`
- Test: `tests/test_composto_rf.py`

**Interfaces:**
- Consumes: Task 1's `kernel_rbf`, `ajustar_kernel`, `ajustar_linear`, `Bloco`, `escolher_alpha`, `GRADE_ALPHA`; `core.views.escore_normal`; `core.sinais.volatilidade_36m`, `momentum_12_1`, `dividend_yield_12m`; `core.valor.NotaDoMes`.
- Produces:
  - `ENTRADAS = ("vol", "momentum", "dy", "formula", "trending", "crescimento")`; `GAMMA = 1.0 / len(ENTRADAS)`; `JANELA = 60`; `MINIMO = 36`; `VALIDACAO = 12`
  - `@dataclass(frozen=True) class FontesRF: retornos: pd.DataFrame; proventos: pd.DataFrame; formula: NotaDoMes; trending: NotaDoMes; crescimento: NotaDoMes`
  - `entradas_no_mes(fontes: FontesRF, mes: pd.Timestamp, candidatos: Sequence[str]) -> pd.DataFrame`: index = names with `vol` and `momentum`, columns `ENTRADAS`, each the normal score among the names that have it, missing = 0.0.
  - `alvo_no_mes(retornos: pd.DataFrame, mes: pd.Timestamp, papeis: Sequence[str]) -> pd.Series`: normal score of month `mes`'s total return among `papeis` with a return.
  - `meses_de_escolha(pontuados: pd.DatetimeIndex) -> pd.DatetimeIndex`: the first scored month and every January after it.
  - `@dataclass(frozen=True) class Notas: kernel: pd.DataFrame; linear: pd.DataFrame; alphas: pd.DataFrame` (month x ticker, month x ticker, month x {"kernel", "linear"}).
  - `notas_rf(entradas: Mapping[pd.Timestamp, pd.DataFrame], alvos: Mapping[pd.Timestamp, pd.Series], meses: pd.DatetimeIndex) -> Notas`: for each M in `meses`, training rows = months M-60..M-1 present in both mappings; fewer than 36 such months -> no row for M.
  - `residuo(sinal: pd.Series, componentes: pd.DataFrame) -> pd.Series`: same definition as `core.indicadores_no_tempo.residuo` on the unmerged `feat/indicadores-risco` (copy its docstring; intercept + component normal scores, missing component 0, empty with fewer names than the fit needs).
  - `nota_do_painel(painel: pd.DataFrame, reserva: NotaDoMes, sem_nota: list[pd.Timestamp]) -> NotaDoMes`: the panel's row for the month on the candidates (dropna); an empty row returns `reserva(mes, candidatos)` and appends the month to `sem_nota`.

- [ ] **Step 1: One test at a time, each run red then green, in this order** (`& $PY -m pytest tests/test_composto_rf.py::<name> -q`)

```python
def test_inputs_are_normal_scores_with_missing_cvm_legs_as_zero():
    # a name with vol and momentum but no formula/trending/crescimento keeps a row with those three == 0.0;
    # every column has mean ~0 over the names that have that leg

def test_a_name_without_vol_or_momentum_has_no_inputs():
    # a name with 20 months of returns (no 36-month vol) is not in entradas_no_mes(...).index

def test_target_is_the_normal_score_of_that_months_return():
    # alvo_no_mes ranks the month's returns: highest return -> highest score, scores sum ~0

def test_choice_months_are_the_first_and_each_january():
    # pontuados 2010-05..2012-03 -> [2010-05, 2011-01, 2012-01]

def test_fewer_than_36_training_months_gives_no_score():
    # 35 months of synthetic entradas/alvos before M -> M not in notas.kernel.index; 36 -> present

def test_the_score_at_m_ignores_returns_of_m_and_later():
    # perturb alvos[M] and alvos[M+1] -> notas.kernel.loc[M] unchanged (to 1e-12)

def test_data_older_than_60_months_does_not_matter():
    # perturb alvos[M-61] -> notas.kernel.loc[M] unchanged; perturb alvos[M-60] -> changed

def test_residual_of_a_combination_of_the_components_is_zero():
    # sinal = 2*c1 - c2 on normal-scored components -> |residuo| < 1e-9 for every name

def test_panel_score_falls_back_to_the_reserve_and_logs_the_month():
    # empty panel row for M -> returns reserva's series, sem_nota == [M]; a full row -> the row on the candidates
```

- [ ] **Step 2: Implement each function right after its test fails.** `notas_rf` stacks the window into one `Bloco` (meses = month of each row), calls `escolher_alpha` only in `meses_de_escolha` months (kernel and linear each keep their own alpha), then fits on the whole window and scores `entradas[M]`. Kernel predictor: `lambda x, y, a: ajustar_kernel(x, y, a, GAMMA).prever`; linear: `x @ ajustar_linear(x, y, a)`.

- [ ] **Step 3: Run the whole file and the full suite**

Run: `& $PY -m pytest tests/test_composto_rf.py tests/test_random_features.py -q`, then `& $PY -m pytest -q -x`
Expected: all pass; paper guards unchanged: `& $PY -c "from scripts import paper_trading, paper_v2; print(paper_trading.codigo_mudado_desde_o_paper(), paper_v2.codigo_mudado_desde_o_paper())"` prints `[] []`.

- [ ] **Step 4: Commit**

```bash
git add core/composto_rf.py tests/test_composto_rf.py
git commit -m "feat: the Composite's six legs as inputs, rolling 60-month kernel and linear scores, residual (composto_rf)"
```

---

### Task 3: Freeze the score panel (the heavy job)

**Files:**
- Create: `scripts/composto_rf_painel.py`
- Create (by the run): `docs/dados/composto-rf/notas_kernel.parquet`, `notas_linear.parquet`, `alphas.parquet`, `SHA256SUMS`; `docs/composto-rf-painel-<run date>.txt`

**Interfaces:**
- Consumes: Task 2's `FontesRF`, `entradas_no_mes`, `alvo_no_mes`, `notas_rf`; loaders `scripts.avaliar_robustez.carregar`, `scripts.avaliar_conservadora.carregar_mensal(FOLDER)`, `scripts.avaliar_valor.carregar_balancos(PASTA_CVM)` and `notas(balancos, mensal)` (keys `FORMULA`, `TRENDING`), `core.crescimento_lucro.nota_de_crescimento(balancos.fundamentos)`; `FOLDER` from `scripts.qpo_montar`.
- Produces: the three parquet files (index = month, first day), read by Tasks 4 and 5.

- [ ] **Step 1: Write the script.** Docstring = how to run and what it freezes (cite the spec). Inputs for every month 2005-01..2026-10 on the month's point-in-time universe (`dados.universo`, columns `month`, `security`), targets 2005-01..last complete month, scores for 2010-01..2026-10; print the input SHA-256s (as `indicadores_risco_ic.py` does), rows per month, the alphas chosen, elapsed time; write the files and `SHA256SUMS`.

- [ ] **Step 2: Dry check on a slice.** Run with `--ate 2010-03` (an argument that stops after three scored months, nothing written): prints three months of scores, no traceback, peak memory below 4 GB (Task Manager or `psutil` print at the end).

- [ ] **Step 3: Commit the script before the run**

```bash
git add scripts/composto_rf_painel.py
git commit -m "feat: script that freezes the random-features scores of every month (no reading yet)"
```

- [ ] **Step 4: Run once, alone, in the background** (expected about 1 hour; outside 21:45-23:45)

Run: `& $PY -X utf8 scripts/composto_rf_painel.py > docs/composto-rf-painel-<run date>.txt`
Expected: exit 0, 202 scored months (2010-01..2026-10) minus any with fewer than 36 training months (listed), files written. Liveness: check the process and the output file, not only a quiet log.

- [ ] **Step 5: Commit the frozen data**

```bash
git add docs/dados/composto-rf docs/composto-rf-painel-*.txt
git commit -m "data: frozen random-features scores 2010-01..2026-10 (kernel and linear ridge)"
```

---

### Task 4: Stage 1, the IC reading (once)

**Files:**
- Create: `scripts/composto_rf_ic.py`
- Create (by the run): `docs/composto-rf-ic-<run date>.txt`, `docs/dados/composto-rf/ics.csv`

**Interfaces:**
- Consumes: the Task 3 parquet files; Task 2's `residuo`; `core.poder.ic_por_mes`; `scripts.avaliar_conservadora.resumo_do_ic`, `retornos_no_universo`; `scripts.avaliar_valor.PORTAO`, `PERIODO_DO_IC`, `SUBPERIODOS_DO_IC`; the three components as in `scripts/indicadores_risco_ic.py` on `feat/indicadores-risco` (Conservadora, `nota_media_dos_escores` of value, profit growth); the Composite's score `scripts.qubo_universo_avaliar._value_score(mensal, com_crescimento=True)[0]`.

- [ ] **Step 1: Write the script.** Same shape as `indicadores_risco_ic.py`: hashes (inputs and the frozen panel), anchors first (stop if any is off by more than 0.003), then one row each for `RF-kernel` and `ridge-linear` with: months, median names, IC, t, months IC > 0, the three subperiods, residual IC and t, rank correlation with each component and with the Composite's score, gate PASS/fail, stage 2 `CANDIDATE`/`no` (only `RF-kernel` can be a candidate; the linear row is description). Then the alphas chosen per year. Save monthly ICs to `ics.csv`.

- [ ] **Step 2: Commit the script before the reading**

```bash
git add scripts/composto_rf_ic.py
git commit -m "feat: stage 1 script of the random-features reading (no number read)"
```

- [ ] **Step 3: Read once**

Run: `& $PY -X utf8 scripts/composto_rf_ic.py > docs/composto-rf-ic-<run date>.txt`
Expected: anchors within 0.003; one table; the line `Stage 2 candidate: RF-kernel` or `Stage 2 candidate: none`.

- [ ] **Step 4: Commit the output as printed, record it**

```bash
git add docs/composto-rf-ic-*.txt docs/dados/composto-rf/ics.csv
git commit -m "result: stage 1 IC reading of the random-features scores (read once)"
```

Then DECISOES 371 in the main checkout's local `DECISOES.md` (CRLF, git-ignored): the numbers, the prior against the result, N unchanged at 189 if no candidate. With no candidate, skip Task 5.

---

### Task 5: Stage 2, the `RF-kernel-15` arm (only if Task 4 named a candidate)

**Files:**
- Modify: `scripts/composto_variantes_avaliar.py` (the `Arm` dataclass, `FAMILIES`, `N_FIXED`, `REPLAY_BASE`, the strategy choice in `main`)
- Modify: `docs/tentativas.csv` (one row)
- Create (by the run): `docs/composto-rf-<run date>.txt`, `docs/dados/composto-rf-curvas-<run date>.parquet`

**Interfaces:**
- Consumes: Task 2's `nota_do_painel`; `notas_kernel.parquet`; `core.composto.estrategia_por_nota`.
- Produces: `Arm.rf: bool = False`; `RF_ARMS = {"RF-kernel-15": Arm(CFG_BT, rf=True)}`; `N_FIXED["rf"] = 190`; `"rf"` in `REPLAY_BASE`; `FAMILIES["rf"]`.

- [ ] **Step 1: Write the failing test** in `tests/test_composto_rf.py`:

```python
def test_the_rf_family_is_one_arm_at_n_190_against_the_replay():
    from scripts import composto_variantes_avaliar as cva
    arms, _ = cva.FAMILIES["rf"]
    assert list(arms) == ["RF-kernel-15"] and arms["RF-kernel-15"].rf
    assert cva.N_FIXED["rf"] == 190 and "rf" in cva.REPLAY_BASE
```

- [ ] **Step 2: Run it, see it fail (`KeyError: 'rf'`), implement, run it green.** In `main`, an `rf` arm's strategy is `estrategia_por_nota(nota_do_painel(painel, notas[False], sem_nota), dados.setores, limites=arm.limites)`; print `sem_nota` after each phase.

- [ ] **Step 3: Add the `tentativas.csv` row** (date, `composto_variantes_avaliar.py`, "RF-kernel-15: the 15 names (<=4 per sector) of largest Gaussian-kernel ridge score on the Composite's six legs (60-month window, gamma 1/6, alpha by January validation), equal weights, house rules, 10 bps"), then commit script, test and row before the run.

- [ ] **Step 4: Read once**

Run: `& $PY -X utf8 scripts/composto_variantes_avaliar.py rf > docs/composto-rf-<run date>.txt`
Expected: three phases, the arm next to the Composite replay, the Conservadora, BOVA11 and the CDI; P[> Composite] and DSR per phase.

- [ ] **Step 5: Commit the output, record it** (DECISOES 372 with N 190; a pass means a forward ghost account only, never the book).

---

### Task 6: Close

- [ ] `graphify update .` in the worktree; `git push -u origin feat/composto-rf` (no merge into `main`: research branch).
- [ ] `session_state.md` checkpoint in the main checkout; a memory file for the reading with the DECISOES numbers.
