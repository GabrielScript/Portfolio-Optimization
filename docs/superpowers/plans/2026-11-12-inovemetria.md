# Inovemetria Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or
> superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Five point-in-time innovation and diffusion signals for every issuer of the lake, read once against the
house IC gate, then eight portfolio arms (one a control) read once against the Composite's gate, plus a disruption map.

**Architecture:** The lake (`b3-datalake`, worktree `../../b3-datalake-inov`, branch `feat/innovation-data`) gains
one command, `innovation`, that writes four parquet exports from raw files it already has plus INPI's BADEPI zip.
The optimizer (`po-inov`, branch `feat/inovemetria`) freezes them in `docs/dados/2026-11-12-inovemetria/`, builds
the signals as `NotaDoMes` functions in a new package `core/inovemetria/`, and reads them with three scripts that
reuse the existing IC harness and backtest engine. Nothing under the paper guards' paths is edited.

**Tech Stack:** Python 3.14; lake: polars; optimizer: pandas, numpy, scipy (`least_squares`), pytest.

**Spec:** `docs/superpowers/specs/2026-11-12-inovemetria-design.md` (committed before any return-based number).

## Global Constraints

- Every signal uses only statements with `available_from < month` (`core.valor.documentos_conhecidos`) and patents
  and subsidiary relations with `known_from < month`.
- Signs, windows and bounds are the spec's and are not tuned: S horizon 4 quarters, at most 40 quarter ends, at
  least 12 consecutive pairs; E 12 quarter ends, at least 3 issuers; P 60 months and 24 + 24 months; X 36 months,
  at least 24; theta 0.5; kernel mix 1/2; 15 names, 4 per sector.
- Stage 1 (IC) is run and committed before stage 2 (arms) is run. Each script's output is redirected once to
  `docs/` and committed with the code that produced it. No arm is rerun after it is read.
- All new text (code, comments, docs, commits) in US English. Tests are written and seen failing one at a time.
- RAM: read the BADEPI CSVs lazily (polars `scan_csv`), one table at a time; no two heavy jobs at once.
- Run optimizer code with `../Portfolio-Optimization/.venv/Scripts/python -X utf8`; lake code with
  `../b3-datalake/.venv/Scripts/python` and `PYTHONPATH=src`.

## Review Focus

- A statement refiled after the decision month must not change that month's signal (dated by `available_from`).
- An issuer with a gap in its quarters (a missing ITR) gets pairs only across consecutive quarter ends, not across
  the gap; with fewer than 12 pairs it has no S.
- Revenue zero or negative in any quarter of the window: no S and no E for that issuer (logs undefined).
- An issuer whose subsidiaries are known only from a later FRE gets no credit for their patents earlier.
- A candidate set where no issuer has patents: the kernel is the identity and arms 6 and 7 must still run.

---

## Part A: lake (`b3-datalake-inov`)

### Task A1: `curated_accounts` takes the account table

**Files:** Modify `src/b3lake/fundamentals.py` (`curated_accounts`); Test `tests/test_fundamentals.py`.

**Interfaces:** Produces `curated_accounts(statements: pl.DataFrame, accounts: Mapping[str, tuple[str, str, str]] =
ACCOUNTS) -> pl.DataFrame` (same output; the mapping replaces the module constant inside the function).

- [ ] Test `test_curated_accounts_reads_the_account_table_it_is_given`: a BPA row with code `1.02.04.01`, fixed,
  latest, period end = dt_refer, value 7 (thousand) gives one row `account == "intangibles"`, value 7000.0 with
  `accounts={"intangibles": ("BPA", "1.02.04.01", "stock")}`, and no row with the default table. See it fail.
- [ ] Implement; run the whole `tests/test_fundamentals.py` (all pass); commit.

### Task A2: `b3lake/innovation.py` readers

**Files:** Create `src/b3lake/innovation.py`, `tests/test_innovation.py`.

**Interfaces (produces):**
- `INTANGIBLES = {"intangibles": ("BPA", "1.02.04.01", "stock")}`
- `cnpj_root(column: str) -> pl.Expr`: digits only, left-padded to 14, first 8.
- `read_subsidiaries(fre_zip: Path) -> pl.DataFrame`: columns `code_cvm` Int64, `cnpj_root` Utf8, `dt_refer`
  Date, `known_from` Date. Rows of `fre_cia_aberta_participacao_sociedade_<year>.csv` with `Tipo_Sociedade ==
  "Controlada"` or `Participacao_Emissor >= 50`, a non-null CNPJ, joined to the zip's index
  (`fre_cia_aberta_<year>.csv`, `ID_DOC == ID_Documento`) for `CD_CVM` and `DT_RECEB` (`known_from`).
- `read_patents(badepi_zip: Path) -> pl.DataFrame`: columns `application` Utf8 (stripped of spaces and quotes),
  `cnpj_root`, `filed_on` Date, `known_from` Date (min `DT_PUBLICACAO` of the application's dispatches),
  `fields` List[Int32] (unique `CAMPO_TEC`, sorted; empty list when none). One row per application and legal-entity
  depositor root (`CD_TIPO_PFPJ == "J"`); applications without any of those dispatches are left out (never public).
- `issuer_roots(cad_csv: Path) -> pl.DataFrame`: `code_cvm` Int64, `cnpj_root` from `cad_cia_aberta.csv`
  (`CD_CVM`, `CNPJ_CIA`), one row per code (the latest `DT_REG` if repeated).

- [ ] Tests, one at a time, each on a tiny zip built in `tmp_path` (as `tests/test_fundamentals.py::_zip`):
  `test_subsidiary_is_a_controlled_company_dated_by_the_fre_receipt` (a "Coligada" at 30% is out; a row with null
  type and 100% is in; `known_from` is the index's `DT_RECEB`);
  `test_patent_is_known_from_its_first_publication_and_keeps_its_fields` (dispatches 2.1, 3.1, 1.3 -> the 3.1 date; a
  person depositor is out; padded id `"PI0501246   "` -> `"PI0501246"`; fields `[1, 4]`);
  `test_an_application_never_published_is_left_out` (only 2.1 and 2.10);
  `test_issuer_root_is_the_first_eight_digits_of_the_registry_cnpj`.
- [ ] Implement with `pl.scan_csv(..., separator=";", encoding="utf8-lossy", infer_schema=False)` over members
  extracted to a temporary directory (the despacho table is 400 MB); commit.

### Task A3: `innovation` command and the export

**Files:** Modify `src/b3lake/cli.py` (new `cmd_innovation`, parser entry); Test `tests/test_innovation.py`.

**Interfaces:** `innovation.build(lake_dir: Path, badepi_zip: Path, out_dir: Path) -> dict[str, int]` writes
`intangibles.parquet` (code_cvm, account, dt_refer, value, available_from; via `fundamentals.read_statements` +
`curated_accounts(..., INTANGIBLES)` over the DFP and ITR zips already in `raw/cvm`, no download),
`subsidiaries.parquet`, `issuer_roots.parquet`, `patents.parquet` (only applications whose root is an issuer's or
a subsidiary's) and `ipca.parquet` (month, ipca_pct from `raw/bcb/ipca_433.json`); returns row counts.
CLI: `b3lake innovation --badepi D:/inovemetria_raw/badepi_patentes/patentes.zip`, output
`LAKE_DIR/exports/innovation/`.

- [ ] Test `test_build_keeps_only_patents_of_issuers_and_their_subsidiaries` on tmp files; see it fail; implement.
- [ ] Run the command on the real lake; record the row counts and, for the commit message, how many of the issuers
  have at least one application with and without subsidiaries. Commit; merge is left for the end (Task C5).

## Part B: optimizer signals (`po-inov`)

### Task B1: freeze the folder

**Files:** Create `scripts/inovemetria_montar.py`; folder `docs/dados/2026-11-12-inovemetria/`.

- [ ] The script copies the five exports, keeps intangibles rows with `available_from <= 2026-10-24` and adds
  `issuer`/`sector` from `docs/dados/2026-10-22/sectors.parquet` by `code_cvm`, writes `SHA256SUMS` (the house
  format read by `conferir_hashes`) and `LEIAME.md` (sources, dates, row counts, the spec's known limits). Run;
  commit data and script.

### Task B2: `core/inovemetria/data.py`

**Interfaces (produces):**
```python
FOLDER = RAIZ / "docs" / "dados" / "2026-11-12-inovemetria"

@dataclass(frozen=True)
class InnovationData:
    fundamentals: pd.DataFrame   # the 2026-10-24 accounts + the frozen `intangibles` rows, same schema
    patents: pd.DataFrame        # application, cnpj_root, filed_on, known_from, fields
    subsidiaries: pd.DataFrame   # issuer, cnpj_root, dt_refer, known_from
    roots: pd.Series             # issuer -> cnpj_root
    activity: pd.Series          # issuer -> CVM activity sector (setor_ativ)
    ipca: pd.Series              # month start -> index level (cumulative product of 1 + pct/100)

def load(folder: Path = FOLDER) -> tuple[InnovationData, dict[str, str]]   # data, sha256 of each file
def issuer_patents(data: InnovationData, month: pd.Timestamp) -> pd.DataFrame
    # issuer, application, known_from, fields: applications known before `month` of each issuer's own root and of
    # the controlled set of its latest FRE known before `month`; an application counts once per issuer
```
- [ ] Tests (`tests/test_inovemetria_data.py`): `test_a_subsidiarys_patent_counts_only_after_the_fre_that_names_it`,
  `test_the_controlled_set_is_the_latest_fre_before_the_month` (a company dropped from the newer FRE stops
  counting), `test_an_application_of_parent_and_subsidiary_counts_once`. One at a time; implement; commit.

### Task B3: `core/inovemetria/diffusion.py`

**Interfaces (produces):**
```python
def step_exponential(x, r)            # x * exp(r)
def step_logistic(x, q, m)            # m x e^q / (m + x (e^q - 1))
def step_bass(x, p, q, m)             # u = (p m + q x)/(m - x); u' = u e^{p+q}; m (u' - p)/(u' + q)
def step_gompertz(x, r, m)            # exp(log m + e^{-r} (log x - log m))

@dataclass(frozen=True)
class Fit:
    law: str                          # "exponential" | "logistic" | "bass" | "gompertz"
    params: tuple[float, ...]
    aicc: float

def fit_laws(before: np.ndarray, after: np.ndarray) -> Fit      # the law of smallest AICc on log(after)
def forecast(fit: Fit, x: float, steps: int = 4) -> float       # x after `steps` quarters under the law
def stage(fit: Fit, x: float) -> str                            # the spec's labels
def simulate_bass_agents(n: int, p: float, q: float, steps: int, dt: float, rng) -> np.ndarray  # adopters/n
```
AICc with n pairs and k = parameters + 1 for the variance (k = 2, 3, 4, 3): `n log(RSS/n) + 2k + 2k(k+1)/(n-k-1)`. Bounds as the spec;
starts: m from `2 max x`, q, r from the mean log growth clipped into bounds, p from 0.01; `x == m` in the Bass
map returns m.

- [ ] Tests (`tests/test_inovemetria_diffusion.py`), one at a time: each step function against a fine Euler
  integration of its ODE over one quarter (rtol 1e-4); `test_bass_with_no_innovation_is_the_logistic`;
  `test_fit_recovers_a_noiseless_logistic_and_its_forecast` (law "logistic" or "bass" with forecast within 1e-3 of
  the truth); `test_a_constant_growth_series_is_exponential`; `test_a_series_above_its_ceiling_forecasts_a_decline`;
  `test_agents_converge_to_the_bass_curve` (n = 20000, max gap < 0.02); `test_stage_labels` (u 0.05 emergence,
  0.3 take-off for the logistic, 0.7 maturing, 0.95 saturation, x > m decline). Implement; commit.

### Task B4: `core/inovemetria/signals.py`

**Interfaces (produces, each returns `core.valor.NotaDoMes`):**
```python
def diffusion_score(data: InnovationData) -> NotaDoMes            # S
def encroachment_score(data: InnovationData) -> NotaDoMes         # E
def effort_score(data: InnovationData, sectors: Mapping[str, str]) -> NotaDoMes   # B (sectors: ticker -> sector)
def patent_score(data: InnovationData) -> NotaDoMes               # P (month clamped to 2026-01-01)
def convexity_score(monthly_returns: pd.DataFrame) -> NotaDoMes   # X
def families(data, sectors, monthly_returns) -> dict[str, NotaDoMes]   # {"S", "E", "B", "P", "X"}
def revenue_quarters(fundamentals, month, ipca=None) -> pd.DataFrame   # dt_refer x issuer, TTM revenue (real if ipca)
def diffusion_table(data, month) -> pd.DataFrame   # issuer: law, stage, forecast, x_over_m (the map uses it)
def field_counts(data, month) -> pd.DataFrame      # issuer x 35 WIPO fields, P1's 60-month window
```
Values per ticker through `core.risco.emissor`, as `core.crescimento_lucro.nota_da_variacao`. `diffusion_score`
memoizes fits by issuer and the bytes of its series. B's sector median uses the candidates' sectors; "Financeiro"
out (`core.valor.SETOR_FORA`).

- [ ] Tests (`tests/test_inovemetria_signals.py`), one at a time, on small hand-made frames:
  S: `test_diffusion_score_is_the_forecast_log_growth_and_ignores_later_filings`,
  `test_no_diffusion_score_with_a_gap_or_a_non_positive_quarter`;
  E: `test_encroachment_is_the_slope_of_the_logit_share` (share rising 10% -> 20% over 12 quarters in a 3-issuer
  sector: slope = (logit 0.2 - logit 0.1)/2.75 per year), `test_a_new_listing_does_not_change_the_incumbents_share`;
  B: `test_effort_is_the_change_over_lagged_assets_minus_the_sector_median`, `test_financials_have_no_effort_score`;
  P: `test_patent_intensity_and_acceleration` (3 applications in the last 24 months, 1 in the 24 before, assets
  R$ 2 bn: P1 = 2.0, P2 = 0.4), `test_an_issuer_without_patents_scores_zero_not_missing`;
  X: `test_convexity_is_positive_for_an_accelerating_path_and_zero_sign_for_a_straight_one`.
  Implement after each; commit per family.

### Task B5: `core/inovemetria/kernel.py`

**Interfaces (produces):**
```python
def technology_kernel(counts: np.ndarray) -> np.ndarray     # n x 35 -> n x n, cosine on patenting rows, identity elsewhere
def mixed_correlation(fields_of_month: Callable[[pd.Timestamp], pd.DataFrame], weight: float = 0.5,
                      correlation: Correlation = shrunk_correlation) -> Correlation
    # treino -> (1 - weight) C_returns + weight K, K on treino's columns through core.risco.emissor
```
- [ ] Tests (`tests/test_inovemetria_kernel.py`): `test_kernel_is_a_correlation_matrix_on_random_counts` (unit
  diagonal, symmetric, smallest eigenvalue >= -1e-10, 200 random cases with zero rows),
  `test_no_patents_gives_the_identity`, `test_the_mix_is_a_correlation_matrix_and_lowers_the_determinant_of_two_same_field_names`.
  Implement; commit.

## Part C: readings

### Task C1: commit the spec amendments, then stage 1

**Files:** Create `scripts/inovemetria_ic.py` (pattern: `scripts/crescimento_lucro_avaliar.py`).

- [ ] For S, E, B, P, X, INOV: `avaliar_valor.ics_e_cobertura`, `resumo_do_ic` over the gate period and the
  subperiods, coverage, `PORTAO.aprova`, and the monthly rank correlation with the Conservadora, value and
  profit-growth scores. Prints the list of families that pass (INOV-gate's members).
- [ ] Run once: `... scripts/inovemetria_ic.py > docs/inovemetria-ic-2026-11-12.txt`; commit output and script.

### Task C2: stage 2

**Files:** Create `scripts/inovemetria_avaliar.py` (pattern: `scripts/fermionica_grade_avaliar.py`).

- [ ] `PASSING: tuple[str, ...]` is a constant typed from Task C1's committed output. Anchors (Composite replay
  1e-9; LogDet-15 t0.5 against `docs/dados/fermionica-grade-*` curves 1e-9), then the seven arms per phase with
  `metricas`, alpha and t against the Composite, shared names, pair correlation, and for arms 6 and 7 the number of
  decisions whose picks differ from the same arm with the returns' correlation alone (taken from `registro`), and arm 6
  against the control arm 8 (`mixed_correlation` with K = I). Eight rows appended to the attempts file.
  Curves saved to `docs/dados/inovemetria-curvas-2026-11-12.parquet`.
- [ ] Run once: `> docs/inovemetria-2026-11-12.txt`; commit.

### Task C3: the map

- [ ] `scripts/inovemetria_mapa.py` writes `docs/inovemetria-mapa-2026-11-12.csv` for the month 2026-10-01 with
  the spec's columns; commit.

### Task C4: record

- [ ] DECISOES entry (next number after the last on `main`), README line if the house pattern asks for one,
  `session_state`; full test suite of both repos; `graphify update .` in both.

### Task C5: finish

- [ ] superpowers:finishing-a-development-branch for both branches (lake first).
