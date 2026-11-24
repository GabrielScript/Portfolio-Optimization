# Debenture credit signal on the Composite: implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or
> superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** build the two debenture credit scores (`CRD-N`, `CRD-V`), read their IC once (stage 1), and, only for a
score that passes, read the two Composite arms once (stage 2).

**Architecture:** one research module `core/debentures.py` (SND loading, the monthly measure, the two scores, the
earned weight ported from `feat/informados`, the veto); one stage 1 script `scripts/debentures_ic.py` on the house
IC harness; stage 2 as family `debentures` of `scripts/composto_variantes_avaliar.py`. The live Composite
(`core/composto.py`, `core/qubo_universo.py`) is not changed.

**Tech Stack:** Python 3 (project `.venv`), pandas, pytest.

**Spec:** `docs/superpowers/specs/2026-11-24-debentures-design.md` (commit `7392e64`).

## Global Constraints

- No return-based number is read before its stage's script is committed; each stage is read once.
- SND files stay on `D:/b3-datalake/raw/snd/` and are never committed (DECISOES 110); their hashes are checked
  against `docs/dados/2026-11-24-snd/SHA256SUMS` on every load.
- Measure: DI-linked papers only (characteristics column `indice` == `DI`), weight `quantidade x pu_med`, rows with
  `pct_pu_curva` missing or outside [50, 150] dropped; issuer by CNPJ root (first 8 digits).
- `CRD-N`(M) = measure of M - 1; `CRD-V`(M) = measure of M - 1 minus measure of M - 4. Higher = better; never flipped.
- Stage 1 gate per score: IC >= 0.02 and t >= 2 (`avaliar_valor.PORTAO`), 2012-01 to 2026-09, months with fewer
  than 20 scored names dropped before the gate. Halves 2012-2018 and 2019-2026.
- Stage 2: 15 names, at most 4 a sector, 10 bps a side, 63 sessions, phases 0 / 21 / 42, frozen export
  `docs/dados/2026-10-22`. Anchor: Composite replay CAGR within 0.1 point of 21.8 / 20.3 / 19.2%.
- Earned weight: lambda_M = clip(mean IC S / mean IC Composite, 0, 1) over months before M with both ICs; 0 with
  fewer than 36 such months. Veto: S at or below `quantile(0.10)` (linear) of S among the month's scored universe
  names; a name without S never vetoed; a month with fewer than 20 scored names vetoes nothing.
- Gate: P[arm > Composite] >= 0.95 and DSR >= 0.95 in every phase. DSR N = 202 + arms read.
- Run Python as `../Portfolio-Optimization/.venv/Scripts/python -X utf8`; tests one at a time, each seen failing first.
- Code, comments, docs and commits in US English.

## Review Focus

1. Asset codes padded with spaces (`"AAJR11    "` in the characteristics export): the join must strip both sides,
   or every trade silently loses its indexer. Test in Task 1.
2. A month whose valid rows all have zero or missing `quantidade x pu_med`: no measure for that month (never a
   division by zero or `inf`). Test in Task 2.
3. `CRD-V` when M - 1 exists but M - 4 does not (a new issuer, a gap): missing, not the level. Test in Task 2.
4. A candidate ticker whose issuer has no CNPJ root, or a root with no trade: absent from the score, no crash, and
   two tickers of one issuer (`PETR3`, `PETR4`) get the same value. Test in Task 2.
5. A veto month with exactly 20 scored names and ties at the 10th percentile: names equal to the threshold are
   vetoed (`<=`); 19 names veto nothing. Test in Task 3.

---

### Task 1: SND loading and the characteristics parser

**Files:**
- Create: `core/debentures.py`
- Create: `docs/dados/2026-11-24-debentures/issuer_roots.parquet` (copied from `origin/feat/informados`,
  `docs/dados/2026-11-12-inovemetria/issuer_roots.parquet`, sha256 `38e0f3a3...c429f`), its `SHA256SUMS` and a
  `LEIAME.md` naming that source (CVM registry: CVM code -> CNPJ root)
- Test: `tests/test_debentures.py`

**Interfaces:**
- Produces: `RAW = Path("D:/b3-datalake/raw/snd")`, `SUMS = <repo>/docs/dados/2026-11-24-snd/SHA256SUMS`,
  `read_characteristics(text: str) -> pd.DataFrame` (columns `cod_ativo`, `cnpj_root`, `indice`),
  `load_snd(raw: Path = RAW, sums: Path = SUMS) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, str]]`
  (trades with `data` as datetime64, characteristics, sha256 by file),
  `issuer_roots(code_roots: pd.DataFrame, sectors: pd.DataFrame) -> pd.Series` (issuer -> CNPJ root).

- [ ] **Step 1: Write the failing test** `test_characteristics_parsed_by_column_name`: a text with two preamble
  lines, a header line starting `Codigo do Ativo` (tab separated, columns `Codigo do Ativo`, `Empresa`, `indice`,
  `CNPJ` in that order plus one unrelated column between them) and two rows, one with `"AAJR11    "` and CNPJ
  `"31907982000123"`, indexer `"DI"`. Assert `cod_ativo == ["AAJR11", ...]`, `cnpj_root == ["31907982", ...]`,
  `indice == ["DI", ...]`.
- [ ] **Step 2: Run it, see it fail** (`ImportError`).
  `../Portfolio-Optimization/.venv/Scripts/python -m pytest tests/test_debentures.py -v`
- [ ] **Step 3: Implement `read_characteristics`**: find the header by its prefix, split on tabs, pick columns by
  stripped name, strip values, root = digits of `CNPJ` [:8].
- [ ] **Step 4: Run it, see it pass.**
- [ ] **Step 5: Write the failing test** `test_load_snd_stops_on_a_hash_mismatch`: a temp folder with a tiny
  `snd_trades.parquet` and `caracteristicas/caracteristicas.txt`, a `SHA256SUMS` with a wrong hash for one file;
  `pytest.raises(SystemExit)`. Then `test_load_snd_reads_matching_files`: correct hashes, `data` dtype is
  datetime64 and the returned hashes equal the file's.
- [ ] **Step 6: Run them, see them fail; implement `load_snd`** (hash each file named in `sums`, names relative to
  `raw`; `SystemExit` naming the file on mismatch); run them, see them pass.
- [ ] **Step 7: Write the failing test** `test_issuer_roots_maps_cvm_code_to_issuer`: `code_roots` with codes 10,
  20, 30; `sectors` with issuer `AAAA` (code 10), `BBBB` (code 20), and code 30 absent. Assert result
  `{"AAAA": root10, "BBBB": root20}`. Run, fail, implement, pass.
- [ ] **Step 8: Copy the roots file and commit**

```bash
git show origin/feat/informados:docs/dados/2026-11-12-inovemetria/issuer_roots.parquet > docs/dados/2026-11-24-debentures/issuer_roots.parquet
sha256sum docs/dados/2026-11-24-debentures/issuer_roots.parquet   # must be 38e0f3a3...c429f
git add core/debentures.py tests/test_debentures.py docs/dados/2026-11-24-debentures
git commit -m "debentures: SND loader with hash check, characteristics parser, issuer roots"
```

### Task 2: the monthly measure and the two scores

**Files:** Modify `core/debentures.py`; Test `tests/test_debentures.py`

**Interfaces:**
- Consumes: Task 1's trades and characteristics frames; `core.risco.emissor(ticker) -> str`; `NotaDoMes` from
  `core.valor`.
- Produces: `monthly_measure(trades: pd.DataFrame, characteristics: pd.DataFrame) -> pd.Series` (MultiIndex
  `month` (month start), `cnpj_root`), `credit_score(measure: pd.Series, roots: pd.Series, kind: str) -> NotaDoMes`
  with `kind` in `{"level", "change"}` (`ValueError` otherwise), `LAG_CHANGE = 3`.

- [ ] **Step 1: Failing test** `test_measure_is_volume_weighted_di_only_within_bounds`: root `11111111` in
  2020-03 with rows (qty 1, pu 100, pct 98), (qty 3, pu 100, pct 102), (qty 1, pu 100, pct 160: dropped), (pct NaN:
  dropped), and an IPCA paper of the same root at pct 80 (dropped). Assert measure == 101.0. Run, fail, implement,
  pass.
- [ ] **Step 2: Failing test** `test_measure_skips_a_month_with_no_weight`: the only valid rows have qty 0;
  assert the (month, root) key is absent and no value is `inf`. Run, fail, fix, pass.
- [ ] **Step 3: Failing test** `test_level_score_uses_the_previous_month`: measure for 2020-03 only; score at
  2020-04 for `["AAAA3", "AAAA4", "ZZZZ3"]` (roots: `AAAA` -> `11111111`, `ZZZZ` absent) equals 101.0 for both
  AAAA tickers, ZZZZ3 absent; score at 2020-03 is empty. Run, fail, implement, pass.
- [ ] **Step 4: Failing test** `test_change_score_needs_both_months`: measures 2020-01 = 99, 2020-04 = 103 for one
  root and 2020-04 only for another; at 2020-05 the first is 4.0, the second absent. Run, fail, implement, pass.
- [ ] **Step 5: Commit** `git commit -m "debentures: monthly measure, CRD-N and CRD-V"`

### Task 3: earned weight, weighted score and veto

**Files:** Modify `core/debentures.py`; Test `tests/test_debentures.py`

**Interfaces:**
- Consumes: `core.views.escore_normal`.
- Produces: `MIN_MONTHS = 36`, `earned_weight(ic_extra, ic_base, month, min_months=MIN_MONTHS) -> float` and
  `weighted_score(base, extra, weight_of_month) -> NotaDoMes`, both copied unchanged from
  `origin/feat/informados:core/informados.py` (DECISOES 354); `VETO_QUANTILE = 0.10`, `VETO_MIN_NAMES = 20`,
  `vetoed_names(panel: pd.DataFrame, quantile=VETO_QUANTILE, min_names=VETO_MIN_NAMES) -> dict[pd.Timestamp,
  frozenset[str]]`, `veto_score(base: NotaDoMes, vetoed: Mapping[pd.Timestamp, frozenset[str]]) -> NotaDoMes`.

- [ ] **Step 1: Failing test** `test_earned_weight_is_zero_before_36_months_and_clipped`: 35 months of both ICs ->
  0.0; 40 months with extra mean 0.01 and base 0.02 -> 0.5; extra 0.05 -> 1.0; extra negative -> 0.0; only months
  before `month` count. Run, fail, copy, pass.
- [ ] **Step 2: Failing test** `test_weighted_score_adds_nothing_for_a_name_without_extra`: weight 0.5; a name
  missing from extra gets z(base) only; with weight 0 the result equals `escore_normal(base)`. Run, fail, copy, pass.
- [ ] **Step 3: Failing test** `test_veto_threshold_ties_and_minimum`: a panel row with 21 names valued 1..21
  (linear 10th percentile = 3.0 exactly) plus two NaN names vetoes exactly the three names valued 1, 2, 3; a row
  with 19 valued names vetoes nothing; a row with exactly 20 vetoes those <= `row.quantile(0.10)`. Run, fail,
  implement, pass.
- [ ] **Step 4: Failing test** `test_veto_score_drops_only_vetoed_candidates`: base over 5 names, vetoed {B} at
  that month and {} at another; result equals base without B, and base unchanged in the other month. Run, fail,
  implement, pass.
- [ ] **Step 5: Commit** `git commit -m "debentures: earned weight (ported from feat/informados) and the veto"`

### Task 4: stage 1 script, then its one reading

**Files:** Create `scripts/debentures_ic.py`; output `docs/debentures-ic-<run date>.txt`

**Interfaces:**
- Consumes: Tasks 1-2; `avaliar_robustez.carregar`, `avaliar_conservadora.carregar_mensal(FOLDER)`,
  `avaliar_valor.ics_e_cobertura`, `avaliar_valor.PORTAO`, `avaliar_conservadora.resumo_do_ic`,
  `core.poder.ic_por_mes`, `core.valor.painel_da_nota`, `qubo_universo_avaliar._value_score(mensal,
  com_crescimento=True)` (the Composite score), `scripts.qpo_montar.FOLDER`.
- Produces: `MIN_NAMES = 20`, `HALVES = {"2012-2018": ("2012-01-01", "2018-12-01"), "2019-2026": ("2019-01-01",
  "2026-09-01")}`, `gated_ics(ics: pd.Series, cobertura: pd.Series, min_names=MIN_NAMES) -> pd.Series`
  (Task 5 imports it, the same months feed lambda), `scores() -> dict[str, NotaDoMes]` keyed `"CRD-N"`, `"CRD-V"`.

- [ ] **Step 1: Failing test** (in `tests/test_debentures.py`) `test_gated_ics_drops_thin_months`: months with
  coverage 19 dropped, 20 kept, NaN ICs dropped. Run, fail, implement in the script, pass.
- [ ] **Step 2: Write `main()`** following `origin/feat/informados:scripts/informados_ic.py`: print all hashes
  (frozen export, CVM, SND, roots); per score print months kept and dropped, coverage min / median / max,
  `resumo_do_ic` for the gate window and the two halves, mean rank correlation with the Composite score, and
  `PASSED` / `not passed` from `PORTAO.aprova(gated)`.
- [ ] **Step 3: Dry check without reading**: `python -c "import scripts.debentures_ic"` imports cleanly; full suite
  `python -m pytest tests/test_debentures.py tests/test_composto_variantes.py -q` passes.
- [ ] **Step 4: Commit and push the script before the reading**
  `git commit -m "debentures: stage 1 script (no number read)"`, `git push`.
- [ ] **Step 5: Read once**
  `../Portfolio-Optimization/.venv/Scripts/python -X utf8 scripts/debentures_ic.py > docs/debentures-ic-<run date>.txt`
- [ ] **Step 6: Record and commit**: DECISOES entry with both readings; if neither passes, N stays 202, the line
  stops here (no Task 5). `git commit -m "debentures: stage 1 read once"`.

### Task 5 (only for scores that passed): stage 2 arms, then their one reading

**Files:** Modify `scripts/composto_variantes_avaliar.py`; Test `tests/test_composto_variantes.py`; output
`docs/debentures-<run date>.txt`, curves `docs/dados/debentures-curvas-<run date>.parquet`

**Interfaces:**
- Consumes: Task 3's `earned_weight`, `weighted_score`, `vetoed_names`, `veto_score`; Task 4's `scores`,
  `gated_ics`.
- Produces: `Arm.score: str = "comp"` (new field; existing families unchanged); `debenture_arms(passing:
  Sequence[str]) -> dict[str, Arm]` giving, per passing S in spec order, `f"Comp+{S}-w"` (score key
  `f"w:{S}"`) then `f"Comp-veto-{S}"` (score key `f"veto:{S}"`); CLI `composto_variantes_avaliar.py debentures
  CRD-V [CRD-N]`; `RUNNING_N["debentures"] = 202`, `NEW_ROWS["debentures"] = len(arms)`,
  `BASELINE_CAGR["debentures"] = {0: 0.218, 21: 0.203, 42: 0.192}` with tolerance 0.001 for this family.

- [ ] **Step 1: Failing test** `test_debenture_arms_order_and_keys`: `debenture_arms(["CRD-V"])` ->
  `{"Comp+CRD-V-w": Arm(CFG_BT, score="w:CRD-V"), "Comp-veto-CRD-V": Arm(CFG_BT, score="veto:CRD-V")}`;
  an unknown score name raises `ValueError`. Run, fail, implement, pass.
- [ ] **Step 2: Wire `main`**: for family `debentures` build the score dict once (Composite memoized; lambda per
  month from `gated_ics` of S and the Composite's ICs; the veto map from `painel_da_nota(S, universo, months to
  LAST_SESSION, columns)`); the arm loop uses `notas[arm.score]`; also print lambda by year and the share of
  decisions where the veto removed a Composite pick.
- [ ] **Step 3: Existing families untouched**: `python -m pytest tests -q -k "composto_variantes or debentures"`
  passes.
- [ ] **Step 4: Commit and push before the reading** `git commit -m "debentures: stage 2 arms (no number read)"`.
- [ ] **Step 5: Read once**: `... scripts/composto_variantes_avaliar.py debentures <passing> > docs/debentures-<run date>.txt`
  (stops on a failed anchor before any arm is printed).
- [ ] **Step 6: Record**: rows in `docs/tentativas.csv`, DECISOES entry (N 202 + arms), memory and session_state;
  commit and push.
