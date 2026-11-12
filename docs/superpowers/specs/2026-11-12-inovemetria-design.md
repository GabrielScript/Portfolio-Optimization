# Inovemetria: innovation and diffusion signals for the full universe

Date: 2026-11-12. Branch `feat/inovemetria` (worktree `../po-inov`). The user: "usar técnicas de Inovemetria ...
Rogers' Diffusion Curve & The Bass Model, Technology-Utility and Market Encroachment Models, Stage-Structured and
ODE Models, Agent-Based Modeling ... ou adicionar algo quântico ... avaliar os +200 ativos pelas principais métricas
de fomento a práticas e investimento em P&D ... crescimento acelerado da empresa e do mercado ... ajudaria a
fermiônica ou a entropia de von Neumann ou até nossa estratégia principal?"; success, in the user's words: "que
passe no portão contra o Composite". **Pre-registration: committed before any return-based number of these signals
or arms exists.** Seen before this commit, and nothing else: patent counts per issuer (116 of 309 active listed
issuers have at least one filing under their own CNPJ root) and how many issuers file each CVM account.

## Why

Every arm since attempt 51 reorders the same three components (Conservadora, value, profit growth). The Fermiônica
chooses among every scored candidate, but with the Composite's score, so it holds 13 to 14 of the Composite's 15
names. A signal built from different information (patents, intangible investment, where the firm is on its growth
curve, share taken from sector peers) is the only way those rules can hold a different portfolio. Whether that
portfolio is better is what this measures.

## Goals, non-goals, anti-goals

- Goal: five signal families scored on every issuer of the lake, point in time; each read once against the house
  IC gate; eight portfolio arms (one a control) read once against the Composite's gate.
- Goal, whatever the gate says: a disruption map of today's candidates (one row per name, every family's value).
- Non-goals: no app screen, no paper log, no fictitious account, nothing in the real buy of 19/12; no US data.
- Anti-goals: a signal dated by anything the market did not have at the decision; choosing families, windows or
  the sign after seeing returns; reading an arm twice.

## Data

Returns, universe, sectors, engine, costs, phases 0 / 21 / 42 and the Composite score: unchanged (frozen exports to
16/10/2026 in `docs/dados/2026-10-22`, CVM folder `docs/dados/2026-10-24-cvm`, as DECISOES 350). New, frozen with
hashes in `docs/dados/2026-11-12-inovemetria/`:

- `intangibles.parquet`: CVM account 1.02.04.01 (intangibles without goodwill) of the balance sheet, a stock, in
  the schema of `fundamentals.parquet` (issuer, account `intangibles`, dt_refer, value, available_from), read by
  the lake's own statement code (`curated_accounts`, standardized codes, consolidated first, dated by receipt) into
  a separate export: the nightly's `fundamentals.parquet` and its `ACCOUNTS` are not changed. Only the rows with
  `available_from` up to 2026-10-24 are frozen here. Every other account keeps coming from the 2026-10-24 folder.
- `patents.parquet`: one row per patent application and issuer: application number, issuer, `filed_on`,
  `known_from` (the earliest RPI date among the application's publication dispatches: 3.1 "publicação do pedido",
  1.1 "pedido internacional PCT publicado", 1.3 "PCT admitido na fase nacional"; the filing notices 2.1 and 2.10
  come months earlier but are not the published application; an application with none of the three was never
  public and is left out), the WIPO technology fields (`CAMPO_TEC`, 35 fields; today's classification, back-dated:
  an accepted limit for the kernel and the map). The last dispatch in the snapshot is of 30/12/2025. Source: INPI BADEPI v12.0 (snapshot of 13/10/2026,
  2000 to 2025, copy in `D:\inovemetria_raw\badepi_patentes\patentes.zip`). An application belongs to an issuer
  when a legal-entity depositor's CNPJ root (8 digits) is the issuer's or that of a company the issuer controls.
- `subsidiaries.parquet`: issuer, CNPJ root of the controlled company, `known_from`. Source: the CVM FRE table
  `participacao_sociedade` (2010+, already in the lake's raw files): rows with `Tipo_Sociedade` "Controlada" or
  `Participacao_Emissor` >= 50, dated by the FRE document's receipt at the CVM. At a month, an issuer's controlled
  set is that of its latest FRE received before the month.
- `ipca.parquet`: the monthly IPCA index (BCB series 433, the lake's raw file), to deflate revenue.

Built by a new lake command (`innovation`, branch `feat/innovation-data` of `b3-datalake`, worktree
`../../b3-datalake-inov`, exports in `D:\b3-datalake\exports\innovation\`) and copied by
`scripts/inovemetria_montar.py`, which writes `SHA256SUMS` and `LEIAME.md`.

Checked and left out of the scores: PINTEC (triennial to 2017 and Semestral 2021-2024), the Lei do Bem open file
(2015-2021), CNI's Atlas, CNC's survey, FNDCT projects and FORMICT are sector, region or university aggregates with
no company identifier; the R&D expense line of the income statement is filed by about 14 issuers.

Known limits, accepted: BADEPI is one snapshot, so the depositor is today's holder (assignments after filing are
back-dated); goodwill that an issuer files inside 1.02.04.01 counts as intangible investment; concession assets
(utilities, toll roads) are intangibles by accounting rule, which the sector-relative reading below reduces and
does not remove; the financial sector has another chart of accounts and gets no score in family B.

## The five families

Each is a `NotaDoMes` (month, candidates -> value per ticker; an issuer's classes share it), from documents with
`available_from` before the month (`core.valor.documentos_conhecidos`) and patents with `known_from` before the
month. Higher is always "more innovative / more accelerated": the sign is the thesis and is not flipped later.

### S. Diffusion stage (Bass, logistic, Gompertz)

x_t: the issuer's trailing-twelve-month revenue at each quarter end known before the month, in constant reais
(IPCA), the last 40 quarter ends at most; at least 12 pairs of consecutive quarters, all x > 0. Four autonomous
growth laws, each as its exact one-quarter map x_{t+1} = f(x_t), fitted by least squares on log x_{t+1} over the
consecutive pairs:

- Exponential (no ceiling): f(x) = x e^r.
- Logistic (Verhulst; Fisher-Pry; Bass with p = 0), dx/dt = q x (1 - x/m): f(x) = m x e^q / (m + x (e^q - 1)).
- Bass, dx/dt = (p + q x/m)(m - x): with u = (p m + q x)/(m - x), u_{t+1} = u_t e^{p+q} and
  x_{t+1} = m (u_{t+1} - p)/(u_{t+1} + q). p is innovation (growth with no installed base), q imitation.
- Gompertz, dx/dt = r x log(m/x): log x_{t+1} = log m + e^{-r} (log x_t - log m).

AICc = n log(RSS/n) + 2k + 2k(k+1)/(n-k-1), n the pairs, k the law's parameters plus one for the residual
variance (2, 3, 4, 3). Bounds per quarter: r in [-0.5, 0.5] for the exponential, r, q in [1e-4, 2], p in [0, 0.5], m in
[0.5 min x, 100 max x] (a ceiling below today's revenue is a decline towards it). The law of smallest AICc is the
issuer's law at that month. The value: **S = log(x_hat_{t+4} / x_t)**, the chosen law's forecast of real revenue
growth over the next four quarters, iterating its map from the last known quarter. A forecast is used because it
stays stable where the ceiling m is poorly identified, which is every firm before its inflection.

Stage, descriptive (the map only): u = x/m of the chosen law against its inflection (logistic 1/2, Gompertz 1/e,
Bass (q - p)/(2q), none if q <= p): emergence u < 0.1, take-off up to the inflection, maturing up to 0.9,
saturation above, decline when x > m; exponential law: unbounded growth (r > 0) or decline.

The agent-based model is the test oracle, not a signal: N agents, each non-adopter adopting in a step dt with
probability (p + q adopters/N) dt, must converge to the Bass map as N grows (`tests`, tolerance of order
1/sqrt(N)). With quarterly statements there is nothing to calibrate an agent model per company on.

### E. Market encroachment (Fisher-Pry substitution)

f_t: the issuer's share of the trailing-twelve-month revenue of its CVM activity sector (`setor_ativ`). The
quarters are the issuer's own last 12 quarter ends known before the month; the peers are the month's candidates
(the tradable universe, which keeps listed subsidiaries of a candidate from counting twice only as far as the
universe does) of the same `setor_ativ` with revenue known before the month at all 12 of those quarter ends; at
least 3 issuers with the issuer itself, the same set at every quarter, so a listing or a late filer is not a share
change. All 12 revenues positive. **E = the least-squares slope of log(f/(1 - f)) on time in years**: the Fisher-Pry
takeover rate.

### B. Innovation effort from the balance sheet

At the latest period end t known before the month and the same quarter a year earlier, over total assets a year
earlier (as `core.crescimento_lucro.variacao_anual`):

- B1 = (intangibles_t - intangibles_{t-1y}) / total_assets_{t-1y}: knowledge investment.
- B2 = (fixed_assets_t + intangibles_t - fixed_assets_{t-1y} - intangibles_{t-1y} + |depreciation_t|) /
  total_assets_{t-1y}: gross investment in fixed and intangible assets (the capex identity; depreciation is the
  DVA's twelve-month depreciation, amortization and depletion, filed negative in most rows, so its absolute value).

Each minus the median of the month's candidates in the same broad sector (the universe's median when the sector
has fewer than 3 scored names); financial sector out. **B = the mean of the normal scores of B1 and B2.**

### P. Patents

For the issuer and its controlled set at the month, applications with `known_from` before the month:

- P1 = applications first published in the 60 months before the month, per billion reais of total assets a year
  before the latest known period end: patent intensity.
- P2 = (n_recent - n_before) / (1 + n_recent + n_before), with n_recent the count first published in the last 24
  months and n_before in the 24 months before those: acceleration, in [-1, 1].

An issuer with total assets and no application has P1 = P2 = 0 (the tie takes the mean rank). The snapshot ends
on 30/12/2025, so from 2026-01 on P (and the kernel) use the month 2026-01 instead of the decision month: the
windows are not truncated, the score is held for the nine months of 2026.
**P = the mean of the normal scores of P1 and P2.**

### X. Price convexity

On the issuer's cumulative monthly total log return over the 36 months before the month (at least 24): the
t-statistic of the quadratic coefficient in the least-squares fit on a constant, t and t^2. **X** is that
t-statistic: a path that bends upward (growth of the growth rate). The slope itself is momentum and is already in
the Conservadora.

### The combined score

**INOV = `nota_media_dos_escores(S, E, B, P, X)`**: the mean of the five normal scores, with those that exist.
S and E need 13 and 12 quarter ends, so the earliest months have none (S from about 2014, E from about 2014): INOV
is then the mean of B, P and X, and the IC of S and E is read over the months that have them (an empty month is
left out of the t, never counted as zero).
**COMP+INOV**: the mean of the normal scores of the Composite's three components and INOV (four components, the
same code as attempt 51's score, INOV as the fourth).

## Stage 1: the IC of each family (read once, before stage 2 is run)

`avaliar_valor.ics_e_cobertura` (monthly Spearman IC, score of the month against the month's total return, the
month's universe, 2012-01 to 2026-09), the house gate IC >= 0.02 and t >= 2, for S, E, B, P, X and INOV. Reported:
subperiods, months with IC > 0, names scored per month, and the monthly rank correlation of each with the
Conservadora, value and profit-growth scores. A negative IC with |t| >= 2 is reported as the thesis being wrong in
sign; using the opposite sign would be a new pre-registration.

**INOV-gate** = the mean of the normal scores of the families that pass their gate. Choosing them on the same
sample that the arms are then read on flatters the arm; it is the house's rule since DECISOES 161 and it is said
again beside the result.

## Stage 2: eight arms (15 names, 4 per sector, equal weights, house rules)

1. **Inov-15**: `estrategia_por_nota(INOV)`. The thesis alone.
2. **Comp+Inov-15**: `estrategia_por_nota(COMP+INOV)`.
3. **Comp+InovGate-15**: the same with INOV-gate in place of INOV. Not run when no family passes or all five do
   (then it is arm 2); it counts in N either way.
4. **Fermi-Inov t0.5**: `estrategia_fermionica`, log det, theta 0.5, z from COMP+INOV, C the returns' correlation.
5. **vonNeumann-Inov t0.5**: arm 4 with the von Neumann functional.
6. **Fermi-Tech t0.5**: log det, theta 0.5, z from the Composite's score, C = (C_returns + K)/2.
7. **Fermi-Inov-Tech t0.5**: z from COMP+INOV and C = (C_returns + K)/2.
8. **Fermi-Half t0.5**: the control of arm 6, z from the Composite's score and C = (C_returns + I)/2. For a pair
   without patents K is the identity, so arm 6 halves their correlation too; that alone weakens the diversity term
   (like a larger theta). Arm 6 is read against arm 8: the difference is the technology, not the shrinkage.

K, the technology kernel: v_i the issuer's count of applications per WIPO field among those of P1 (60 months; an
application adds 1 to each distinct field it has);
K_ij = cos(v_i, v_j) when both have applications, 0 otherwise, K_ii = 1. K is the Gram matrix of unit vectors on
the patenting block and the identity elsewhere, so K and the half-and-half mix are correlation matrices (positive
semidefinite, unit diagonal; `tests` check it on random cases). In the determinant, two names then repel when they
move together or when they patent in the same fields.

Anchors per phase, before any arm is read: the Composite's replay is its published curve (1e-9); LogDet-15 t0.5
reproduces its saved curve of DECISOES 350 (1e-9).

Gate (unchanged): candidate only if P[arm > Composite] >= 0.95 and DSR >= 0.95 in all three phases. A pass is at
most a forward fictitious account, opened by a separate decision. **N 109 -> 117.**

Read besides the gate, fixed now: names shared with the Composite and mean pairwise correlation of each arm; alpha
and its t against the Composite; for arms 6 and 7, how many decisions the kernel changes a pick in; arm 6 against
arm 8 (alpha, t, P[6 > 8]).

## The disruption map

`docs/inovemetria-mapa-2026-11-12.csv`, the candidates of the last decision month of the frozen data: ticker,
sector, the five values and their normal scores, INOV, the chosen growth law, stage, forecast real growth, x/m,
share slope, applications in 60 months and the leading WIPO field. Descriptive; no decision reads it.

## Expectation (registered)

P: IC within 0.01 of zero (two thirds of the issuers tie at zero; patents here are Petrobras, Vale, capital goods).
B: IC negative or zero (the investment and asset-growth results abroad; our sales-growth reading of DECISOES 159,
where low growth won). S: IC zero or negative for the same reason. E: IC between 0 and 0.02 (share gains overlap
profit growth). X: IC negative (a path bending upward is overextension). INOV does not pass. Inov-15 loses 3 to 8
points a year to the Composite with a deeper drawdown; Comp+Inov-15 loses 0.5 to 2 points; arms 4 and 5 land within
1 point of LogDet t0.5 on either side; the technology kernel changes a pick in fewer than a third of the decisions
and arm 6 is within half a point of its control, arm 8. No arm passes the gate.

## Builder check (calls made after the independent read, before any number)

A fresh reader raised six points; all settled here by me, none changes what the user asked for: the publication
dispatches that date a patent; holding P at 2026-01 after the snapshot ends; E's quarters and peers; the control
arm 8 (N 116 -> 117); S and E empty before about 2014; AICc's k with the variance, |depreciation| in B2, an
application counting once per distinct field in K.

## Left to the builder

Module layout (suggested: `core/inovemetria/` with `difusao.py`, `sinais.py`, `nucleo.py`, `dados.py`; scripts
`inovemetria_montar.py`, `inovemetria_ic.py`, `inovemetria_avaliar.py`, `inovemetria_mapa.py`), the optimizer for
the fits (bounded least squares, deterministic starts), caching the S panel per issuer and known quarter set so a
fit runs once per new statement, and the lake command's internals. Nothing under the paper guards' paths
(`core/market_data.py`, `core/paper.py`, `core/bcb.py`, `assets.py`) is edited.
