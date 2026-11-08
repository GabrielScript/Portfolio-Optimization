# Earnings surprise (SUE) and buyback announcements on the B3 — design and pre-registration

_2026-11-08. Written and committed before any return, IC or curve of these signals exists. The user's goal (08/11):
"execute as estratégias 1 e 2 até o final" — ideas 1 (post-earnings drift from a surprise measured against the
firm's own history, no analyst consensus) and 2 (CVM filing events by type only, no LLM reading) of the 08/11 answer.
Branch `feat/pead-eventos` from `feat/mars` (`5dfd76d`): same engine, frozen data and gate as the MARS, HERC, KAN and
periphery readings. Sections 1 to 7 are fixed from the commit on; anything left open is fixed in an addendum
committed before the number it touches._

## 1. Sources and priors

- **SUE / PEAD.** Bernard and Thomas (1989, 1990): prices under-react to earnings news; the standardized unexpected
  earnings of a seasonal random walk (this quarter minus the same quarter a year before, over the volatility of
  past such changes) predicts returns for about 60 sessions. Replicated widely, including emerging markets (Hou,
  Xue and Zhang 2020 keep SUE among the surviving anomalies; JKP's `niq_su` theme). The Composite already holds a
  slower relative: profit growth (dNI = 12-month net income change over lagged assets, DECISOES 161). SUE differs in
  two ways: one quarter's change, not four, and scaled by the firm's own volatility, not by assets.
- **Buybacks.** Ikenberry, Lakonishok and Vermaelen (1995): open-market repurchase announcements are followed by
  positive long-run abnormal returns; Brazilian studies find a positive announcement reaction and weaker drift.
- **Prior, written now.** SUE passes stage A1 (alone) with about 50%, stage A2 (beyond the Composite) with about
  25%; the SUE arm passes the strategy gate with about 10%. Buyback stage A passes with about 25%, its arm about 8%.
  The likelier outcome for both: a real but small effect that the Composite already captures in part, failing the
  strategy gate.

## 2. Data, frozen

- Statements: `docs/dados/2026-10-24-cvm/fundamentals.parquet` (DFP 2010+, ITR 2011+, 12-month flows, each value
  dated by its receipt at the CVM, `available_from`), the same file as the Composite's value and growth.
- Filings: the CVM IPE index (`dados.cvm.gov.br/dados/CIA_ABERTA/DOC/IPE`, the yearly files 2010 to 2026 already in
  `../../cvm/`), reduced by `core.recompra.e_recompra` (section 4) to the matching rows and frozen with its hash
  in `docs/dados/2026-11-08-ipe/` before the reading. Issuer of a filing = the `issuer` of its `Codigo_CVM` in the
  fundamentals file (one issuer per CVM code); filings of codes without one are dropped and counted.
- Prices, universe, sectors, CDI: `docs/dados/2026-10-22/`, `2026-10-21-macro/`, cut 16/10/2026, as every reading
  since attempt 51.

## 3. Signal 1: SUE (fixed now)

For month m and each issuer, with only values received before m (`available_from < m`):

- NI(q) = the 12-month net income at quarter end q. Then dE(q) = NI(q) - NI(q - 1 quarter) is exactly this quarter's
  income minus the same quarter's a year before (the seasonal random walk's surprise). It needs both quarter ends
  to be consecutive (3 months apart).
- t = the latest quarter end with a dE. **SUE = dE(t) / standard deviation of dE over the 8 quarter ends before t**
  (t-1 to t-8), at least 6 of them present and the deviation > 0. Out otherwise.
- Score of a ticker = its issuer's SUE (an issuer's classes share it), normal-scored among the month's candidates
  (`core.views.escore_normal`), as every component of the Composite.

## 4. Signal 2: buyback announcements (fixed now)

- **Text.** Per IPE row: `Tipo | Especie | Assunto`, lower case, accents removed (NFKD).
- **A buyback row** (`e_recompra`): category in {Fato Relevante, Comunicado ao Mercado, Reunião da Administração,
  Assembleia}; the text matches `recompra`, `aquisicao de acoes de (sua )?propria emissao` or
  `acoes de (sua )?(propria )?emissao (da )?propria`; and it is not an end of a program: rows that match
  `encerramento|conclusao|termino|cancelamento|alienacao` are out unless they also match
  `novo|nova|aprovacao|aprova|renovacao|institui|abertura|inicio`.
- **Equity issues are not tested** (fixed now, from a count only, no return seen): only ~100 IPE offering notices
  can be told apart as shares rather than debt, almost none before 2015 (restricted offers do not pass through this
  index), and net issuance already failed to add to the Conservadora (NPY, DECISOES 176).
- **Event date.** One event per (issuer, `Data_Referencia`); its information date a = the later of
  `Data_Referencia` and the earliest `Data_Entrega` of its rows. Events with a later than `Data_Referencia` + 30
  days are dropped (no clear information date) and counted. **Entry = the close of the first session after a.**
  Per issuer, an event whose entry is within 63 sessions of the previous kept event's entry is dropped (one
  program, many filings).

### Stage A for buybacks (event study, read once)

For each event with entry e from 2011-01-03 and e + 63 sessions on or before 16/10/2026: the issuer's return from the
close of e to the close of e + 63 (the mean over its classes in the universe of e's month with a price at e; a
price that stops is carried forward), minus the equal-weighted mean of the same-window return of every name in
that month's universe with a price at e (same carry). Events whose issuer has no class in that month's universe are
out (counted). **Monthly portfolio:** the mean abnormal return of the events entering in each calendar month;
test over the months with events. **Gate: mean >= +1.0% and Newey-West t (`core.qml_ic.t_newey_west`, its lags)
>= 2.** Reported, not gates: events, months, the 21-session window, the share of positive events, the abnormal
return by subperiod (2011-16, 2017-21, 2022-26), and the median.

## 5. Stage A for SUE (the project's S2 harness, read once)

`scripts/avaliar_valor.ics_e_cobertura`: monthly Spearman IC of the score of month m with the total return of m in
m's universe, 2012-01 to 2026-09.

- **A1:** SUE alone. **Gate: mean IC >= 0.02 and t >= 2** (`avaliar_valor.PORTAO`).
- **A2:** SUE beyond the Composite: each month, the SUE normal score's residual of a cross-sectional OLS on the
  Composite score's normal score (names with both), then the same IC and the same gate.
- Both must pass. Reported: subperiods, months with IC > 0, names scored, rank correlation with the Composite score
  and with profit growth.

## 6. The arms (stage B, only for a signal that passed its stage A)

- **Composto+SUE-15:** the Composite's score with SUE as a fourth equal component
  (`core.qubo_universo.composite_score` over the Conservadora, value, profit growth and SUE), then the Composite
  rule unchanged (`core.composto.estrategia_por_nota`: 15 names, 4 per sector, equal weights, fixed limits).
- **Composto+Recompra-15:** at decision session d, the Composite score's percentile among the names that have one,
  **plus 0.25** for each name whose issuer has a kept buyback event with a < d inside the 63 sessions that end at d;
  then the Composite rule unchanged.
- One configuration each; no search over the window, the 0.25, the 8 quarters or the 63 sessions.
- **Engine and gate** (as the periphery and MARS readings): `walk_forward_dinamico`, `CFG_BT` (1,260 sessions, every
  63, 10 bps a side), phases 0/21/42; the Conservadora must reproduce 17.41/16.41/16.90% and the Composite
  21.4/19.9/19.1% first. **Gate, every phase, all >= 0.95:** P[> Composite], P[> Conservadora], P[> BOVA11], DSR.
  Reported: CAGR, MDD, volatility, turnover, year table, picks shared with the Composite, latest picks.
- **N.** Two configurations: two rows in `docs/tentativas.csv` in this commit, whether or not an arm runs (a stage
  that stops a line still spent the look). The DSR uses the union of every branch's trials, **N = 108** (106 after
  the MARS reading + these two).

## 7. What follows

- An arm that passes: a forward paper log (ghost account in the simulator) is pre-registered next; nothing goes to
  real money from a backtest.
- A signal that fails stage A: its arm is not run; the line is closed. No retuning of the windows, thresholds or
  text rules after a reading.
- Output: `docs/pead-recompra-2026-11-08.txt` and DECISOES entries. Nothing goes to the app.
