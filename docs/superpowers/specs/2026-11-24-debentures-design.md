# The company's own debentures as a credit signal on the Composite (pre-registered 2026-11-24)

Written and committed before any return-based number. Branch `feat/debentures`, from `main` (`92781b3`, the
Composite after the growth-age fix, DECISOES 360). The user, 2026-11-24, after the stock-lending history proved
unavailable (DECISOES 354, 373): "vamos focar em informação nova"; on the debentures: "se esses dados da debentures e
SND atrelado ao composite melhorar o nosso retorno está valendo".

Why this is new information: the credit market prices the same firm the stock does, with mostly institutional
holders. No score of ours reads it. The market-wide credit veto failed (IDA-DI minus CDI, DECISOES 112): this test is
cross-sectional, one firm against another, not a timing rule.

## Data looked at (no return involved)

- **SND secondary market** (debentures.com.br, ANBIMA), the user's authorization of 2026-11-24 for the full download.
  Every traded debenture and day, 2004-01-02 to 2026-11-23, active and matured (`op_exc=Nada`): 939,223 rows,
  1,439 issuers. Per row: date, issuer, asset code, ISIN, quantity, number of trades, minimum / mean / maximum unit
  price and **"% PU da Curva"** (the mean price as a percentage of the paper's own accrual curve at its issue
  rate). Two sessions answer SND's internal server error to every query (2020-12-16, 2021-04-09) and are missing.
- **SND characteristics** (same site, one export, 10,091 debentures): asset code, issuer CNPJ, indexer (6,551 DI,
  1,835 IPCA, ...), percentage of the indexer, guarantee, maturity, Lei 12.431 flag. 3 of 939,223 trade rows have no
  characteristics row.
- Stored on `D:/b3-datalake/raw/snd/` (LEIAME, SHA256SUMS, the download scripts). **The SND files are never committed**
  (DECISOES 110: the repository is public); `docs/dados/2026-11-24-snd/` keeps only the hashes and a LEIAME.
- **Coverage** (an issuer counts in a month when one of its own debentures traded in it; issuer to lake by the CNPJ
  root, 8 digits, through CVM's registry): 268 listed issuers ever. DI-linked only, monthly mean of covered universe
  names: 27 (2011), 34-37 (2012-2013), 33-34 (2014-2015), 26 (2016-2017), 34 (2018), 43 (2019), 53-66 (2020-2021),
  86-99 (2022-2026). First month with 30 or more: 2011-11; with 40 or more: 2012-08 for all indexers, 2019 for DI.

## The credit measure

For an issuer and a calendar month m: the mean of "% PU da Curva" over every trade row of the issuer's own
**DI-linked** debentures (characteristics `indice` = DI) dated in m, weighted by quantity x mean unit price. Rows
with "% PU da Curva" missing or outside [50, 150] are dropped (misprints and defaulted papers trading on recovery,
which the measure cannot rank). No row in m: no measure for m.

Why DI only: a DI-linked paper's "% PU da Curva" moves with its credit spread alone; an IPCA-linked or fixed-rate
paper's also moves with the real or nominal rate, which ranks firms by duration, not credit. Why the issuer's own
debentures only: a subsidiary's paper (a concession of a listed group) has its own credit; mapping subsidiaries to
parents is a later choice, to be amended only before any number.

## Scores (known at the first session of the decision month M: trades through the last session of M - 1)

- **`CRD-N` (level):** the measure of M - 1.
- **`CRD-V` (change):** the measure of M - 1 minus the measure of M - 4 (three months); missing if either is.

Sign fixed now and never flipped: a higher value is healthier credit and a higher score. Characteristics are
static terms of each paper (CNPJ, indexer), not information from the future; a CNPJ changed by a merger keeps its
old root (reported, not fixed).

## Stage 1: does either score predict the month's return? (one reading)

The house IC harness (`scripts/avaliar_valor.ics_e_cobertura`): monthly Spearman IC of the score against the
month's total return in the month's universe, 2012-01 to 2026-09. A month with fewer than 20 scored names is dropped
before the gate (the number of months kept is reported). **Gate per score: IC >= 0.02 and t >= 2.** Described, not
gated: coverage, the halves 2012-2018 and 2019-2026, and the rank correlation with the Composite score.

If neither score passes, the project stops here and the result is recorded; N grows by none.

## Stage 2: arms on the Composite (only for scores that pass stage 1; each read once)

For each passing score S, two arms (15 names, at most 4 a sector, the house limits, 10 bps a side, 63-session
rebalance, phases 0 / 21 / 42, the frozen export of attempt 51; `scripts/composto_variantes_avaliar.py` family
`debentures`):

1. **`Comp+S-w` (earned weight):** score = z(Composite) + lambda_M x z(S), over the names with a Composite score, z
   the house normal score among them, a name without S adding 0. lambda_M = clip(mean IC of S / mean IC of the
   Composite score, 0, 1), both means over the same months before M, and 0 while fewer than 36 such months exist
   (DECISOES 354's rule, unchanged).
2. **`Comp-veto-S`:** the Composite after removing, at each decision, the names whose S is at or below the 10th
   percentile of S among the universe names scored that month (pandas `quantile(0.10)`, linear). A name without S
   is never vetoed; a month with fewer than 20 scored names vetoes nothing.

Reference: the Composite replayed in the same run. **Anchor:** its CAGR within 0.1 point of the current baseline
21.8 / 20.3 / 19.2%, before any arm is read.

**Gate (the house's):** P[arm > Composite] >= 0.95 and DSR >= 0.95 in every phase. **N:** 202 (DECISOES 387) plus
the arms read (0, 2 or 4); the DSR uses that count. A pass is at most a forward fictitious account; the app, the MT5
demo and the real book change only by the user's decision.

**Reported:** CAGR, Sharpe, max drawdown, worst calendar year, alpha against the Composite, P[> Composite], DSR,
turnover, names shared with the Composite, how many decisions each arm changes, and lambda over time.

## Prior, stated before the data

`CRD-V` has the better published support (credit deterioration leads the stock, mostly before distress); `CRD-N`
mixes each paper's age with its credit and is the weaker of the two. Coverage is a third to a half of the universe,
so a veto touches at most one or two of the 15 names in a typical decision. Most likely reading: `CRD-V` passes
stage 1 with a small IC or neither passes; no arm passes the gate (the veto ties, as the CVM-filing veto did,
DECISOES 384).
