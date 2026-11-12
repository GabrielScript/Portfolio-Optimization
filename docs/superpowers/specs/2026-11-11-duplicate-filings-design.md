# Refiled statements: one value per issuer, account and period end

Date: 2026-11-11. Branch `fix/duplicate-filings`. The user, after DECISOES 346: "Corrija os 229 balanços duplicados".
**Rule and measurement fixed here, before the corrected Composite is run.**

## The defect (inputs only)

The lake's `fundamentals.parquet` carries 229 (issuer, account, period end) groups with two rows, all of twelve
issuers (CIIT, CCHI, AZZA, ECOM, TEND, ESTR, ORVR, RIOS, TRAD, SQIA, PMET, BIDI), period ends in 2011-2012, 2014-2016
and 2021-2023; 458 of 417,930 rows. They are refilings of the same statement: 180 groups have two receipt dates
(median one day apart, at most 824), 49 have the same receipt date. The values differ in 83 groups (66 with two
receipt dates, 17 with one: CCHI 12/2011, ORVR 12/2021, TEND 03/2023).

`core.valor.fundamentos_no_mes` picks one row by an unstable sort on the period end, `core.crescimento_lucro
.variacao_anual` and `core.classicas.balancos_em_tres_anos` by the file's row order (the latest receipt in 81 of the
180 two-receipt groups). The pick is arbitrary and can change with the lake's row order.

## Rule

Among the rows received before the decision month, one row per (issuer, account, period end):

1. the row of the latest receipt: a refiling replaces the statement from the day it arrives, not before;
2. two rows of the same receipt day with the same value count once;
3. two rows of the same receipt day with different values are unknown and left out (the lake's own rule for a code
   a document repeats with different values): the account falls back to the previous period end, as when a
   statement is missing. The export has no version number to order them; carrying it from the lake is the follow-up.

One function, `core.valor.documentos_conhecidos(fundamentos, mes)`, used by the three readers above (and through
`fundamentos_no_mes` by `core.indicadores`). No other change to any score.

## Measurement (fixed now, read once)

1. The Composite's replay of attempt 51 in the three phases (`scripts/composto_variantes_avaliar.py`'s anchor)
   against the published curves: the largest relative gap of the curve, the number of decisions whose 15 names
   change, and CAGR before and after. If the gap is below 1e-9 in every phase the published baseline stands.
2. The Composite's portfolio on the record in force (27/10/2026) from the lake's current exports, before and after.

If a phase's curve moves, the corrected curves become the baseline of later readings and the published ones are
kept beside them; DECISOES says which numbers moved. The trial count does not change: this is a data-handling fix,
not an arm.

## Live consequence

`core/valor.py` and `core/crescimento_lucro.py` are in paper-v3's guarded path: from this commit on its reading
prints that they changed since the `paper-v3` tag. That note is correct and stays: the live Composite is paper-v3's
rule with this fix. The record already written (27/10) is not rewritten; the fix acts from the next decision.

## Expectation (registered)

The twelve issuers are small or were briefly listed; none is expected among the 15 names on a decision date. The
published curves stand (gap below 1e-9) and the 27/10 portfolio is unchanged.
