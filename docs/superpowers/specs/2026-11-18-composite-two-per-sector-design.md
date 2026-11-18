# Composite with at most 2 names per sector, and the uncapped arm re-read on today's code

Date: 2026-11-18. Branch `feat/composto-2-por-setor`. The user: "Temos um teto de nas nossas 15 ações de no máx. 4
setores. Se removermos esse teto e deixar livre ou reduzir para o máximo de setores distintos como maximo 2 ações por
setor muda algo?". **Pre-registration: committed before any number of either arm exists.**

## What is already known

DECISOES 347 (11/11): the Composite without the cap, `por_setor = 15`, fails: 20.3 / 18.8 / 17.6% against
21.4 / 19.9 / 19.1%, P[> Composite] 0.24 / 0.25 / 0.16 (N 68). The 4-per-sector cap binds in 51 of 66 decisions.
Since then the Composite changed twice (refiled statements, DECISOES 349; growth age, 360): the baseline is now
21.8 / 20.3 / 19.2% and attempt 51's saved curve no longer anchors a replay.

## Arms (one new row, N 201 -> 202)

N is DECISOES' running count (201 after 384); main's `docs/tentativas.csv` holds 109 rows because several branches'
ledgers were never merged. The DSR uses the larger.

1. **Composite-2-por-setor (new):** the Composite with `por_setor = 2` (at most 2 of the 15 names a sector, so at
   least 8 sectors), the house limits unchanged (30% a sector is not binding at 2/15 = 13.3%). If a decision has
   fewer than 8 sectors with eligible names, the strategy holds what the rules allow and the rest is cash; the names
   held per decision are reported.
2. **Composite-sem-teto (re-read, not a new row):** DECISOES 347's arm, unchanged, on today's code and data, so the
   user's two questions sit on one table. It adds nothing to N.

Reference: the Composite (`por_setor = 4`) replayed in the same run on the same data. The anchor is that replay's
CAGR against the current baseline (21.8 / 20.3 / 19.2%, within 0.1 point), not attempt 51's saved curve.
Same frozen export, phases 0 / 21 / 42, engine, 63-session rebalance and 10 bps costs as attempt 51
(`scripts/composto_variantes_avaliar.py setor2`).

## Gate

Candidate only if P[arm > Composite] >= 0.95 and DSR >= 0.95 in all three phases. The sector cap is a fixed risk rule
of the real-money plan: a pass is at most a forward fictitious account, never a change without the user's decision.

## Reported

CAGR, Sharpe, max drawdown, worst calendar year, alpha against the Composite, P[> Composite], DSR, turnover, names
shared with the Composite, eligible names, and how many decisions the 2-per-sector rule changes.

## Expectation (registered)

2 per sector: CAGR 0.5 to 2 points a year below the Composite in every phase (it forces lower-score names in to fill
new sectors), volatility and drawdown within 2 points of the Composite's, P[> Composite] 0.05 to 0.40. Fails.
Without the cap (re-read): still below the Composite, as in 347. Fails.
