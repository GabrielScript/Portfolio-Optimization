# The four signals that passed their IC gate, combined into one small fixed tilt on the Composite (pre-registered 2026-11-13)

Written and committed before any number of this arm. The user, 2026-11-13: "Pode pré-registrar e rodar o teste de
sinais combinados", after the answer that the Fermiônica family is exhausted (DECISOES 342-350) and that the honest
remaining bet is signal, not solver.

## What is already known (not a free choice)

Each of these four scores passed its stage 1 IC gate and then lost, alone, as an earned-weight tilt on the Composite:
P patents and X convexity (DECISOES 352), INS insider net purchases (354), RET-3y the 36-month total return (358).
Their stage 2 curves were seen; their combination was never computed.

## Rule (one arm, read once)

`Comp+Extras-0.25`: z(Composite) + 0.25 x z(EXTRAS), where EXTRAS is the house mean of normal scores
(`core.views.nota_media_dos_escores`) of P, X, INS and RET-3y among the candidates, each name with the signals it
has; z the house normal score over the names with a Composite score; a name with none of the four adds nothing
(`core.informados.weighted_score` with a constant weight). Then the Composite's rule: 15 names, at most 4 per sector,
house rules, 10 bps, walk-forward on the dynamic universe.

Why 0.25 and fixed: chosen before reading, with no fitting. Averaging four weak, partly independent signals should
lower the noise of each; the weight is kept small so the tilt changes a few names, not the book. No earned weight,
no grid, no second weight.

## Data and anchor

The Composite with the growth age rule (DECISOES 360, `main` after `e201814`). Before the arm is read, in each phase
the Composite's replay must reproduce its saved curve `docs/dados/pos-idade-curvas-fase{0,21,42}-2026-11-13.parquet`
(column `Composite`) within 1e-9, or the run stops. The four signals are built exactly as in their own readings
(`scripts/inovemetria_avaliar.py`, `scripts/informados_avaliar.py`, `scripts/indicadores_risco_avaliar.py`).

## Gate

P[> Composite] >= 0.95 and DSR >= 0.95 in each of phases 0, 21 and 42. N 132 -> 133 in `docs/tentativas.csv`.
A pass means a fictitious account followed forward, never money.

## Prior

CAGR within +-1 point of the Composite, P[> Composite] 0.55 to 0.80, about 12 of 15 names shared. Chance of a pass
15-20%. Most likely reading: no pass.
