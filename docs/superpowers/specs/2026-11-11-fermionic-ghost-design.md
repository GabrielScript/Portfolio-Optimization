# A fictitious account for the determinantal selection (log det, theta 0.5)

Date: 2026-11-11. Branch `feat/fermionic-ghost`. The user, after DECISOES 344: "Abra a conta fictícia do determinante
theta = 0,5". **Rule fixed here, before any live number.**

## What it is and what it is not

DECISOES 344 read six determinantal arms; log det at theta 0.5 was the best of them (CAGR 22.7 / 20.4 / 20.8% against
the Composite's 21.4 / 19.9 / 19.1%, P[> Composite] 0.87 / 0.65 / 0.91) and failed the gate. It is the best of six
correlated arms, so its backtest flatters it. This account follows it forward next to the Composite, like the Top-5
(DECISOES 290) and the 12/3 (DECISOES 299): not a decision, no trial row, no change to the real buy of 19/12.

## Rule (frozen)

At each Composite record (the Composite's own decision dates), on the lake's exports with the Composite's own score
and training window (`scripts.paper_v3`): `core.fermionica.estrategia_fermionica` with theta = 0.5, D = log det,
C = Ledoit-Wolf on standardized returns, 15 names, at most 4 per sector, one class per issuer, 1/15 each, house
rules. A refused portfolio keeps the previous one, as in every account.

## Live

Fictitious R$ 10,000 account "Fermiônica" in the simulator, same costs and liquidity rule as the others. Record
`simulador/fermionica.csv` (the Top-5's columns and dating), written and committed by the upkeep before the
simulator step. Registered on Wednesday 11/11/2026 after the close: the record in force (27/10) trades at the close
of Thursday 12/11/2026 (`INICIO`).

## Reading (fixed now)

Compared with the "Composto" account on the same days. No claim before 252 sessions; after that, P[> Composto] on
the live daily returns with the same `psr_da_diferenca`. A lead before that is not evidence.

## Engineering constraint

The Streamlit app's requirements do not include PySCIPOpt. `core/fermionica.py` imported `core.qubo_universo` (which
imports it) for two names; the module now defines them itself, and a test imports it with PySCIPOpt blocked.
