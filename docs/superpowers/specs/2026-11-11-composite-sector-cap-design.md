# Composite without the sector cap

Date: 2026-11-11. Branch `feat/fermionic-v2`. The user: "Esse teto fixo que colocamos é a forma mais inteligente? Se
deixarmos ou afrouxarmos ... para testar?". **Pre-registration: committed before any number of the arm exists.**

## What was measured before (inputs only, no return)

On phase 0's 66 decisions: the 40% ex-ante volatility limit never binds (the Composite's is 16-28%; 0 refusals in
every reading), so loosening it changes nothing and is not tested. The 4-per-sector cap binds in 51 of 66 decisions;
without it the largest sector holds a median of 6 and up to 9 of the 15 names (Utilities in 26 of the 51, then
Consumer Cyclical, Basic Materials, Financials) and 1.9 names change on average. A cap of 8 (half the portfolio)
changes 1.88 names against 1.92 with none: one arm answers both.

## Arm (one row, N 67 -> 68)

**Composite-sem-teto**: attempt 51 with `por_setor = 15` (no cap; one class per issuer kept) and the house check
run with the 30% sector rule lifted (`teto_setor = 1.0`); every other limit unchanged. Same data, phases, engine and
costs as attempt 51 (`scripts/composto_variantes_avaliar.py setor`).

## Gate

Candidate only if P[arm > Composite] >= 0.95 and DSR >= 0.95 in all three phases. The sector cap is a fixed risk
rule of the real-money plan: whatever the result, it does not change without the advisor's review and the user's
decision; a pass is at most a forward fictitious account.

## Reported

As the variants' reading, plus the worst calendar year and the maximum drawdown next to the Composite's.

## Expectation (registered)

CAGR within +-1 point of the Composite, drawdown 0 to 4 points deeper, P[> Composite] 0.3 to 0.7. Fails the gate.
