# Fermionic selection, second reading: code review fixes, lower theta, von Neumann entropy

Date: 2026-11-11. Branch `feat/fermionic-v2`, worktree `../po-fermion`. The user, after DECISOES 343: "Revise todo o
código da Fermiônica, linha a linha ... se encontrar inconsistências corrija e rode novamente ... prossiga com o
rebaixamento do theta, se quiser implementar a entropia de von Neumann vai ser ótimo (recomendo)".
**Pre-registration: committed before any number of the new arms exists.** This is a second look at the idea of
DECISOES 343 (which tied the Composite): every arm below is a new row and the family is read as such.

## Review findings fixed here (first reading: `core/fermionica.py` at `e1ce6ee`)

Code, none reached by the first reading's data (candidates per decision 55-171, caps always admitted 15):

1. With no more than 15 candidates every candidate went in, skipping the 4-per-sector cap the Composite applies.
   Fix: that case, and caps that admit fewer than 15, delegate to `core.sinais.carteira_da_nota` (the Composite).
2. Caps admitting fewer than 15 of more than 15 candidates raised ValueError instead of holding fewer names.
3. The swap test compared F computed by two numerical routes (Schur update against a fresh determinant) with an
   absolute 1e-12 tolerance: at a large theta rounding could accept a zero-gain swap and cycle. Fix: every F of a
   sweep comes from one routine and a swap needs a gain above 1e-9 x max(1, |F|).

Method:

4. C was the correlation of the Ledoit-Wolf covariance, which shrinks towards a scaled identity: correlations of
   low-volatility names are deflated more (measured on phase 0's 66 decisions: a name's correlations keep 88-99%
   of their sample size, and that share correlates 0.81 with the name's volatility; shrinkage 1.6-4.0%). The
   diversity term carried a mild low-volatility tilt. Fix: C = Ledoit-Wolf on the standardized returns (unit
   variance per name), i.e. (1 - delta) R + delta I, the same shrinkage for every name.
5. The DSR of the first reading was computed with V from two near-identical curves (prints ~0.99, meaningless).
   Here the DSR table holds every arm, the Composite and the Conservadora; it is still reported with that caveat
   and the verdict rests on P[> Composite].

## Diversity functionals

For the picks' correlation C_S with eigenvalues l_1..l_k (sum = k):

- **log det** (fermionic, DECISOES 343): D(S) = sum log l_i. Unbounded below: a duplicate is forbidden (Pauli).
- **von Neumann** (new): the picks' density matrix is rho_S = C_S / k (PSD, trace 1) and its von Neumann entropy
  S(rho_S) = -tr(rho_S log rho_S), between 0 (one common factor) and log k (uncorrelated); e^S is the effective
  number of independent bets. D(S) = k (S(rho_S) - log k) = -sum l_i log l_i. Bounded below by -k log k: it
  penalizes the weight of the dominant (market) mode rather than near-duplicates.

Both are <= 0 with equality only at C_S = I (Hadamard; Jensen on the concave -l log l with mean eigenvalue 1), so
both share the scale of the score term. The rule is unchanged: S* = argmax 2 theta sum z_i + D(S), |S| = 15, at most
4 per sector, over every scored candidate; equal weights; house rules; deterministic swap ascent from two starts.
The entropy is checked in the tests against PennyLane 0.45.1 (`qml.math.vn_entropy`) on an 8-name density matrix.

## Arms (one row each, N 57 -> 63)

C corrected (finding 4) in all six: log det at theta 1, 0.5, 0.25; von Neumann at theta 1, 0.5, 0.25. The halvings
double the weight of diversity twice; no other theta is run.

## Anchors (before any arm is read)

1. The Composite's own strategy reproduces attempt 51's published curve (below 1e-9).
2. theta = 1e6 (log det, corrected C) picks the Composite's names or, on a score tie, names of equal total z-score.
3. Regression: the fixed code with the first reading's C (correlation of the covariance) at theta = 1 reproduces
   DECISOES 343's CAGR to three decimals (0.218, 0.191, 0.199). Not a new row.

## Gate (fixed now)

Per arm, the house rule: candidate only if P[arm > Composite] >= 0.95 and DSR >= 0.95 in all three phases. With six
correlated arms the chance that one passes by luck is above 5%: a pass is at most a candidate for a forward
fictitious account, never an adoption, and would be reported with the family size.

## Reported

Per phase and arm: CAGR, volatility, Sharpe and Sortino over the CDI, MDD, Calmar, alpha and Newey-West t and beta
against the Composite and BOVA11, P[> Composite], DSR, annual turnover, names shared with the Composite, mean
pairwise correlation and effective number of bets e^S of the picks; calendar years 2010-2026.

## Expectation (registered)

Lower theta moves the picks away from the ranking: about 13, 12 and 10 names shared at theta 1, 0.5, 0.25 for log
det, similar for von Neumann. CAGR falls with theta (0 to -2.5 points against the Composite at 0.25), volatility
and drawdown fall a little. No arm passes (probability that any does: about 5%). The corrected C at theta = 1
changes at most one name in a few decisions against DECISOES 343.

## Data and engine

As DECISOES 343: frozen exports `docs/dados/2026-10-22`, phases 0, 21, 42, `CFG_BT` (1260-session window, 63
sessions, 10 bps), `walk_forward_dinamico` with `Congelamento()`, score `_value_score(mensal, com_crescimento=True)`.
