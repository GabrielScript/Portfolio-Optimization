# The number of names, the theta ladder and two new constructions

Date: 2026-11-11. Branch `feat/fermionic-grid`. The user: "testa a decomposição fermiônica e a entropia de von
Neumann, se mudarmos para 5/10/+20 ações muda algo ... variações da composite, variações de theta (adicione o teta
de 0,75 também) ... Esse número e ações pré-fixadas dos modelos quânticos são imutáveis? ... se quiser deixar outra
estratégia quântica, fique à vontade". **Pre-registration: committed before any number of these arms exists.**

Same data (frozen exports to 16/10/2026), phases 0 / 21 / 42, engine, costs and Composite score as attempt 51 and
DECISOES 344. The refiled-statements fix (`2026-11-11-duplicate-filings-design.md`) is in the code that runs this.

## Family A: size x theta x functional (38 rows)

Sizes k = 5, 10, 20, 30 (15 is read: DECISOES 344). Names per sector `3k // 10` (1, 3, 6, 9): the house's 30%
sector rule in names; at k = 15 it is the Composite's 4, at k = 5 the Top-5 ghost's 1. Equal weights 1/k, one class
per issuer, house rules unchanged (20% a name allows k >= 5).

- **Composite-k** (4 rows): `estrategia_por_nota(nota, setores, nomes=k, por_setor=3k//10)`.
- **LogDet-k t** and **vonNeumann-k t** (32 rows): `estrategia_fermionica` at that k and cap, theta = 0.25, 0.5,
  0.75, 1, C = Ledoit-Wolf on standardized returns.
- **LogDet-15 t0.75** and **vonNeumann-15 t0.75** (2 rows): the missing rung at the read size.

Anchors per phase, before any arm is read: the Composite's replay is its published curve (1e-9); the six read arms
of DECISOES 344 are not rerun (their curves are in `docs/dados/fermionica-v2-curvas-2026-11-11.parquet`) except
LogDet-15 t0.5, rerun as the code anchor (1e-9 against its saved curve).

## Family B: two constructions of ours (3 rows)

### B1. Grand-canonical determinantal selection: the size is not fixed

The fixed-size rule is the canonical ensemble (the k-DPP). Dropping the size constraint gives the L-ensemble
P(S) proportional to det L_S, L = diag(e^{theta z}) C diag(e^{theta z}), whose log is the same F(S) = 2 theta sum z
+ log det C_S for a set of any size (chemical potential 0: no new parameter).

Marginal value (Schur complement; `tests/test_diversidade.py` checks it numerically):

    F(S + i) - F(S) = 2 theta z_i + log(1 - R2_{i|S})

R2_{i|S} the share of name i's standardized variance explained by the names already held. A name enters only while
its score pays for what the portfolio already knows about it; the size is where that stops.

**GrandCanonical t0.5** (1 row): at each decision, the fixed-size picks S_k (same optimizer, cap `3k // 10`) for k in
5, 8, 10, 12, 15, 20, 25, 30, 40, 50; the portfolio is the S_k of largest F (ties: the smaller k), 1/k each, theta =
0.5 (the rung already followed forward), house rules. Reported: the distribution of the chosen k.

### B2. Equal information contribution: weights from the von Neumann entropy

For held names with correlation C, take unit vectors |c_i> with <c_i|c_j> = C_ij and the state
rho(w) = sum_i w_i |c_i><c_i| (trace 1 on the simplex). Its spectrum is that of diag(sqrt w) C diag(sqrt w).

- S(rho(w)) <= H(w), the Shannon entropy of the weights, with equality iff the names are uncorrelated: the von
  Neumann entropy is the weights' entropy with the shared part discounted, and exp S is the effective number of
  independent bets.
- rho is linear in w and S is concave, so maximizing S over the weights is a concave program with one optimum.
- At that optimum without binding caps every held name has -<c_i| log rho |c_i> = S(rho): each name is exactly as
  surprising to the portfolio as the portfolio's own entropy (the capacity condition of a pure-state ensemble). The
  classical cousin is equal risk contribution; this is equal information contribution (EIC).

**Composite-EIC** (1 row): the Composite's 15 names, weights argmax S(rho(w)) subject to sum w = 1, 0 <= w_i <= 20%,
each sector <= 30%, each issuer <= 20% (`core/diversidade.eic_weights`, SLSQP from equal weights), house rules.
**LogDet-15 t0.5-EIC** (1 row): the same weights on the followed arm's 15 names.

`tests/test_diversidade.py` checks the three statements numerically on random correlation matrices before the run.

## Gate (unchanged) and what is read

Candidate only if P[arm > Composite] >= 0.95 and DSR >= 0.95 in all three phases, against the 15-name Composite. A
pass is at most a forward fictitious account; nothing here touches the real buy of 19/12. N 68 -> 109.

Read besides the gate, fixed now:

1. Size: Composite-k against the Composite (CAGR, volatility, drawdown, turnover).
2. Does the determinant add at each size: alpha and its t of each arm against the Composite-k of its own size.
3. Theta: for each functional and size, whether CAGR is single-peaked in theta and where.
4. The table is 41 correlated rows: the best cell is reported with the count of rows beside it, never alone.

## Expectation (registered)

Composite-5 has the highest CAGR and the deepest drawdown; 20 and 30 names give up 1 to 3 points a year for 2 to 4
points less volatility. The determinant adds 0 to 1.5 points against the same-size Composite, largest at 20-30
names where there is more to choose from, never with t above 2. theta 0.5-0.75 is the peak. GrandCanonical picks
25 or more names in most decisions and lands near Composite-30. EIC weights move CAGR by less than 0.5 point. No
arm passes the gate.
