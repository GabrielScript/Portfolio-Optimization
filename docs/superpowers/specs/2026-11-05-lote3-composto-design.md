# The Composite's signal inside three optimizers (pre-registration)

Pre-registered on 05/11/2026, before any arm below was computed on real data. The user's request (05/11): "vamos rodar essas
combinações do 1 ao 3, apenas", after DECISOES 234 and 265: the same optimizer differs by about 6 points a year on the forecast alone
(Markowitz fed the Composite's alpha, MV-continuo, 18.9-20.3%, ties the Composite; fed historical means, 13-14%), and the 25 book arms read
historical means, except MVSK-tilting. Here three optimizers get the Composite's forecast instead. Only the B3: the US is shelved (DECISOES
263).

## 1. Arms (N 89 -> 92)

**Cost, stated before the result:** three more trials raise the DSR bar for every strategy, the Composite included.

Fixed for all: the names are `core.qpo.candidates(score, sectors, FULL_UNIVERSE)` at each decision, so every name with a Composite score
and a sector, one class per issuer (as MV-continuo); Sigma is the training window's Ledoit-Wolf in low-rank form, annual
(`ledoit_wolf_baixo_posto`, as the book batches); 20% per name, 30% per sector, 40% volatility, at most 10 names by the app's two-stage
rule (as the book batches). The forecast is Grinold-Kahn's alpha, exactly MV-continuo's: alpha_i = IC0 x sigma_i x z_i, IC0 = 0.059 (the
Conservadora's 1998-2007 IC, before every out-of-sample date), sigma_i the annual volatility from Sigma, z the score's cross-sectional
z-score (`core.qpo.qubo_alphas`). Per session it is alpha / 252. Rows 90 and 91 read the training rows **recentred on the forecast**:
r~_t = r_t - mean(r) + alpha / 252, so their risk comes from the history and their mean from the signal. No parameter is tuned afterwards.

| # | Arm | Palomar | Definition |
|---|---|---|---|
| 90 | Meio-Kelly-alfa | 7.3 | max (1/T) sum U(1 + w'r~_t), U(y) = -1/y (CRRA gamma 2, the half-Kelly of MacLean, Thorp and Ziemba), the QP sequence of `pesos_utilidade`, at most 10 |
| 91 | Mean-CVaR-alfa | 10.4 | max w'alpha/252 - CVaR_0.90(w'r~), per session, Rockafellar-Uryasev's LP (`pesos_cvar`), at most 10 |
| 92 | Robust-MV-alfa | 14.2 | max w'alpha - kappa \|\|S^(1/2) w\|\| - (1/2) w'Sigma w, S = Sigma / T, T in years, kappa = 1 (`pesos_robusto`), at most 10 |

**Choices, and why:**

- **Half Kelly, not full:** full Kelly with this alpha is MV-continuo's objective (lambda 1, already read), and the second batch's exact
  Kelly on raw means fell to -64%. Gamma 2 is the standard fractional Kelly; it is fixed here, not searched.
- **CVaR at 0.90** is the book batches' `ALFA`, unchanged.
- **Row 92's ellipsoid** is the book's (uncertainty of a sample mean over T years) around the alpha; the alpha's own error is not modelled
  separately. kappa = 1 as the first batch's Robust-MV.
- **At most 10 names** keeps these rows comparable with the 25 book arms; MV-continuo had no cardinality.

## 2. Engine, gate and order

Same as the second batch (`2026-11-01-lote2-livro-design.md` section 2) on the B3: `walk_forward_dinamico`, phases 0/21/42, the frozen
folders and the same sha256 line; the Conservadora and the Composite must reproduce their numbers (the script's plumbing checks).

**Gate, each arm, every phase, all >= 0.95:** P[> Composite], P[> Conservadora], P[> BOVA11], DSR (N 92). A pass makes the arm a candidate
for a forward paper log beside the Composite, nothing more: real money stays with the Composite (19/12/2026) and the user decides. Read once.

## 3. Prior, written before any reading

No arm passes. All three stay close to the Composite and MV-continuo, P[> Composite] between 0.2 and 0.5. Row 92 gives up return for a lower
volatility (the penalty pulls toward minimum variance); row 91 has the smallest drawdown or close to it; row 90 sits between MV-continuo and
the Composite.

**Added 05/11, after the synthetic tests and before any real data:** per session the alpha is about 1e-4 against a daily CVaR near 2e-2,
so row 91 sits close to the minimum-CVaR portfolio and the signal moves it little (on the synthetic test, 32% against 22% of weight on the
favoured half). Kept as registered (the book batches' per-period convention); stated so a loss is not read as the signal failing.

## 4. Output

`docs/lote3-composto-b3-2026-11-05.txt`, `docs/dados/lote3-{anos,portao}-b3.parquet`, a DECISOES entry. Nothing goes to the app's selector
without passing the gate.
