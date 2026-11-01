# The book's second batch, for very aggressive investors

Pre-registered on 01/11/2026, before any arm below was computed on real data. Only synthetic timing spikes ran before this commit.
The user's request: steps 2 to 6 of the order in DECISOES 227. It closes part of the gap found in DECISOES 226, where
`2026-11-01-todas-as-estrategias-design.md` left Palomar's formulations out without saying so. Steps 7 and 8 of that order wait for the
user and are not part of this test.

## 1. Arms (N 75 -> 89)

**Cost, stated before the result:** 14 more trials raise the DSR bar for every strategy, the Composite included. The count starts at
75 because `docs/tentativas.csv` here also holds the seven HRP rows of `feat/us-hrp`, already read on the US and shown on the screen.

Fixed for all unless the row says otherwise: 20% per name, 30% per sector, 40% volatility, only names with a sector. "At most 10" is the
app's two-stage rule: solve on all, keep the 10 largest, solve again. mu is `media_por_periodo`: Jorion's on the B3, raw on the US, where
it turns itself off. Sigma is Ledoit-Wolf in low-rank form. Parameters are the book's or fixed here, and none is tuned afterwards.

| # | Step | Arm | Palomar | Definition |
|---|---|---|---|---|
| 76 | 2 | Kelly-exato | 7.3.1 | max (1/T) sum log(1 + w'r_t) on the training rows, at most 10 |
| 77 | 2 | Kelly-exato-sem-teto | 7.3.1 | The same without the 40% volatility cap. 20% per name and 30% per sector stay |
| 78 | 3 | Quintil-mu | 6.4.4 | 1/k on the top k = ceil(N/5) by mu, cut to the caps |
| 79 | 3 | Quintil-mu-sigma | 6.4.4 | The same by mu/sigma |
| 80 | 3 | Quintil-mu-sigma2 | 6.4.4 | The same by mu/sigma^2 |
| 81 | 3 | Box-robusto | 14.2.2 | max w'(mu - delta) - (1/2) w'Sigma w, delta_i = sigma_i / sqrt(T), T in years, at most 10 |
| 82 | 4 | Subset-resampling | 14.3.2 | Kelly lambda 2 on ceil(N^0.7) random names, B = 100, seed 0, no cardinality inside, zeros filled, averaged, projected on at most 10 and the caps |
| 83 | 4 | Subset-bagging | 14.3.2 | The same, each draw also on a circular block bootstrap of the rows: blocks of 21 sessions on the B3, 1 month on the US |
| 84 | 5 | Downside-1 | 10.3.1 | max w'mu - (1/T) sum (0 - w'r_t)+, both per period, at most 10 |
| 85 | 5 | Semi-variancia | 10.3.1 | The same with the square, exact and not Estrada's approximation |
| 86 | 5 | Downside-3 | 10.3.1 | The same with the cube |
| 87 | 6 | MVSK-tilting | 9.3.3 | From w0 = the Composite's portfolio: max delta with phi1 >= phi1(w0) + delta \|phi1(w0)\|, phi2 <= phi2(w0)(1 - delta), phi3 >= phi3(w0) + delta \|phi3(w0)\|, phi4 <= phi4(w0)(1 - delta), Herfindahl <= 1.01 Herfindahl(w0), caps |
| 88 | 6 | Utilidade-raiz | 7.3.2 | max (1/T) sum sqrt(1 + w'r_t), at most 10 |
| 89 | 6 | Retorno-minimo | 7.1 | min w'Sigma w with w'mu >= the mean of mu over the eligible names, at most 10; if no portfolio meets it, the minimum variance |

**Choices, and why:**

- **L1-robust mu is not a separate arm.** The book's Example 14.3 and (6.13) show its solution is the quintile portfolio for the right
  uncertainty size, and 1/N for a large one. Rows 78 and 12 are those two points. A test holds the identity on a small case.
- **Three quintiles** because the book names three rankings and gives no reason to prefer one.
- **Row 77 relaxes a mandatory rule on purpose.** `Limites.vol_maxima` is a fixed rule of the app. This arm asks whether the cap is what
  holds the growth objective back. At the one US decision checked, lambda 0.25 to 4 gave the same portfolio with the cap binding. A pass
  would not change the rule by itself: that would be the user's decision.
- **Exact Kelly and the root utility use a sequence of QPs,** Newton steps on the sample objective with the caps as constraints, at most
  25 steps, each halved up to 3 times if it does not improve. CLARABEL's exponential cone failed on the synthetic B3 size. Tests hold the
  result to SciPy's optimizer on small problems.
- **No Jorion in rows 76, 77, 84-86 and 88:** they read the training rows directly, as the book writes them. Their means are raw on both
  markets.
- **tau = 0 in rows 84-86:** a loss, not a shortfall from the mean. (10.1) defines the semi-variance around the mean, and (10.7) leaves
  tau free. lambda = 1, per period, as Mean-CVaR's. So the balance between mean and risk differs between the B3's days and the US's months;
  it is not tuned to match.
- **Tilting** takes the book's defaults: kappa = 0.01 xi(w0) with xi the Herfindahl, and g_i = delta x phi_i(w0), with absolute values
  so that a negative reference moment still moves the right way. Moments come from the scenarios per period, phi2 from Sigma. phi3 and phi4
  are linearized at each step, at most 20. All eligible names may enter. delta = 0 at w0 is always feasible, so a failed step keeps w0.
- **The root utility** is the least risk-averse concave one on the book's list, and the book says the others differ negligibly once
  estimation error is counted.
- **Row 89's beta** is 1/N's, the book's rule of anchoring on a benchmark.

## 2. Engines, gates and order

Same as `2026-11-01-todas-as-estrategias-design.md` section 5:

- **B3:** `walk_forward_dinamico`, phases 0/21/42, the frozen folders. The Conservadora must reproduce 17.41/16.41/16.90% and the Composite
  21.4/19.9/19.1%.
- **US:** `core.us_teste`, monthly grid, every 3 months, 36 months, 10 bps per side, phases 0/1/2, the frozen exports. The Composite, the
  Conservadora and 1/N must reproduce the report made from those exports.

**Gate, each arm, every phase, all >= 0.95:** P[> Composite], P[> Conservadora], P[> reference], DSR. A pass makes the arm a candidate for a
forward paper log, nothing more. Nothing is tuned after a reading. CAGR, volatility, drawdown, turnover and the year tables are reported, never
gates. Each configuration is read once per market.

**Order:** the first batch's US reading goes first. Then this batch on the B3, then on the US, in the row order above, one heavy job at a time,
outside 21:30-00:30.

**Changed on 01/11/2026 at 18:20, before any reading:** this batch's B3 reading goes first, because the first batch's US reading waits for the Tiingo
batch and the splits fix. Nothing is tunable between them and the data is frozen (DECISOES 230).

## 3. Prior, written before any reading

No arm passes against the Composite on either market, nor on the US against SPY. Rows 76 and 88 will sit close to Kelly at lambda 1 on the B3,
as the book expects from Pulley (1983). Row 77 will show more volatility and a deeper drawdown than row 76 without enough extra return to pass.
Quintile by mu will look most like momentum and be the most volatile of the three. Row 87 stays close to the Composite, so its P[> Composite]
stays near 0.5 and it cannot pass. On the US the raw 36-month means make rows 76, 77, 84-86 and 88 chase past winners.

## 4. Output

`docs/lote2-livro-2026-11-01.txt`, derived files in `docs/dados/`, a row per arm in the screen's table, a DECISOES entry. No arm becomes an option
in the app's selector without passing the gate. No real money moves from this table: the user decides.
