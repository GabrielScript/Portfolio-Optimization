# A broad characteristic panel with Ridge and PLS on the B3 — design and pre-registration

_2026-11-16. Written and committed before any feature, forecast, IC or curve of it exists. The user approved the broad
panel ("Pode rodar o setorial e o painel amplo") after DECISOES 374 (trees: Ridge stays the best model on the six
angles) and 375. Branch `feat/painel-amplo` from `feat/arvores`: the 213 harness (rows, target, purge, training rows,
IC reading, portfolio reading with the Composite at 21.4/19.9/19.1%) is reused; only the inputs change. Sections 1-6
are fixed from the commit on._

## 1. Why

Gu, Kelly and Xiu (2020) and Kelly, Pruitt and Su (2019): out-of-sample gains in the cross-section come from many
characteristics, shrunk, not from model complexity on a few. Every model on our six angles tied or lost to Ridge
(213, 296, 374), so the binding limit is the inputs. This test widens them using only frozen data already in the repo
(`docs/dados/2026-10-22/`, `docs/dados/2026-10-24-cvm/`): no paid source.

**Prior, written now:** Ridge on the broad panel reaches IC 0.12-0.15; about 25% that its difference with Ridge-6
has t >= 2; about 10% that a portfolio arm passes the gate.

## 2. Rows, target, training rows: the 213 package's, unchanged

One row per (month, name of that month's universe with a returns column), the 3-month normal-scored target, at most
20,000 training rows from the last 120 months of closed labels, seeded by the month (`core.qml_painel`). The new matrix
is aligned to the package's `chaves.parquet` (month, security) and the build fails if any row differs; `y`, `mes`,
`previstos` and `treinos` are copied from `pacote.npz` and checked equal.

## 3. Features (29), each known before the decision month

Each is turned into the 213 angle, `pi (u - 1/2)` with `u` the month's percentile among the names that have it,
missing = 0. Fundamentals are the issuer's latest document received before the month and at most 18 months old
(`core.valor.fundamentos_no_mes`); "a year ago" is the same function at month - 12; market value is the lake's
`market_cap` of the decision month (close of m-1). Ratios with a non-positive denominator are missing.

| # | Feature | Definition |
|---|---|---|
| 1-6 | the 213 six | momentum 12-1, low volatility, dividend yield 12m, Magic Formula, Trending Value, profit growth (as built for 213) |
| 7 | momentum 6-1 | total return m-6 .. m-2 |
| 8 | short-term reversal | minus the return of m-1 |
| 9 | long-term reversal | minus the total return m-36 .. m-13 (all 24 months present) |
| 10 | beta | OLS slope on the equal-weighted mean return of the month's candidates, m-36 .. m-1, at least 24 months |
| 11 | idiosyncratic volatility | standard deviation of that fit's residuals (minus, low is high) |
| 12 | max daily return | minus the largest daily close-to-close return in the last 21 sessions before the month (Bali-Cakici-Whitelaw) |
| 13 | daily volatility | minus the deviation of daily returns over the last 63 sessions (at least 40) |
| 14 | size | minus log market value |
| 15 | earnings yield | net income (parent, else total) / market value |
| 16 | book-to-market | (equity - minority interest) / market value |
| 17 | sales yield | revenue / market value |
| 18 | cash-flow yield | operating cash flow / market value |
| 19 | EBITDA / EV | `core.valor` definition |
| 20 | ROE | net income / equity |
| 21 | ROA | net income / total assets |
| 22 | gross profitability | gross profit / total assets (Novy-Marx) |
| 23 | operating margin | EBIT / revenue |
| 24 | accruals | minus (net income - operating cash flow) / total assets (Sloan) |
| 25 | leverage | minus (short + long-term debt) / total assets |
| 26 | cash / assets | (cash + short-term investments) / total assets |
| 27 | asset growth | minus (total assets / total assets a year ago - 1) (Cooper-Gulen-Schill) |
| 28 | sales growth | revenue / revenue a year ago - 1 |
| 29 | net issuance | minus (shares today / shares a year ago - 1), from `market_caps.shares_today_units` |

Signs are conventions only (the models learn them). Income-statement accounts are taken as the documents report
them, as every earlier reading does.

## 4. Models, one configuration each, in `.venv-qpo`

| Model | Definition |
|---|---|
| `Ridge-29` | `qml_sinal.ridge_gcv` on the 29 angles (the 213 Ridge: intercept, penalty by GCV on the month's training rows) |
| `PLS-29` | scikit-learn `PLSRegression(n_components=3, scale=True)` on the 29 angles, refit every month |

Each refit per forecast month on that month's training rows, deterministic (no seeds).

## 5. Gates

Signal, window 2012-01 to 2026-07: **IC >= 0.02 and Newey-West t (2 lags) >= 2**. Edge claimed for a model only if it
passes and its differences against `Ridge` (the 213 six) and against the Composite both have t >= 2. Descriptive: each
feature's own IC, and the last month's Ridge coefficients.

Portfolio arms `Ridge29-15` and `PLS29-15`: the forecast through `estrategia_por_nota` (15 names, 4 per sector, equal
weights), engine and frozen data as the tree reading. **Gate, every phase, all >= 0.95:** P[> Composite],
P[> Conservadora], P[> BOVA11], DSR. A candidate for a forward ghost account only if its model also passes the
signal gate. The official book stays the pure Composite.

**N.** Two configurations, two rows in `docs/tentativas.csv` in this commit: DSR N = 197 (195 after the sectoral
test, + 2).

## 6. Compute and output

Features built in the app's `.venv` (writes to `painel_run/` in the worktree; the venvs cannot write on D:), forecasts
in `.venv-qpo`; the merged `previsoes.npz` is copied to D: by the shell after a backup, earlier columns checked by hash.
Output: `docs/painel-amplo-ic-2026-11-16.txt`, `docs/painel-amplo-carteiras-2026-11-16.txt`, a DECISOES entry.
