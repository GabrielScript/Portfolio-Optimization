# H2: BR/US/gold sleeve rotation with a CDI filter — design and pre-registration (attempt 53)

_2026-10-29, session 59. Written and committed before any code or curve of this rule exists. User's /goal step 3;
DECISOES 171 (FinRL-X's Adaptive Rotation idea, H2), 173 (the winning funds were US tech exposure)._

## 0. What was already seen (data snooping, disclosed)

Before this design: IVVB11 beat the Conservadora since 2014 (DECISOES 149, 153, 171) and a descriptive BR + US BDR
optimizer mix (`feat/mix-br-eua`, 27/10, never merged) showed IVVB11 CAGR 20.8% vs BOVA11 12.0% (2014-04 to 2026-10).
So the US leg's good decade is known. The rule below takes every parameter from the literature (Antonacci's dual
momentum, 12 months, monthly; FinRL-X's rotation between equity, gold and cash), none from our data.

## 1. Sleeves (BRL, daily values on the B3 calendar)

- **BR:** the Conservadora's daily curve of each phase, `docs/dados/composto-curvas-2026-10-28.parquet` (sha256 in
  `docs/dados/SHA256SUMS`; net of its own trading costs; 15 names, rules).
- **US:** SPY total return (Yahoo adjusted close) x PTAX selling (BCB SGS 1); on B3 session d the SPY close of the
  latest US session strictly before d, times the PTAX of d (published ~13h). Fee drag 0.25%/yr (IVVB11 0.23%).
- **Gold:** GLD x PTAX, same timing, fee drag 0.40%/yr (GLD 0.40%).
- **Cash:** the frozen daily CDI (`docs/dados/2026-10-21-macro/cdi_daily.parquet`).
- Frozen copy of SPY, GLD and PTAX through 16/10/2026: `docs/dados/2026-10-29-us/` with SHA256SUMS.

## 2. The rule (fixed now)

On the last B3 session of each month: the 12-month return (252 sessions) of BR, US and Gold; the highest one is held
from the next session **if** it beats the CDI's 12-month return, else the CDI. 100% in one sleeve, no leverage.
Switching costs 10 bps per side (`CFG_BT`). First decision: the first month-end with 252 sessions of every sleeve.

## 3. Gate (read once)

One trial row in this commit (N 52 -> 53; attempt 52 is the CDA best ideas). In each phase (0/21/42, the BR curve's), on the days both exist: P[> Conservadora]
(`psr_da_diferenca`) >= 0.95, P[> BOVA11] >= 0.95 and DSR >= 0.95 (`tabela_do_periodo`, N from `docs/tentativas.csv`).
Reported: CAGR, MDD, sleeve shares by year, switches per year, year table with 2025-2026.

- **Candidate:** pre-register a forward paper log against the same-day Conservadora.
- **Not a candidate:** H2 is closed; the allocation layer stays fixed weights (no timing).

Limits stated: gross of income tax (a switch out of a B3 ETF pays 15% on its gain, the stock sleeve has the R$ 20,000
exemption); before IVVB11 (2014) and GOLD11 (2020) the US and gold sleeves are proxies a B3 investor could not buy.

## 4. Expectation, written before

Rotation rules catch long regimes and lose at turns. The US leg's decade is known, so P[> Conservadora] may be high in
the full window; expected not a candidate, failing the 0.95 bar in at least one phase (the BR sleeve won 2016-2019 and
2022, with whipsaws at each turn).
