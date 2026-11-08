# One ranking for B3 stocks and US BDRs (the rule picks how many BDRs), B3 — design and pre-registration

_2026-11-08. Written and committed before any number of this arm exists. The user (08/11): "Pode começar pela 1 mesmo
das BDR's" — idea 1 of the 08/11 list: rank B3 stocks and US BDRs in one queue by one score and let the rule choose
how many BDRs enter, instead of DECISOES 298's fixed counts and the live 12/3 ghost (DECISOES 299). Branch
`feat/bdr-conjunto` from `feat/pead-eventos` (`bd38570`), with `core/mix_bdr.py` taken unchanged from `feat/mix-bdr`.
Sections 1 to 6 are fixed from the commit on; anything left open is fixed in an addendum before the number it
touches._

## 1. What the data allow, and the prior

The same window as DECISOES 298: US BDRs opened to retail in 10/2020, so the reading runs from the first rebalance on
or after **2021-01-04 to 16/10/2026**, about 5.8 years and one US tech run. The engine needs 1,260 NaN-free sessions
per name, so no BDR can sit inside it; the joint portfolio is simulated outside the engine on the engine's own
rebalance dates and candidates, and that simulator must first reproduce the engine's Composite (section 4).

**Prior, written now:** about 15% that the arm passes. Likelier: a CAGR above the Composite's in 2021-2026, as the
fixed mixes had, with P[> Composite] or the DSR below 0.95 in some phase because of the short window.

## 2. The pool at each rebalance d

- **B3:** the engine's eligible names at d (the columns of its training window) that have a Composite score
  (attempt 51's, `_value_score(com_crescimento=True)`), each with its sector.
- **US:** DECISOES 141's 32 names (30 US-company BDRs, IVVB11, NASD11; the list of `scripts/mix_bdr_avaliar.py`),
  eligible at d by `core.mix_bdr.elegiveis_no_mes` (traded in >= 90% of the previous 12 months' sessions, median daily
  volume >= R$ 1 mi) and with the 252 sessions of prices before d. Prices: the lake's gold `tr_close` read live, its
  slice hashed, AVGO34 2021-09-17 dropped (as DECISOES 298). All US names form one sector, "EUA".

## 3. The score and the portfolio (`Conjunto-15`)

- **Price components, over the whole pool (B3 + US):** momentum 12-1 (`core.mix_bdr.momentum_12_1`: total return
  from 252 to 21 sessions before d) and low volatility (minus the standard deviation of the last 252 daily returns up
  to d), each turned into a normal score (`core.views.escore_normal`) among the pool's names that have it.
- **Fundamental component, B3 only:** the Composite score's normal score among the B3 pool.
- **Score = sum of a name's available normal scores / sqrt(their count)** (Stouffer): a US name has 2 components, a
  B3 name 3, and each total has unit variance when its parts are independent, so the US names are not pushed to the
  extremes by having fewer parts.
- **Portfolio:** the 15 highest scores, at most 4 per sector ("EUA" counts as one sector, so 0 to 4 BDRs; at most
  4 x 1/15 = 26.7% in the US, under the 30% sector cap), one class per issuer (the Composite's `carteira_da_nota`
  rule), equal weights 1/15, held to the next rebalance. The name and sector caps hold by construction; the
  volatility cap is not checked (it never refused a Composite portfolio in any reading); the ex-post volatility is
  reported.
- One configuration; no search over the components, the Stouffer weights, the 4-name US cap or the windows.

## 4. Simulation and plumbing

- Rebalance dates and B3 candidates: recorded from the engine's Composite run of each phase (`walk_forward_dinamico`,
  `CFG_BT`: 1,260 sessions, every 63, frozen `docs/dados/2026-10-22/`), phases 0/21/42; the Conservadora must
  reproduce 17.41/16.41/16.90% and the Composite 21.4/19.9/19.1% first.
- Simulator: from the first rebalance on or after 2021-01-04; at each rebalance the target weights, 10 bps on each
  side of the traded weight; between rebalances each holding drifts with its price (B3: the engine's price file; US:
  gold `tr_close`), a price that stops carried forward.
- **Plumbing check, before the arm:** the simulator fed the Composite's own picks must give a CAGR within 0.3 p.p. of
  the engine's Composite over the same window in each phase, or the script stops.
- Pre-tax, as every engine reading; a BDR gain pays 15% with no exemption, so tax can only lower the arm.

## 5. Gate

Against the Composite (engine curve) and the Conservadora on the same window. **Gate, every phase, all >= 0.95:**
P[> Composite], P[> Conservadora], P[> BOVA11], DSR. Reported, not gates: CAGR, MDD, volatility, turnover, the year
table, IVVB11 and the fixed BDR-3 rule over the same window (from the same simulator), the number of BDRs held at
each rebalance (mean, min, max), the latest picks.

**N.** One configuration, one row in `docs/tentativas.csv` in this commit. The DSR uses the union of every branch's
trials, **N = 109** (108 + 1).

## 6. What follows and output

A pass makes it a candidate for a forward ghost account net of the BDR tax, beside the 12/3; nothing goes to real
money from a backtest. A fail closes the joint-ranking line; the 12/3 ghost keeps running as pre-registered. Output
`docs/bdr-conjunto-2026-11-08.txt` and a DECISOES entry. Nothing goes to the app.
