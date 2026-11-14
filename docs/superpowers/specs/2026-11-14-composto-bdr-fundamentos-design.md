# The Composite's score on B3 BDRs: SEC fundamentals, B3 prices (pre-registered 2026-11-14)

The user, 14/11, after the sector tables: "Pode rodar os fundamentos por favor, se for para as BDR's da B3. As ações
americanas foram tiradas por web scraping e não são confiáveis". The sector picks of DECISOES 367/368 read their own
window; this rule chooses each month with what was known then, so it can face the gate.

Written and committed before any number of it is read. Read so far: names, sessions traded, volumes, which US filers
have fundamentals and market values, and the B3 jump table (no return of any arm).

## Data, and where each piece comes from

- Prices and returns: B3 only. The lake's gold `tr_close` of each BDR (B3 cash events from 11/2025 included; none
  before, so the book is price-only before that, about 0.5 to 1 point a year short for the dividend payers).
- B3 price defects, corrected from B3 data only: (1) each BDR row of the lake's `gold/jumps.parquet` (an opening gap the
  lake flags but did not resolve: L1RC34's 10:1 split of 03/10/2024, D1EL34's VMware spin-off of 03/11/2021) sets that
  session's return to 0; (2) a session of -30% or worse, or +40% or better, that the next session undoes (the two-day
  product within 10% of 1, e.g. AVGO34 and PYPL34 on 17/09/2021) is a print error: both returns set to 0.
- Fundamentals: SEC EDGAR, the lake's US export (`D:/b3-datalake/exports/us_stocks/fundamentals.parquet`), each value
  usable from its filing date (`available_from`) only.
- Market value (for the value yields only, never for returns): the export's `market_caps.parquet` (SEC share counts
  times the US close). It is the one scraped input; it is checked against B3 before the run's numbers are read: its
  monthly change in reais (times the PTAX change, BCB) against the BDR's own month on B3, median absolute gap printed.
  Names without a market value (multi-class and 20-F filers: GOOGL, META, BRK.B, DELL, BABA, BIDU, JD, PDD, BILI, SE)
  get the neutral value rank, the code's rule for a missing yield.
- Dividends in the score: none (B3 lists only a BDR's last ~3 events, the US files are scraped): the 12-month dividend
  yield is 0 for every name, so it ranks nothing.
- Sectors: the export's SEC-based sectors (`sectors.parquet`).

## The map BDR -> SEC filer

`docs/dados/bdr-sec-mapa-2026-11-14.csv`, made by hand from the names of the 133 BDRs (ETF BDRs, ticker 39, have no
fundamentals and are left out) that ever met the liquidity rule since 2020. Left out: BDRs of Brazilian or Latin
companies (XPBR31, INBR32, ROXO34, STOC31/34, PAGS34, AURA33, G2DI33, GPIV33, NUBR33, WSON33, JBSS32, EXCO32) and
issuers without SEC fundamentals in the export (TSMC34, N1VO34, ASML34, A1ZN34, B1NT34, RIOT34, S1PO34, S2GM34).
FBOK34 is Meta before its rename to M1TA34; GOGL34 and GOGL35 are both Alphabet (one issuer, the rule keeps one class).

## The rule

- Decision at each month's first session from 04/01/2021, with data through the previous close.
- Eligible: a mapped BDR traded on at least 90% of the last 252 sessions with a median daily volume of at least
  R$ 200 thousand (the 14/11 sleeve's sensitivity level; R$ 1 million leaves 13-44 names, too few for 15 after the
  score's own filters), with the Composite score's inputs complete (36 monthly returns for the volatility, 12-1 momentum).
- Score: `scripts.paper_v3.nota_do_composto` unchanged (Conservadora + value + profit growth, attempt 51's score, the
  Composite's own), on monthly BDR returns from B3, the SEC fundamentals and the market values above, with the BDRs
  under their US security codes so the issuer, sector and fundamentals keys are the US test's.
- Book: `core.sinais.carteira_da_nota(nota, sectors, 15, POR_SETOR)`, equal weights in the 15 best (at most 4 a
  sector, one class per issuer); fewer eligible, equal weights in those.
- Arms (6 rows): `Comp90+CompBDR` ... `Comp50+CompBDR` (the B3 Composite curve and this book, back to the mix at each
  decision) and `CompBDR` alone. Costs and tax as in DECISOES 366-368: 50 bps of each BDR trade, 10 bps of the Composite
  leg's traded value, 15% of each BDR sale's gain (`core.manga_bdr.simular`).
- Window 04/01/2021 to 16/10/2026 (the end of the saved Composite curves); phases 0 / 21 / 42.
- Measures and gate: `scripts.xquad_avaliar.metricas`; N for the DSR = 188 (182 + 6). Gate per arm: P[> Composite]
  >= 0.95 and DSR >= 0.95 in every phase. An arm that passes becomes a forward ghost account only.

## Expectation (before the run)

The B3 Composite's score did not carry to the US test (DECISOES 237) and a joint BDR ranking made 12-15% against the
Composite's 20-21% (DECISOES 314-316). Expected: `CompBDR` alone below the Composite, the mixes a little below it with
a lower MDD at 90/10-80/20 (the dollar), no arm passing.
