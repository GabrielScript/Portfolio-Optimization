# Fifteen BDRs, one per high-chance sector, mixed with the Composite (pre-registered 2026-11-14)

The user, 14/11, after the list of 18 sectors for 2030: "Retorne as 15 BDR's da maior variedade de setores possíveis,
pode ser 1 setor em alta por 1 ação ... com o maior crescimento e líquidas a partir de 2021 ... se a BDR tiver dados e
líquidas a partir de 2024 pode usar ... Divida a tabela dos backtests em composite puro, composite 90/10, 80/20, 70/30,
60/40, 50/50 e apenas essas 15 BDR's".

Written and committed before any return of these BDRs is read. Only names, sessions traded and volumes were read.

## What this is, and what it is not

The fifteen names are chosen by their growth from 02/01/2024 to 13/11/2026, and the backtest covers the same years:
the result is good by construction and decides nothing (look-ahead). It is a description of what the user asked for.
No arm can become a ghost account or replace the Composite from it. The window 04/01/2021 to 29/12/2023 is not used by
the selection; it is reported apart as the only read that the selection did not see (the sectors themselves were
still chosen in 2026, with hindsight).

## Sectors, in order of chance by 2030 (the reply of 14/11), then the user's PDF sectors

Each BDR belongs to one sector only. Grade A = high chance, B = medium-high, C = medium.

| Order | Sector | Grade | Candidate BDRs |
|---|---|---|---|
| 1 | AI compute (GPUs, ASICs, HBM, networking, AI servers) | A | NVDC34, AVGO34, A1MD34, MUTC34, A1NE34, M2RV34, D1EL34 |
| 2 | Power for data centers (generation, nuclear, uranium, grid, cooling) | A | V1RT34, G2EV34, V1ST34, E1TN34, C1CJ34, N1RG34, Q1UA34, NEXT34, BURA39, DUKB34 |
| 3 | Chip manufacturing and equipment | A | TSMC34, ASML34, A1MT34, L1RC34, K1LA34, BSOX39 |
| 4 | Obesity and metabolic drugs | A | LILY34, N1VO34 |
| 5 | Enterprise AI agents and cloud | A | MSFT34, GOGL34, AMZO34, ORCL34, SSFO34, N1OW34 |
| 6 | Robotaxi and autonomous driving | A | TSLA34, U1BE34, BIDU34 |
| 7 | Cybersecurity | A | C2RW34, P2AN34, F1TN34, N2ET34, Z2SC34, A1KA34, CSCO34, BBUG39 |
| 8 | Defense (drones, missiles, space sensors) | A | P2LT34, LMTB34, RYTT34, NOCG34, BAER39, SHLD39 |
| 9 | Direct-to-cell satellites and reusable launch | A | SPCX34, ORBX39 |
| 10 | Grid batteries | A | Q2SC34, E2NP34, BLBT39 |
| 11 | Critical minerals (rare earths, copper, lithium) | A | M2PM34, FCXO34, BCPX39, S2GM34, A1LB34, RIOT34, EART39 |
| 12 | Aging economy (senior housing, Medicare) | A | W1EL34, V1TA34, UNHH34, CVSH34, H1CA34 |
| 13 | Digital twins and simulation (PDF sector 24) | A | S1NP34, A1UT34, P2TC34 |
| 14 | Surgical robotics (PDF sector 28) | A | I1SR34, S1YK34, MDTC34 |
| 15 | Biopharma tools and CDMOs | B | TMOS34, DHER34 |
| 16 | Precision agriculture (PDF sector 20) | B | DEEC34, C1TV34 |
| 17 | AI drug discovery | C | BGNO39 |
| 18 | Early diagnostics | C | I1LM34, D1EX34, ABTT34 |
| 19 | Humanoid and industrial robotics | C | BOTZ39 |
| 20 | Machine-to-machine payments (PDF sector 30) | C | C2OI34, PYPL34, X1YZ34 |

Left out by their grade (low chance by 2030): drones and eVTOL, quantum computing, direct air capture, longevity, BCI,
space mining, orbital compute, space habitats, wireless power.

## The pick in each sector

- Window for the pick: B3 sessions from 02/01/2024 to 13/11/2026 (BOVA11's calendar, 718 sessions).
- Liquid: traded on at least 90% of those sessions and a median daily volume, on the sessions it traded, of at least
  R$ 50 thousand. A sector with no liquid candidate takes candidates of at least R$ 10 thousand, flagged "thin". A
  sector with neither is skipped.
- Growth: the total return of `tr_close` from its first close on or after 02/01/2024 to its last close on or before
  13/11/2026 (BDR prices only: the lake has no BDR cash dividends). The best growth in the sector is its pick.
- Sectors are taken in the order above until fifteen are picked.

## The backtest

- Arms: the Composite alone; `Comp90+15BDR`, `Comp80+15BDR`, `Comp70+15BDR`, `Comp60+15BDR`, `Comp50+15BDR`; and
  `15BDR` (the fifteen alone). Six rows beside the Composite.
- The sleeve: equal weights among the fifteen with a price that session, back to (1 - w) Composite and w sleeve at
  each month's first session (the 14/11 BDR sleeve's rule, `core.manga_bdr.simular`). `15BDR` is the same book with
  w = 1 and a flat cash leg that only carries the costs and tax.
- Costs and tax as in the 14/11 BDR sleeve: 50 bps of each BDR trade, 10 bps of the Composite leg's traded value, 15% of
  each BDR sale's gain (no exemption, losses not carried).
- Composite leg: the saved curves of phases 0 / 21 / 42 (`docs/dados/pos-idade-curvas-fase{0,21,42}-2026-11-13.parquet`),
  which end on 16/10/2026; every window ends there.
- Windows:
  - 02/01/2024 to 16/10/2026: all fifteen priced (the headline the user asked for; in-sample).
  - 04/01/2021 to 16/10/2026: a name enters at the first monthly rebalance with a price.
  - 04/01/2021 to 29/12/2023: the years the pick did not read.
- Measures: `scripts.xquad_avaliar.metricas` (CAGR, Vol, Sharpe, Sortino, MDD, Calmar, P[> Composite], DSR, alpha and
  t against the Composite) with BOVA11 and the CDI beside them. N for the DSR = 155 (149 after DECISOES 366 + these 6).
- No gate verdict: the selection reads the window, so P and DSR are reported for the record only.

## Expectation (before the run)

The 2024-2026 window flatters the sleeve by construction: the picks are each sector's best performer, several of them
with 100%+ in the window, so the more BDR weight the higher the CAGR, with a higher volatility and a deeper MDD from the
April 2025 fall. In 2021-2023 (not read by the pick) the sleeve should lose most of that edge: 2022 was a deep fall for
US technology in reais.
