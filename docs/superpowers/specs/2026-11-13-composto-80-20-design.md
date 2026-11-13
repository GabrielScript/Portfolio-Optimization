# Composite 80/20: the Composite with a 20% Bitcoin sleeve, official on MT5 and a ghost account (2026-11-13)

The user, 2026-11-13, after the daily 2010-2026 description of the Bitcoin sleeve (crypto-datalake
`docs/mix-btc-*.txt`): "Vou de composite 80/20, vamos colocar essa estratégia como a Oficial no MetaTrader 5 e no
fantasma do Aleph Quant". A user decision, not a passed test: every Bitcoin mix failed its registered gate
(P[> Composite] 0.74-0.89 < 0.95) and the 80/20 weight was picked after reading the 2018-2026 grid (hindsight).

## Rule

- 80% in the Composite's record in force (`paper-v3/composto.csv`, its own weights scaled by 0.8) and 20% in one
  Bitcoin ETF listed on B3.
- Back to 80/20 at the first session of every month, and at every new Composite record.
- ETF: **BITH11** (Hashdex, Bitcoin only, 0.70%/yr). October 2026 liquidity snapshot (`lago/liquidez.parquet`): median
  daily volume R$ 8.0 mi, half spread 0.21%. QBTC11 has the same fee and spread at R$ 2.1 mi; IBIT39 costs 0.25%/yr but
  its half spread is 0.64% at R$ 2.3 mi; HASH11 is a crypto index, not Bitcoin only. Its 20% sits exactly on the house
  cap of 20% a name (`LIMITES.peso_maximo_acao`).
- Difference from the backtest: there the stock leg drifted between its 63-session records and the sleeve was reset
  monthly; here each rebalance resets the stock names to the record's weights too (the ghost and MT5 are rebalanced
  books, as the other accounts are).

## MT5 (demo, `scripts/mt5_executor.py`) - reverted the same day

The user, 13/11 ~16:00: "Permanece com a estratégia do composite puro, tanto no meta trader 5 e Aleph Quant, o
Composite 80/20 deixa como estratégia fantasma no meu app". The executor trades the pure Composite again (its
version before this spec); `orders_for(unit_lot=...)` and the first-tick wait stay. What was built:


- The book's target is the 80/20 mix of the last Composite record; its record key is `<Composite date>|<YYYY-MM>`, so
  a new month or a new Composite record is a new record that the executor sends (one resend next session, as before).
- BITH11 has no fractional symbol on the terminal (volume step 1): it is sent whole on its standard symbol; the stocks
  keep the lot of 100 plus the fractional rest.
- Starts with the 13/11/2026 closing call (the book was still empty).

## Ghost account (`core/simulador.py`, "Composto 80/20")

- Record `simulador/composto_80_20.csv` (`data`, `ticker`, `peso`, `base`), written by the nightly upkeep: one record on
  13/11/2026, then one dated the first day of each month (traded at its first session) and one on each new Composite
  record date. R$ 10,000 like the others, the Selic on cash.
- 15% income tax on BITH11's gains when sold (a B3-listed ETF has no R$ 20 mil exemption), as the 12/3 pays on BDRs.

## Not changed

The Composite's record, paper-v3, the real buy of 19/12 (still the Composite unless the user says otherwise), the
other ghost accounts, the paper guards.
