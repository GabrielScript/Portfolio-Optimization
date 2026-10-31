# Simulator: the fictitious R$ 10,000 account

Decided by the user on 30-31/10/2026. Follows the Composite's registered portfolios from the 03/11/2026 close on, so
that a live curve exists before any real money depends on it.

## What it holds

| Sleeve | Start | Rule |
|---|---|---|
| Stocks | R$ 9,000 | The last `paper-v3/composto.csv` record, in whole shares |
| Options | R$ 1,000 | CDI until the options test passes; never mixed with the stocks |

The Conservadora runs as a control line with the same mechanism and capital, from `paper-v3/conservadora.csv`. Its
record is decided on the same close as the Composite's, so the two differ only by the signal. BOVA11, IVVB11 and CDI
appear as R$ 10,000 on the same day, with no costs.

## Rules

1. **First buy:** the last record dated up to 03/11/2026, at the 03/11/2026 close. The trigger is "that close is in the
   price files", never "today is 04/11": a failed nightly delays the recording and cannot skip the buy.
2. **The lag is real:** the current record is from 27/10, so the account starts five sessions after paper-v3 and the two
   curves differ by that week's return.
3. **Later records:** each record trades at its own date's close, however late the upkeep sees it.
4. **Sizing:** `plano_de_rebalanceio` on the sleeve's patrimony, minimum one share, the risk module's maximum per stock,
   no sector groups (the record already obeys them).
5. **Costs:** `custo_com_liquidez(lago/liquidez.parquet, Custos(0.0003, 1.0), 10)`, the same as the app's real-money
   plan.
6. **IR is zero by construction:** stocks only, and a rebalance sells at most the sleeve, below the R$ 20 thousand
   monthly exemption. To revisit if the sleeve passes R$ 20 thousand.
7. **Dividends:** prices are total-return, so a dividend stays inside the position until the next rebalance, when the
   gap between the position's value and quantity x close becomes cash. Nothing is lost or counted twice.
8. **Cash:** the rounding leftover, the rebalance gap and the options sleeve earn the CDI. The CDI of the last session is
   not out yet: that day repeats the last known rate.

## The ledger

`simulador/ordens.csv`, committed by the upkeep and append-only: `data, carteira, ticker, ordem, preco, custo`, one row
for each stock held or in the target on that rebalance (`ordem` 0 for one left as it is, so the file holds that day's
close). Positions, cash and the curve are derived, never stored.

Valuation is `quantity x recorded close x index(t) / index(rebalance)`, with both index values from the same read of the
price files. The lake re-anchors its total-return index every night, so an index value stored in the ledger would go
stale, and only the ratio is meaningful.

A late upkeep across an ex-date records a close slightly off the traded price, which shifts the share count by about
one dividend. It is accepted: the alternative is keeping raw prices in the app.

## Where the code lives

- `core/simulador.py`: pure logic (orders due, curve, recent returns, positions). No file access.
- `scripts/manutencao.py`: the "simulador" step, after the price files are updated and before the push. It is injected
  like `gerar_curvas`.
- `ui/simulador.py` and `ui/servicos.carregar_simulador`: the screen, on every plan mode.

The screen shows the day, week, month and since-start returns, the curve in R$, the drawdown and the positions. Before
the first buy it says when the account starts. It says plainly that the account is fictitious and that two weeks of
returns are noise.

## Out of scope

"Alepha's moves", email alerts, the XP positions and the options test are later projects (DECISOES 192). The simulator
never changes a paper record and never reads the Composite's signal itself.
