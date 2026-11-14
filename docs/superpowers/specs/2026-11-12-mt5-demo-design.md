# MT5 executor: the Composite traded by itself on XP's demo account (design, 2026-11-12)

Branch `feat/mt5-demo`. The user, 2026-11-12: "Quero você 100% autônomo, emitindo ordens sem minha supervisão e
obedecendo as nossas ordens e prerrogativas do Aleph Quant". Not a trial: no return is read to choose anything, N does
not move. What it measures is execution: what the fictitious account (`paper-v3`, fills at the close with 10 bps)
leaves out.

## What exists (checked 2026-11-12, read only, no order sent)

- XP's MetaTrader 5 simulator, terminal `C:/Program Files/MetaTrader 5 Terminal/terminal64.exe`, server
  `XPMT5-DEMO`, account type DEMO, netting, BRL 1,000,000 fictitious, leverage 1, algo trading allowed.
- Official `MetaTrader5` Python package 5.0.6231 (Windows only: it talks to the running terminal, not to the
  internet). `initialize(path=..., timeout=60000)`; the terminal must be logged in.
- All 15 names of the live book and their fractional symbols (suffix `F`) quote in the session. Standard lot 100,
  fractional lot 1. Quoted spread at 15:30: standard 2 to 48 bps (median 6), fractional 4 to 96 bps (median 13);
  MLAS3 is the widest. `filling_mode` 3, `trade_exemode` exchange.
- The book: `paper-v3/composto.csv` (date, ticker, weight, price), a new record each time `core.paper.rebalanceio_devido`
  says so, written after the close by the nightly routine.

## Behavior

1. **When.** A Windows scheduled task runs `scripts/mt5_executor.py` every weekday at 17:50. It exits doing nothing
   unless the last record of `paper-v3/composto.csv` is newer than the last book it executed
   (`paper-mt5/execucoes.csv`), or an earlier execution left orders unfilled.
2. **The book's money.** BRL 100,000 at the start (the user's choice for the demo), then reinvested: the book's value
   is the market value of the positions it opened plus its own cash in `paper-mt5/caixa.csv`. The rest of the demo
   balance is never touched.
3. **The orders.** Target quantities = weight x book value / last price, whole shares (`core.ordens`' rules: the
   house's cost reserve, at most 20% a name). Orders = target minus held, sells first. Each quantity splits into
   lots of 100 on the standard symbol and the rest on the `F` symbol. Limit orders sent in the closing call
   (17:55-18:00) so the fill is the close the fictitious account uses: buy limit = last price + 0.5%, sell limit = last
   price - 0.5% (2% until 13/11/2026: XP's demo filled each limit order at its limit at once) (the auction's price is one for all; the limit only bounds a broken print). Day orders, filling
   `RETURN`.
4. **After the close.** The next run reads the deals of the day and writes each fill to `paper-mt5/ordens.csv`:
   time, symbol, side, quantity, limit, fill price, the lake's close of the day, the difference in bps, and what the
   order would have cost at R$ 1,000 with XP's per-order fee (R$ 4.90 until the user confirms XP's table). An
   unfilled order is resent once on the next session's closing call, then given up and logged.

## Guardrails (in code, each with a test; any failure sends nothing and logs why)

- The account must be DEMO (`account_info().trade_mode`), server `XPMT5-DEMO`. A real account is refused until a
  later spec adds it.
- Only the names of the current record and the names the book still holds (to sell them).
- The book's buys never exceed its own cash; the book never exceeds BRL 100,000 plus what it earned.
- No order on a symbol whose quote is zero or older than 5 minutes.
- A file `paper-mt5/PARAR` stops everything (the user's switch).
- One execution per record (idempotent): rerunning the same day does not trade twice.

## Not in scope

- The real account (a later spec, after clean demo rebalances; the user decides when).
- Dividends: the demo may not credit them. The book is compared with `paper-v3` on price and on total return, and
  the gap is reported, not hidden.
- The app (Streamlit Cloud runs Linux and cannot import `MetaTrader5`; the package stays out of `requirements.txt`,
  in `requirements-mt5.txt`).
- Any MCP server: the executor is plain code; no AI model decides an order.

## Report

`scripts/mt5_executor.py --leitura`: the book's curve against `paper-v3`'s Composite from the first execution,
fills vs close in bps (mean, by name, standard vs fractional), unfilled orders, and the cost at R$ 1,000.

## Plan (tests first, one at a time, each seen failing)

1. `core/mt5_book.py`, pure functions, no `MetaTrader5` import: lot split, target quantities, order diff, guardrail
   checks, ledger updates. Tests with plain data.
2. `core/mt5_corretora.py`: a thin adapter over the package behind a `Protocol`; tests with a fake terminal (refused
   account, zero quote, partial fill, PARAR file, rerun of the same day).
3. `scripts/mt5_executor.py`: the run and `--leitura`; a dry run (`--simular`) that prints the orders without sending.
4. First live step: a dry run during a session, shown to the user; then one real demo rebalance at the closing call;
   then the scheduled task.
