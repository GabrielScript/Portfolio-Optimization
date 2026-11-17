# The MT5 demo book on the Aleph Quant screen

Date: 2026-11-17. Asked by the user: "Se adicionarmos o saldo do MT 5 no app streamlit do Aleph Quant?" then "Sim, por
favor". Branch `feat/conta-mt5-app` (worktree `../po-mt5app`).

## Problem

The MT5 demo book (XP demo, BRL 100,000 fictitious, pure Composite, DECISOES 355-365) keeps its value only on the
user's PC: `paper-mt5/` was never committed, so Streamlit Cloud cannot read it. The 21:00 email is the only place the
user sees it.

## Design

1. **Published data: two files only.** `paper-mt5/patrimonio.csv` (date, value; written by `scripts/resultado_diario.py`
   at 21:00) and `paper-mt5/fills.csv` (the book's deals, rewritten from the terminal on each run). Neither holds the
   login, account number or server (checked 17/11). `.gitignore` ignores the rest of `paper-mt5/` and so keeps the
   `PARAR` stop file, the orders and the targets local.
2. **Who commits:** a new step "conta MT5" of `scripts/manutencao.py`, after the simulator steps and before the push:
   commits the two files when they differ from HEAD, message `data: conta demo MT5 até DD/MM/YYYY (manutenção)`.
   The upkeep already runs after the 21:00 report (22:30, 23:30, next day 10:00 and 16:00) and already pushes. The
   live order path (`scripts/mt5_executor.py`) and the report are not touched.
3. **Pure part:** `core/conta_mt5.py` builds the screen's data from the two files: the value curve, the day and
   since-start changes (`core.resultado_diario.linha` on BRL 100,000), the cash (`core.mt5_book.cash_after` over the
   fills), and the positions (net shares by name, lot and fractional merged, with the average buy price). None
   without a value file. `INICIAL` is checked equal to the executor's by a test, not imported (the executor is the
   order path).
4. **Screen:** `ui/conta_mt5.py`, a new panel right after the simulator: value with the since-start change, the day,
   the cash (flagged when negative: the 16/11 send was refused for it), the Composite simulator account over the same
   dates, the curve in reais, the positions table, and the last date. Says plainly it is a demo account with
   fictitious money, not the real 19/12 buy.
5. **Loader:** `ui/servicos.carregar_conta_mt5`, cached, keyed by the files' modification times.

## Not in scope

The book's market value per position (the screen shows shares and cost; the value is the 21:00 total), any change to
the executor, the report or the email.

## Tests (each seen failing first)

Positions and average price; cash from fills; None without files; the initial amount equals the executor's; the
upkeep step commits exactly the two files and says "em dia" when nothing changed; `.gitignore` publishes the two files
and keeps `PARAR` out; the panel renders with and without data (`AppTest`).
