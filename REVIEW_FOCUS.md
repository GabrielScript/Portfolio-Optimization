# Review focus: the live path, from the Composite's record to the orders and the reports

The backtest says what the Composite would have earned; this branch decides what is actually held: the paper-v3
record (the portfolio of each decision, `paper-v3/composto.csv`), the fictitious account, the MT5 demo executor and
the 20:00 email. A defect here makes the live book differ from the tested rule. `core/paper.py`, `core/market_data.py`,
`core/bcb.py` and `assets.py` are guarded and not here.

Look first for, in this order:

1. **The record matches the rule** (`scripts/paper_v3.py`, `scripts/paper_v2.treino_do_dia`): the decision day, the
   training window and the score are the backtest's; nothing after the decision close leaks in; a rerun never writes
   a second record for the same decision.
2. **Orders** (`core/ordens.plano_de_compra`, `core/mt5_book`, `scripts/mt5_executor.py`, `core/mt5_corretora`): whole
   shares, the 20% cap, lots of 100 plus the fractional market, the 0.5% limit band, cash never exceeded, sells
   before buys, a resend never doubling a position, a real (non-demo) account always refused.
3. **The fictitious account** (`core/simulador`, `scripts/manutencao.ordens_do_simulador`): trades at the record's
   session close, costs, the Selic on cash, the 15% tax on BDR/ETF gains, a missing price.
4. **Prices and reports** (`core/precos_do_app`, `core/resultado_diario`, `scripts/resultado_diario.py`): splicing the
   base and recent price files, the 20:00 preview (today's MT5 close moves on the last session's total-return prices),
   the book's value and its day-by-day history.
