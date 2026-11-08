# Simulator options sleeve: the fictitious R$ 1,000 trades options (pre-registration)

_2026-11-05. User (05/11): "Aprovado o veto por opções, o uso de derivativos está devidamente autorizado, desfaça a
proibição agora! Vamos usar o dinheiro fictício para os derivativos e opções." Written and committed before any
option price, signal or return was read for this design. What was looked at: the lake's `b3.option_daily` and
`b3.option_greeks` schemas and their last date (2026-11-04). The rules below are the author's calls (autonomous mode,
CLAUDE.md), kept for the user to review; the user's two ideas are DECISOES 191 (a) and (b)._

## What changes

- The simulator's options sleeve (R$ 1,000, `docs/superpowers/specs/2026-10-31-simulador-design.md`) stops waiting
  in the CDI for a test: from its first cycle it buys options with the rules below. It stays fictitious and separate
  from the stocks sleeve; the Conservadora control line keeps its sleeve in the CDI, so the two lines differ by the
  options sleeve too and the screen shows the sleeve's own curve against the CDI.
- Nothing about real money changes: real money waits for a forward result agreed beforehand (DECISOES 192).
- The options veto on the Conservadora (DECISOES 190, `2026-10-29-opcoes-veto-design.md`) is built as written, as
  research; this sleeve does not depend on it and does not use its signals.

## Rules

1. **Only buying calls and puts.** The most the sleeve can lose is the premium paid. No option selling, no margin, no
   futures, no short selling (futures are out at this size, DECISOES 194).
2. **Cycle:** the first B3 session of each month, at its close (the first cycle is the first session close after this
   code is live). **Budget per cycle:** 10% of the sleeve's value that day, half for each idea; the rest earns the CDI.
3. **Liquid underlyings:** stocks of the app's universe whose options traded, on average, at least R$ 1 million a day
   over the 21 sessions before the cycle (all strikes and expiries).
4. **Idea A, a put on a stretched stock near its peak:** among the liquid underlyings, those whose close is at least
   95% of their highest close of the last 252 sessions; the one with the highest 252-session total return. Its put
   with the strike nearest 0.95 x the stock's close.
5. **Idea B, a call on a volatile winner:** among the liquid underlyings in the top third by 63-session realized
   volatility, the one with the highest 12-1 momentum (return from t-252 to t-21). Its call with the strike nearest
   1.05 x the stock's close.
6. **Contract:** the expiry whose business days left are nearest to 21 within [10, 45] (ties: the earlier); the
   option must have traded that day (trades > 0) and have a closing ask > 0. Bought at the closing ask, in lots of
   100; the quantity is the largest lot that fits the idea's half of the budget; if one lot does not fit, no order
   (the money stays in the CDI and the reason is recorded).
7. **Holding:** to expiry, never closed early. Daily value = quantity x the closing bid-ask mid when 0 < bid <= ask,
   else the option's close, else the last known value. At expiry the value is the intrinsic value at the stock's close
   of the expiry session (B3 exercises in-the-money options automatically); it becomes cash.
8. **Costs and tax:** the stocks sleeve's cost function on each premium (`Custos(0.0003, 1.0)`) plus paying the ask;
   no cost on expiry. IR 15% on each month's positive net result of the sleeve's expired options, after losses carried
   from earlier months (swing trade options; no R$ 20 thousand exemption).
9. **The ledger:** `simulador/opcoes.csv`, append-only like `ordens.csv`: `data, ideia, ticker, ativo, tipo, strike,
   vencimento, quantidade, preco, custo, motivo` (a cycle without an order writes its reason).

## Amendments while building (2026-11-05, still before any option price was read)

- **A separate account, not a sleeve.** The stocks accounts already hold all their R$ 10,000 in stocks since 03/11
  (`CAPITAL_OPCOES = 0`, DECISOES 243), and their clocks stay untouched. The options account is its own fictitious
  R$ 10,000 ("Opções"), shown beside the Selic; 10% a month is then about R$ 500 per idea, which fits lots of 100.
- **Marks are stored.** The screen runs on the cloud, where the lake's chains do not exist: the upkeep appends each
  bought option's daily unit value to `simulador/opcoes_marcas.csv` (`data, ticker, valor, vencida`), and the curve
  is replayed from the ledger and the marks.
- **IR on the expiry session.** Both ideas expire on the month's B3 expiry, so the month's net result is the expiry
  session's; 15% of it above the losses carried is paid that session (the curve never changes with the reading day).
- **Costs:** the stocks' fees fraction (0.03%) on the premium; the ask already pays the spread.
- **First cycle:** the 06/11/2026 close (`INICIO`).
- **A real two-sided quote (after the dry run, the only change it caused).** A dry run of the plumbing from 01/10
  (nothing recorded) bought VAMO3's call at a R$ 3.00 closing ask against a R$ 0.22 bid: a lone ask, not a price.
  Rule 6 now also requires a closing bid > 0 and ask <= 2 x bid (else the next nearest strike), and a mark uses the
  mid only under the same condition, else the option's close. Nothing else was tuned on the dry run.

## Amendments after the first cycle failed (2026-11-08, no cycle recorded yet)

- **An ask of zero is no price.** The 06/11 22:30 upkeep failed (ZeroDivisionError): 45% of the chain's traded rows
  have a bid and an ask of zero, and "ask <= 2 x bid" let them through. Rule 6 now needs bid > 0 and
  bid <= ask <= 2 x bid (56af121).
- **A band and the most traded strike, sized by the day's volume.** The fixed dry run of 06/11 then bought VAMO3's
  call at a strike 22% away (3.26 against a 2.67 close; every nearer strike had a one-sided quote), R$ 498 of an
  option that traded R$ 30 that day. Rule 6 becomes: the strike inside 0.92-0.98 (put) or 1.02-1.08 (call) x the
  close; on the expiry nearest 21 business days among those candidates, the one with the largest traded volume that
  day (ties: nearest the target, then the lower strike); the quantity never above that day's traded volume. Shared
  with the deep account (`2026-11-08-opcoes-deep-otm-design.md`). The first cycle stays 06/11/2026: the upkeep trades
  it with the 06/11 close data on its next run, as the catch-up of any missed cycle.

## Data

The lake exports each night `option_chain_recent.parquet` to the app's `lago/`: the last 30 sessions' option rows
(no exercise rows) of the liquid underlyings with 5 to 60 business days left, plus each underlying's raw close
(`ticker, date, underlying, option_type, strike, expiry, du, close, best_bid, best_ask, trades, volume_brl,
spot_close`), and the 21-session option volume per underlying for rule 3.

## Expectation, written before

Buying out-of-the-money options loses on average (DECISOES 194): the expected result is a sleeve below the CDI, and
a month with two expired options loses about 10% of the sleeve. Two weeks or two months of this are noise; the
sleeve is read only after 12 cycles, against the CDI, and that reading decides whether the user's ideas are kept,
changed (as a new pre-registration) or dropped.
