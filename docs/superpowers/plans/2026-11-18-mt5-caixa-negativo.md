# MT5 book: sell what a resend needs when the cash is negative (2026-11-18)

The user chose option 1 of three: "Pode seguir com a opção 1".

## Problem

The 13/11 first buy was sized at the last price and filled at the +0.5% limit: the book spent R$ 101,798.33 of its
R$ 100,000 and its cash is -R$ 1,798.33 (DECISOES 365, item 9; the sizing was fixed on 14/11). MLAS3F did not fill,
so every session at 17:50 the executor wants to resend about R$ 158 of MLAS3, `check_cash` refuses it, the run does
not count as an attempt, and the task exits 1 every day until the next Composite record.

## Rule

`core.mt5_book.trim_for_cash(target, held, weights, quotes, cash)`: while `check_cash` refuses the orders from
`held` to `target`, take one share off the target of the name furthest above its weight (target value share minus
record weight). The result is the smallest cut, sold from the overweight names, that makes the buys fit the
cash plus the sells at the limit prices. The target is unchanged when the orders already fit. A target cut to zero
everywhere is returned as it is, and `check_cash` still refuses it.

The executor calls it before the cash check on every run that trades. A trimmed target is appended to `alvos.csv`
(read last-wins) so the book's target is what was sent. `check_cash` stays as the guard.

## Steps

- [ ] Test the pure rule (fails first), then implement it.
- [ ] Test the executor's resend on a negative-cash book (fails first: refused today), then wire it.
- [ ] Run the MT5 tests (`test_mt5_book`, `test_mt5_executor`, `test_mt5_corretora`, `test_conta_mt5`).
- [ ] `--simular` on the real demo book: read-only, check the orders.
- [ ] Merge to main and push before the 17:50 run. DECISOES entry.
