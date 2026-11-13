# Insider dealing, patents at a weight earned by their IC, and the mean of the theta ladder (pre-registered 2026-11-12)

Written and committed before any return-based number. Branch `feat/informados`, from `feat/inovemetria` (it carries
the patent signal and the Fermiônica). The user's order of 2026-11-12: "a negociação de administradores e
controladores, as patentes com peso pequeno e proporcional à força do sinal, média dos tetas".

## Data looked at (no return involved)

CVM open data `cia_aberta-doc-vlmo` (art. 11 filings, "Valores Mobiliários Negociados e Detidos"): one zip a year,
2018 (partial) to 2026, 420,379 movement rows, 43,155 filings, about 450 companies a month; each filing has its
reference month and its delivery date (median 38 days after the reference month starts, 90% within 40); 52,696 cash
purchases and 62,051 cash sales. Frozen in `docs/dados/2026-11-12-vlmo/` with SHA256SUMS.

## Signal `INS` (insider net purchase ratio)

For a decision month M and an issuer: the filings delivered strictly before the first day of M (the latest version
of each company and reference month among those), reference months within the six calendar months before M; rows
of the company's own shares or units (`Tipo_Empresa` Companhia; `Tipo_Ativo` Ações or Units), held by a person or
body (`Tipo_Cargo` filled: board, officers, fiscal council, statutory bodies, controller, and those tied to them;
the company's own treasury rows have no `Tipo_Cargo` and stay out), movements "Compra à vista" and "Venda à
vista". B and S are the BRL volumes bought and sold (`Volume`, or quantity x price when it is missing).
INS = (B - S) / (B + S); an issuer with B + S = 0 has no score. Issuer to filing by the CNPJ root.

**Stage 1.** The house IC harness (`avaliar_valor.ics_e_cobertura`), months with a score (from 2019), gate
IC >= 0.02 and t >= 2; coverage, subperiods and the rank correlation with the Conservadora, value and profit-growth
scores as description. One reading.

## The weight a new score earns (`+w` arms)

score = z(Composite) + lambda_M x z(extra), over the names with a Composite score, z the house normal score among
them, a name without the extra score adding 0. lambda_M = clip(mean IC of the extra score / mean IC of the Composite
score, 0, 1), both means over the same months before M (the IC of a month is known when the month ends), and 0
while there are fewer than 36 such months. Nothing in lambda is read from the future, and nothing is tuned.

## Arms (15 names, <= 4 per sector, the house rules, 10 bps; three phases; each read once)

1. `Comp+Pat-w`: the extra score is the patent family P of the Inovemetria reading (IC 0.024, t 3.56, which failed
   as an equal fourth component, DECISOES 352).
2. `Fermi-mean`: the mean of the four equal-weight books of the log det selection on the Composite score at
   theta 0.25, 0.5, 0.75 and 1.0 (the ladder of DECISOES 350). It answers the selection bias of having picked 0.5
   after 48 rows: no theta is chosen. It may hold more than 15 names.
3. `Comp+Ins-w`: the extra score is INS. Run only if INS passes stage 1.

**Gate (the house's):** P[> Composite] >= 0.95 and DSR >= 0.95 in every phase. Anchors before any arm is read: the
Composite's replay and LogDet-15 t0.5 reproduce their saved curves (1e-9). **N:** 121 after DRL v3, 123 with the
two MARS-R rows still running; these arms take it to 125 or 126, and the DSR is computed at 126.

## Prior, stated before the data

`Comp+Pat-w` and `Fermi-mean`: a tie with the Composite is the likely reading (the first moves one or two names at
a small weight, the second averages books that share 13 of 15 names). INS: insider purchases predict returns in
most published samples, sales do not; seven years of months is a short sample for a t of 2.
