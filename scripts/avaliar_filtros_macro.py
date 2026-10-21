"""
Macro filters on the core (the user's request of 2026-10-21; DECISOES 101 and 102 step 5): does a veto from the
equity risk premium, a veto from a credit proxy, or a meta-label built on the two, switching BOVA11 to the CDI, add
value beyond its own average exposure? It does not change the paper trading (paper-v1) nor the Conservadora
(paper-v2). Network only if the frozen CDI copy is unusable.

Usage: .venv/Scripts/python -X utf8 scripts/avaliar_filtros_macro.py > docs/filtros-macro-2026-10-21.txt

Pre-registration (2026-10-21, written and committed before running; after that commit this text does not change):
- Data, each checked against its folder's SHA256SUMS (the script stops on a mismatch): BOVA11 from
  `docs/dados/2026-10-19/universe_prices.parquet` (its sessions are the calendar); the CVM fundamentals and market
  values (`docs/dados/2026-10-20-cvm/`), the point-in-time universe (`docs/dados/2026-10-18/`); the IDA-DI from
  ANBIMA Data's own history file (`docs/dados/2026-10-21-anbima/`, not in git, see its LEIAME); the daily CDI and the
  real curve (`docs/dados/2026-10-21-macro/`, the lake's `cdi_daily.parquet` and `real_curve.parquet`: NTN-B
  Principal sell rates from the Tesouro's CDN workbooks, interpolated linearly in years at 10 years, the coupon NTN-B
  where no pair of Principals brackets it; DECISOES 110). Levels on the BOVA11 sessions as in
  `scripts/avaliar_acoes_titulos.py`.
- Equity risk premium of month M (`monthly_erp`): the earnings yield of the point-in-time universe of M (sum of the
  12-month `net_income_parent` of the documents received at the CVM before the 1st of M, at most 18 months old, over
  the sum of the issuers' market values at the close of M-1; issuers missing either are out; a loss counts as it is)
  minus the 10-year real rate on the last date before M (the Tesouro's morning quote of day d is public on d; the CDN
  lags ~2 months, so Sep-Oct 2026 use 21/08/2026). Monthly from 07/2011 (the first month with the CVM market values of
  most issuers).
- Credit proxy at the close of d (`credit_excess`): the IDA-DI's return over the 63 sessions ending d-1 minus the
  CDI's over the same sessions (the IDA-DI of d is published ~19h, after the close). It is not a spread level: ANBIMA
  publishes no free history of spreads, and rebuilding one from the daily snapshots would need one request per day,
  which ANBIMA's terms (2.4) forbid; the proxy is the realised carry minus the losses of widening spreads, a lagging
  credit momentum. Recorded as such in DECISOES 110.
- Three configurations, counted in `docs/tentativas.csv` (N, already with the five rows of this commit: 38), each
  holding BOVA11 or the CDI whole, decided at a close and held from the next session:
  - premium veto: the CDI in month M when its premium is below the median of the premiums of all the months before
    it (expanding median, from 07/2011; never the full sample);
  - credit veto: the CDI when the credit proxy at the close is below 0 (a fixed threshold, nothing estimated);
  - meta-label: the CDI when P[BOVA11 beats the CDI in the month] < 0.5, from a logistic regression (scikit-learn
    defaults: L2, C = 1) on the two standardised features of every month before the decision's month (features of
    month M: its premium and the credit proxy at the last close before M; label: 1 if BOVA11 beat the CDI from the
    last close before M to the last close of M), refitted at each decision; at the decision, the month's premium and
    the proxy at the close. With fewer than 36 months to learn from, no veto; with one class only, that class.
- Engine: `walk_forward` on [BOVA11, CDI], decisions every 21 sessions from the first close since 01/07/2014 (36
  months of premiums and labels after 07/2011) moved 0, 7 or 14 sessions (the 3 phases), to 16/10/2026 (~12.3
  years); 10 bps per side on each switch; R$ 10 thousand.
- The comparison that tells timing from exposure: each filter against its timing control, the fixed mix of BOVA11
  and the CDI at the filter's average BOVA11 weight over its decisions (in the same phase), back to target at the
  same closes. Also against BOVA11 and the CDI held, on the same dates.
- Measures per phase: CAGR, MDD, average BOVA11 weight, switches; Sharpe over the CDI, IR against BOVA11, P[>
  BOVA11] arithmetic and in log, DSR of the IR (V = the variance of the three filters' daily IRs), MinTRL; P[>
  control] arithmetic and in log; CAGR excess over the control in the subperiods fixed now: 2014–2018 (from the first
  decision), 2019–2022 and 2023–2026 (to 16/10/2026).
- Decision fixed now, for each filter (`scripts.avaliar_nucleo.decidir` on the control): candidate if P[> control] in
  log >= 0.95 in the 3 phases (condition 1) and the CAGR excess over the control is positive in at least 2 of the 3
  subperiods in every phase (condition 2). A candidate goes to a final proof on the core of the real-money plan,
  with the user's OK. Not a candidate = no veto on the core; the two features stay registered for the meta-labeling
  of the S2 strategies after the v1.0 freeze (DECISOES 102 step 5), whose own pre-registration adds its own N. No
  other threshold, tenor, window or model is tried without a new pre-registration, counted in N.
- Limitations written here: ~12 years and ~150 decisions; the ERP's numerator is the last document's 12-month profit
  (no forecasts; the Focus survey has no market earnings); the universe of 2026-10-18 lacks the 71 security-months
  the lake's CODBDI 58 fix added (DECISOES 106; GOLL4, AZUL4, RPMG3, ENEV3, MMXM3, VAGV4); the label horizon (the
  month) is not the holding period (21 sessions); taxes are not modelled (each switch out of BOVA11 realises gains).
- Expectation written down: the premium veto out of stocks in 35-65% of the decisions, the credit veto in 10-25%, the
  meta-label anywhere from 0 to 60%; every filter with a smaller MDD than BOVA11; P[> control] in log between 0.2 and
  0.8 for the three (timing rules rarely beat their own average exposure out of sample); no candidate.
"""

from __future__ import annotations

import sys
import time
from collections.abc import Callable, Iterable, Mapping
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.backtest import ResultadoBacktest, retornos_da_carteira, walk_forward  # noqa: E402
from core.config import ConfigBacktest  # noqa: E402
from core.metrics import cagr, drawdown  # noqa: E402
from core.robustez import psr_da_diferenca  # noqa: E402
from scripts.avaliar_acoes_titulos import (ANBIMA_FOLDER, CDI_YEARS, LAST_SESSION, MACRO_FOLDER,  # noqa: E402
                                           PRICES_FOLDER, cdi_index, on_sessions, read_anbima)
from core.cdi_lago import daily_cdi  # noqa: E402
from scripts.avaliar_nucleo import decidir, excessos_por_subperiodo  # noqa: E402
from scripts.avaliar_robustez import PASTA_UNIVERSO, _tabela, conferir_hashes, contar_tentativas, tabela_do_periodo  # noqa: E402
from scripts.avaliar_valor import carregar_balancos  # noqa: E402

EARNINGS = "net_income_parent"      # the controlling shareholders' 12-month profit, as the market value is theirs


def earnings_yield(fundamentals: pd.DataFrame, market_caps: pd.Series, issuers: Iterable[str]) -> float:
    """Sum of the 12-month profits over sum of the market values of the ``issuers`` that have both (a loss counts as
    it is): the market's E/P, value-weighted by construction."""
    profits = fundamentals[EARNINGS] if EARNINGS in fundamentals.columns else pd.Series(dtype=float)
    both = [code for code in dict.fromkeys(issuers)
            if pd.notna(profits.get(code)) and pd.notna(market_caps.get(code))]
    return float(profits[both].sum() / market_caps[both].sum())


TENOR = 10      # years: the real rate the E/P is set against


def real_rate_before(curve: pd.DataFrame, month: pd.Timestamp, tenor: int = TENOR) -> float:
    """The lake's constant-maturity real rate at ``tenor`` on the last date before ``month`` starts (the Tesouro's
    morning quote of day d is public on d); the last one known, however old (the CDN lags ~2 months)."""
    rows = curve[(curve["tenor_years"] == tenor) & (curve["date"] < month)]
    return float(rows.sort_values("date")["real_rate"].iloc[-1])


CREDIT_SESSIONS = 63


def credit_excess(ida: pd.Series, cdi: pd.Series, sessions: int = CREDIT_SESSIONS) -> pd.Series:
    """The credit proxy known at each close: the IDA-DI's return over the ``sessions`` ending the session before,
    minus the CDI's over the same sessions (both levels on the B3 sessions). Not a spread level: the realised carry
    minus the losses of spreads that widened, a lagging credit momentum."""
    return (ida / ida.shift(sessions) - cdi / cdi.shift(sessions)).shift(1)


MINIMUM_MONTHS = 36     # of history before the premium's median or the meta-label's model is trusted


def erp_veto(erp: pd.Series, month: pd.Timestamp, minimum: int = MINIMUM_MONTHS) -> bool:
    """True = stocks out in ``month``: its premium is below the median of the premiums of all the months before it
    (an expanding median, never the full sample); with fewer than ``minimum`` months before, no veto."""
    before = erp[erp.index < month].dropna()
    return len(before) >= minimum and bool(erp[month] < before.median())


def monthly_labels(bova: pd.Series, cdi: pd.Series) -> pd.Series:
    """By month (first day): 1 if BOVA11 beat the CDI from the last close before the month to its last close, else
    0. The first month of the levels has no close before it and no label."""
    month_ends = pd.DataFrame({"bova": bova, "cdi": cdi}).groupby(bova.index.to_period("M")).last()
    growth = month_ends / month_ends.shift(1)
    beat = (growth["bova"] > growth["cdi"]).astype(int)[growth["bova"].notna()]
    return beat.set_axis(beat.index.to_timestamp())


FEATURES = ("erp", "credit")


def meta_label_probability(features: pd.DataFrame, labels: pd.Series, month: pd.Timestamp,
                           now: pd.Series) -> float:
    """P[BOVA11 beats the CDI this month] from a logistic regression (scikit-learn defaults: L2, C = 1) on the
    standardised ``FEATURES`` of every month before ``month`` with a label (a month's label is known only after it
    ends); ``now`` = the features known at the decision. Fewer than ``MINIMUM_MONTHS`` such months: 1.0 (no veto);
    one class only: its value."""
    train = features.join(labels.rename("label"), how="inner")
    train = train[train.index < month].dropna()
    if len(train) < MINIMUM_MONTHS:
        return 1.0
    if train["label"].nunique() == 1:
        return float(train["label"].iloc[0])
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler().fit(train[list(FEATURES)])
    model = LogisticRegression().fit(scaler.transform(train[list(FEATURES)]), train["label"])
    return float(model.predict_proba(scaler.transform(now[list(FEATURES)].to_frame().T))[0, 1])


def signal_strategy(stocks_on: Callable[[pd.Timestamp], bool]) -> Callable[[pd.DataFrame], pd.Series]:
    """An ``Estrategia`` for the engine's ``walk_forward`` on [BOVA11, CDI]: the engine hands it the returns up to the
    close where it decides (weights from the next session on), and ``stocks_on(that close)`` picks BOVA11 or the CDI
    whole."""
    def decide(returns: pd.DataFrame) -> pd.Series:
        on = stocks_on(returns.index[-1])
        return pd.Series({"BOVA11": 1.0 if on else 0.0, "CDI": 0.0 if on else 1.0})
    return decide


def monthly_erp(balances, universe: pd.DataFrame, curve: pd.DataFrame, months) -> pd.Series:
    """The equity risk premium of each month: the point-in-time universe's ``earnings_yield`` (documents received
    before the month, market values at the previous month's close: ``core.valor.Balancos.no_mes``) minus
    ``real_rate_before`` the month. E/P is a real yield (profits grow with prices), so it is set against the real
    rate, not the nominal one."""
    issuers_by_month = universe.groupby(pd.to_datetime(universe["month"]))["issuer"].apply(list)
    values = {}
    for month in months:
        fundamentals, market_caps = balances.no_mes(month)
        values[month] = (earnings_yield(fundamentals, market_caps, issuers_by_month.get(month, []))
                         - real_rate_before(curve, month))
    return pd.Series(values, dtype=float)


def mean_stock_weight(decided: pd.DataFrame) -> float:
    """The filter's average BOVA11 weight over its decisions (the engine's ``pesos_alvo``): the fixed mix with it is
    the control that tells timing from mere average exposure."""
    return float(decided["BOVA11"].mean())


# ---------------------------------------------------------------- run (network only if the frozen CDI is unusable)

IDA_DI_FILE = "IDADI-HISTORICO.xls"
SIGNAL_START = pd.Timestamp("2011-07-01")          # the first month with the CVM market values of most issuers
FIRST_DECISION_FROM = pd.Timestamp("2014-07-01")   # 36 months of premiums and labels after SIGNAL_START
PHASES = (0, 7, 14)                                # sessions the first decision moves; then every STEP
STEP = 21
WARMUP = 60                                        # the engine's minimum training slice; the filters do not use it
CFG = ConfigBacktest(janela_treino=WARMUP, janela_teste=STEP, capital_inicial=10_000.0, custo_bps=10.0)
ERP_VETO = "BOVA11 with the premium veto (E/P - 10y real rate below its expanding median -> CDI)"
CREDIT_VETO = "BOVA11 with the credit veto (IDA-DI - CDI over 63 sessions < 0 -> CDI)"
META = "BOVA11 meta-labelled by premium and credit (logistic, P < 0.5 -> CDI)"
FILTERS = (ERP_VETO, CREDIT_VETO, META)
SUBPERIODS = {"2014–2018": ("2014-07-01", "2018-12-31"), "2019–2022": ("2019-01-01", "2022-12-31"),
              "2023–2026": ("2023-01-01", "2026-10-16")}


def month_of(close: pd.Timestamp) -> pd.Timestamp:
    return close.to_period("M").to_timestamp()


def load() -> tuple[pd.DataFrame, pd.Series, pd.Series, dict[str, str]]:
    """BOVA11, CDI and IDA-DI levels on the BOVA11 sessions from SIGNAL_START's year before to LAST_SESSION, the
    monthly premium and the credit proxy; every file checked against its folder's SHA256SUMS."""
    balances, sha256 = carregar_balancos()
    sha256 |= conferir_hashes(PASTA_UNIVERSO, ["universe_monthly.parquet"])
    sha256 |= conferir_hashes(PRICES_FOLDER, ["universe_prices.parquet"])
    sha256 |= conferir_hashes(ANBIMA_FOLDER, [IDA_DI_FILE])
    sha256 |= conferir_hashes(MACRO_FOLDER, ["cdi_daily.parquet", "real_curve.parquet"])
    table = pd.read_parquet(PRICES_FOLDER / "universe_prices.parquet", columns=["date", "BOVA11"])
    bova = pd.Series(table["BOVA11"].to_numpy(dtype=float), index=pd.DatetimeIndex(pd.to_datetime(table["date"])))
    sessions = bova.loc[SIGNAL_START - pd.DateOffset(years=1):LAST_SESSION].dropna().index
    cdi = daily_cdi(CDI_YEARS, LAST_SESSION.date(), path=MACRO_FOLDER / "cdi_daily.parquet")
    levels = pd.DataFrame({"BOVA11": bova.reindex(sessions), "CDI": on_sessions(cdi_index(cdi), sessions),
                           "IDA-DI": on_sessions(read_anbima(ANBIMA_FOLDER / IDA_DI_FILE), sessions)})
    universe = pd.read_parquet(PASTA_UNIVERSO / "universe_monthly.parquet")
    curve = pd.read_parquet(MACRO_FOLDER / "real_curve.parquet")
    curve["date"] = pd.to_datetime(curve["date"])
    months = pd.date_range(SIGNAL_START, month_of(LAST_SESSION), freq="MS")
    erp = monthly_erp(balances, universe, curve, months)
    credit = credit_excess(levels["IDA-DI"], levels["CDI"])
    return levels, erp, credit, sha256


def stock_rules(levels: pd.DataFrame, erp: pd.Series, credit: pd.Series) -> dict[str, Callable[[pd.Timestamp], bool]]:
    """For each filter: given the close where it decides, BOVA11 (True) or the CDI (False)."""
    labels = monthly_labels(levels["BOVA11"], levels["CDI"])
    credit_at_month_start = pd.Series({month: credit[credit.index < month].iloc[-1] for month in erp.index})
    features = pd.DataFrame({"erp": erp, "credit": credit_at_month_start})

    def meta(close: pd.Timestamp) -> bool:
        now = pd.Series({"erp": erp[month_of(close)], "credit": credit[close]})
        return meta_label_probability(features, labels, month_of(close), now) >= 0.5

    return {ERP_VETO: lambda close: not erp_veto(erp, month_of(close)),
            CREDIT_VETO: lambda close: not bool(credit[close] < 0.0),   # no proxy yet (NaN): no veto
            META: meta}


def phase_curves(levels: pd.DataFrame, rules: Mapping[str, Callable[[pd.Timestamp], bool]],
                 phase: int) -> tuple[dict[str, ResultadoBacktest], dict[str, ResultadoBacktest]]:
    """The filters, their timing controls (the fixed mix at each filter's average BOVA11 weight) and BOVA11 and the
    CDI held, all deciding at the same closes: the first close since FIRST_DECISION_FROM moved ``phase`` sessions,
    then every STEP."""
    first = int(levels.index.searchsorted(FIRST_DECISION_FROM)) + phase
    returns = retornos_da_carteira(levels[["BOVA11", "CDI"]].iloc[first - WARMUP:])
    filters = {name: walk_forward(returns, signal_strategy(rule), CFG, name) for name, rule in rules.items()}
    fixed = {f"control of {name}": mean_stock_weight(curve.pesos_alvo) for name, curve in filters.items()}
    fixed |= {"BOVA11": 1.0, "CDI": 0.0}
    others = {name: walk_forward(returns, lambda _, w=weight: pd.Series({"BOVA11": w, "CDI": 1.0 - w}), CFG, name)
              for name, weight in fixed.items()}
    return filters, others


def phase_table(filters: Mapping[str, ResultadoBacktest], others: Mapping[str, ResultadoBacktest],
                rf: pd.Series, n_trials: int) -> pd.DataFrame:
    bova = others["BOVA11"].valores
    table = tabela_do_periodo({name: curve.valores for name, curve in filters.items()}, bova, rf, n_trials)
    for name, curve in filters.items():
        control = others[f"control of {name}"].valores
        table.loc[name, "CAGR"], table.loc[name, "MDD"] = cagr(curve.valores), drawdown(curve.valores)[1]
        table.loc[name, "BOVA11 weight"] = mean_stock_weight(curve.pesos_alvo)
        table.loc[name, "Switches"] = float((curve.turnover.iloc[1:] > 0).sum())
        table.loc[name, "Control CAGR"] = cagr(control)
        table.loc[name, "P[> control]"] = psr_da_diferenca(curve.valores, control)
        table.loc[name, "P log [> control]"] = psr_da_diferenca(curve.valores, control, log=True)
    for name in ("BOVA11", "CDI"):
        table.loc[name, "CAGR"], table.loc[name, "MDD"] = cagr(others[name].valores), drawdown(others[name].valores)[1]
    return table


def main() -> int:
    started = time.perf_counter()
    levels, erp, credit, sha256 = load()
    rules = stock_rules(levels, erp, credit)
    n_trials = contar_tentativas()
    print(f"Files (sha256): {sha256}")
    print(f"Premium {erp.index[0]:%m/%Y} to {erp.index[-1]:%m/%Y}: median {erp.median():.2%}, "
          f"min {erp.min():.2%}, max {erp.max():.2%}; credit proxy < 0 on {(credit < 0).mean():.0%} of the sessions; "
          f"DSR N = {n_trials}")
    p_log = {name: [] for name in FILTERS}
    excesses = {name: [] for name in FILTERS}
    for phase in PHASES:
        filters, others = phase_curves(levels, rules, phase)
        dates = others["BOVA11"].valores.index
        rf = levels["CDI"].pct_change().reindex(dates).fillna(0.0)
        table = phase_table(filters, others, rf, n_trials)
        by_subperiod = pd.DataFrame({name: excessos_por_subperiodo(filters[name].valores,
                                                                   others[f"control of {name}"].valores, SUBPERIODS)
                                     for name in FILTERS}).T
        for name in FILTERS:
            p_log[name].append(float(table.loc[name, "P log [> control]"]))
            excesses[name].append(by_subperiod.loc[name])
        print(f"\n=== Phase {phase}: {dates[1]:%d/%m/%Y} to {dates[-1]:%d/%m/%Y}, "
              f"{len(others['BOVA11'].pesos_alvo)} decisions ===")
        print(_tabela(table, ["CAGR", "MDD", "BOVA11 weight", "Control CAGR"]))
        print("CAGR excess over the timing control by subperiod:")
        print(_tabela(by_subperiod, list(SUBPERIODS)))
        print(f"(phase {phase} done in {time.perf_counter() - started:.0f} s)", flush=True)
    for name in FILTERS:
        verdict = decidir(p_log[name], excesses[name])
        print(f"\n{name}: condition 1 (P log [> control] >= 0.95 in the 3 phases) "
              f"{'met' if verdict.condicao_1 else 'not met'}; condition 2 (positive excess over the control in >= 2 "
              f"of 3 subperiods, every phase) {'met' if verdict.condicao_2 else 'not met'} -> "
              f"{'CANDIDATE' if verdict.candidato else 'not a candidate'}")
    print(f"(total {time.perf_counter() - started:.0f} s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
