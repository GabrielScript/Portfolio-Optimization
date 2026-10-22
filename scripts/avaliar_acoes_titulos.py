"""
Stocks x bonds in the core (PLAN, S4 item "the stocks x CDI fraction"; DECISOES 102 step 2): does a core of half
BOVA11 and half fixed income beat BOVA11 alone, and should that half be the CDI or the IMA-B (NTN-B, the real curve)?
It does not change the paper trading (paper-v1) nor the Conservadora (paper-v2). Network only if the frozen CDI copy
is unusable.

Usage: .venv/Scripts/python -X utf8 scripts/avaliar_acoes_titulos.py > docs/acoes-titulos-2026-10-21.txt

Pre-registration (2026-10-21, written and committed before running; after that commit this text does not change):
- Data, each checked against its folder's SHA256SUMS (the script stops on a mismatch):
  - BOVA11 from `docs/dados/2026-10-19/universe_prices.parquet` (the lake's daily total-return index); its sessions
    are the calendar.
  - IMA-B from ANBIMA Data's own "Baixar histórico (XLS)" file (`docs/dados/2026-10-21-anbima/`, downloaded once on
    2026-10-21; ANBIMA regenerates it every day, so a later download will not match the hash). The file itself is
    not in git (ANBIMA's terms of use, 8.2); see that folder's LEIAME.
  - The daily CDI (BCB SGS 12) from the lake's `cdi_daily.parquet`, frozen in `docs/dados/2026-10-21-macro/`.
- Alignment: each level on the BOVA11 sessions is the last one on or before the session, so the CDI of a day the B3
  was closed lands in the next session. The IMA-B Number Index of day d is published ~19h of d; the engine trades at
  the close of d, so its return of d is the index's d-1 -> d move, as for BOVA11.
- No fixed-income ETF is in COTAHIST (IMAB11, IRFM11, B5P211 and FIXA11 were searched in every market code, DECISOES
  110), so the bond leg is the IMA-B net of IMAB11's 0.25% a.a. fee (ETFs Brasil; the fund's regulation was not
  read), taken a 252nd per session. The CDI leg has no fee (Tesouro Selic). No ETF of the IMA-B traded before 2018:
  before that the leg is a hypothetical index fund.
- Span: the first session with the three (02/12/2008, the first BOVA11) to 16/10/2026, ~17.9 years; 3 phases (0, 21
  and 42 sessions cut from the start).
- Two configurations, counted in `docs/tentativas.csv` (N, already with the five rows of this commit: 38):
  - "Core BOVA11 50% + IMA-B 50%";
  - "Core BOVA11 50% + CDI 50%".
  Both with fixed weights back to target every 63 sessions and 10 bps per side on every leg (`backtest_pesos_fixos`,
  `CFG_BT`). The 50% is the 1/N of the two legs, chosen without estimating anything on the span (as in the
  BOVA11 + IVVB11 core test). References, not counted: BOVA11, IMA-B and CDI held alone, same engine and dates.
  As in that test, a fixed fraction of this core inside the real-money plan (70% core + 30% satellite) has the same
  P against BOVA11 as the core alone, so only the core is tested.
- Measures per phase: CAGR, vol, MDD, turnover; Sharpe over the CDI; IR against BOVA11; P[> BOVA11] arithmetic and
  in log; DSR of the IR (V = the variance of the two mixes' daily IRs); MinTRL; CAGR excess over BOVA11 in the
  subperiods fixed now: 2008–2014 (from the start), 2015–2020 and 2021–2026 (to 16/10/2026); rolling 36- and 60-month
  windows against BOVA11; P[IMA-B mix > CDI mix] in log.
- Decision fixed now, for each mix (`scripts.avaliar_nucleo.decidir`): candidate if P[> BOVA11] in log >= 0.95 in
  the 3 phases (condition 1) and the CAGR excess is positive in at least 2 of the 3 subperiods in every phase
  (condition 2). The bond leg (`bond_leg`): the IMA-B mix if it is the only candidate, or if both are and P[IMA-B mix >
  CDI mix] in log >= 0.95 in the 3 phases; else the CDI mix. A candidate core goes to the final proof and to a
  paper-v2 alongside, and the user decides the default of the real-money plan (DECISOES 102 step 3). No candidate =
  the core stays BOVA11, the S4 item closes with no fixed-income leg, and the stocks x CDI split stays a risk-profile
  choice of the user, not a backtest result. No other fraction or index is tried without a new pre-registration,
  counted in N.
- Limitations written here: one rate regime (the CDI averaged ~10% a.a., real rates high most of the span), so a pass
  says "fixed income beat Brazilian equities in this regime", not skill; the span starts at the 2008 low of BOVA11;
  taxes are not modelled (BOVA11 and a fixed-income ETF pay 15% on gains when sold, with no monthly exemption; the
  Tesouro Selic pays 22.5% to 15% by holding time); the ETF's tracking error against the index is unknown; each return
  to target realises gains.
- Expectation written down: IMA-B alone with a higher CAGR than BOVA11 and a much smaller MDD; both mixes with
  BOVA11's CAGR or a little more and about half its vol and MDD; P[> BOVA11] in log between 0.55 and 0.85 for both
  (a log excess of ~0.5-1.5% a.a. against a tracking error of ~12% over ~18 years); condition 2 met (positive in
  2008–2014 and 2021–2026, negative in 2015–2020); condition 1 not met; neither mix a candidate; P[IMA-B mix > CDI mix]
  in log between 0.6 and 0.9.
"""

from __future__ import annotations

import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core import market_data  # noqa: E402
from core.backtest import ResultadoBacktest, backtest_pesos_fixos, retornos_da_carteira  # noqa: E402
from core.cdi_lago import daily_cdi  # noqa: E402
from core.config import DIAS_UTEIS_ANO  # noqa: E402
from core.metrics import cagr, drawdown, retornos_de_valores  # noqa: E402
from core.robustez import janelas_moveis, psr_da_diferenca, resumir_janelas  # noqa: E402
from scripts.avaliar_nucleo import comeco_comum, decidir, excessos_por_subperiodo  # noqa: E402
from scripts.avaliar_robustez import (FASES, PORCENTO_JANELAS, _tabela, conferir_hashes,  # noqa: E402
                                      contar_tentativas, tabela_do_periodo)
from scripts.avaliar_universo_dinamico import CFG_BT  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
PRICES_FOLDER = ROOT / "docs" / "dados" / "2026-10-19"
ANBIMA_FOLDER = ROOT / "docs" / "dados" / "2026-10-21-anbima"
MACRO_FOLDER = ROOT / "docs" / "dados" / "2026-10-21-macro"
IMA_B_FILE = "IMAB-HISTORICO.xls"
LAST_SESSION = pd.Timestamp("2026-10-16")       # the frozen prices end there; the IMA-B and the CDI go on
ETF_FEE = 0.0025                                # IMAB11 (It Now IMA-B): 0.25% a.a., the stand-in for the bond ETF
MIX_IMA = "Core BOVA11 50% + IMA-B 50%"
MIX_CDI = "Core BOVA11 50% + CDI 50%"
MIXES = (MIX_IMA, MIX_CDI)
WEIGHTS = {MIX_IMA: {"BOVA11": 0.5, "IMA-B": 0.5}, MIX_CDI: {"BOVA11": 0.5, "CDI": 0.5},
           "BOVA11": {"BOVA11": 1.0}, "IMA-B": {"IMA-B": 1.0}, "CDI": {"CDI": 1.0}}
SUBPERIODS = {"2008–2014": ("2008-12-01", "2014-12-31"), "2015–2020": ("2015-01-01", "2020-12-31"),
              "2021–2026": ("2021-01-01", "2026-10-16")}
WINDOW_MONTHS = (36, 60)
CDI_YEARS = 22
P_MINIMUM = 0.95
_TOLERANCE = 1e-9


def anbima_levels(grid: pd.DataFrame) -> pd.Series:
    """An ANBIMA Data history sheet read without a header (row 0 = column names; then index name, reference date as
    "YYYY-MM-DD 00:00:00" text, Number Index, ...) -> the daily Number Index by date. The index name column is not
    read: its spelling changes inside one file ("IMA-B", later "IMA - B")."""
    body = grid.iloc[1:]
    dates = pd.to_datetime(body.iloc[:, 1].astype(str).str.slice(0, 10), format="%Y-%m-%d")
    return pd.Series(body.iloc[:, 2].astype(float).to_numpy(), index=pd.DatetimeIndex(dates, name=None),
                     name="level")


def net_of_fee(level: pd.Series, annual_fee: float) -> pd.Series:
    """The index as a fund tracking it with no other cost would be quoted: ``annual_fee`` taken a 252nd at a time,
    one per session of ``level`` (the B3 sessions, once aligned)."""
    sessions = pd.Series(range(len(level)), index=level.index, dtype=float)
    return level * (1.0 - annual_fee) ** (sessions / 252)


def cdi_index(cdi_daily: pd.Series) -> pd.Series:
    """Daily CDI (decimal) -> a total-return level, the CDI of day d in the level of d (the convention of
    ``market_data.serie_rf_diaria``)."""
    return (1.0 + cdi_daily.sort_index()).cumprod()


def on_sessions(level: pd.Series, sessions: pd.DatetimeIndex) -> pd.Series:
    """``level`` on the B3 ``sessions``: the last level on or before each one, so what accrued on a day the B3 was
    closed lands in the next session; empty before the first level."""
    return level.reindex(level.index.union(sessions)).ffill().reindex(sessions)


def bond_leg(ima_candidate: bool, cdi_candidate: bool, p_log_ima_over_cdi: Sequence[float]) -> str | None:
    """The fixed-income leg of the core when some mix is a candidate: the IMA-B one only if it is the sole candidate
    or beats the CDI one (P[IMA-B mix > CDI mix] in log >= 0.95) in every phase; otherwise the CDI, the simpler leg
    with no duration risk. None when neither mix is a candidate."""
    if ima_candidate and (not cdi_candidate or all(p >= P_MINIMUM - _TOLERANCE for p in p_log_ima_over_cdi)):
        return "IMA-B"
    return "CDI" if cdi_candidate else None


# ---------------------------------------------------------------- run (network only if the frozen CDI is unusable)

def read_anbima(path: Path) -> pd.Series:
    """One frozen ANBIMA Data history file (an .xlsx named .xls: read from its bytes, so the format is sniffed)."""
    import fastexcel  # research-only dependency (requirements-dev.txt)
    sheet = fastexcel.read_excel(path.read_bytes()).load_sheet(0, header_row=None)
    return anbima_levels(sheet.to_pandas())


def load() -> tuple[pd.DataFrame, dict[str, str]]:
    """BOVA11, the IMA-B net of the ETF fee and the CDI on the BOVA11 sessions, from the first one with all three
    to ``LAST_SESSION``; every file checked against its folder's SHA256SUMS."""
    sha256 = {**conferir_hashes(PRICES_FOLDER, ["universe_prices.parquet"]),
              **conferir_hashes(ANBIMA_FOLDER, [IMA_B_FILE]), **conferir_hashes(MACRO_FOLDER, ["cdi_daily.parquet"])}
    table = pd.read_parquet(PRICES_FOLDER / "universe_prices.parquet", columns=["date", "BOVA11"])
    bova = pd.Series(table["BOVA11"].to_numpy(dtype=float), index=pd.DatetimeIndex(pd.to_datetime(table["date"])))
    sessions = bova.loc[:LAST_SESSION].dropna().index
    cdi = daily_cdi(CDI_YEARS, LAST_SESSION.date(), path=MACRO_FOLDER / "cdi_daily.parquet")
    prices = comeco_comum(pd.DataFrame({"BOVA11": bova.reindex(sessions),
                                        "IMA-B": on_sessions(read_anbima(ANBIMA_FOLDER / IMA_B_FILE), sessions),
                                        "CDI": on_sessions(cdi_index(cdi), sessions)}))
    prices["IMA-B"] = net_of_fee(prices["IMA-B"], ETF_FEE)
    return prices, sha256


def phase_curves(prices: pd.DataFrame, phase: int) -> dict[str, ResultadoBacktest]:
    """The two mixes and the three references, back to target every 63 sessions, on the same dates."""
    returns = retornos_da_carteira(prices.iloc[phase:])
    return {name: backtest_pesos_fixos(returns, pd.Series(weights).reindex(returns.columns).fillna(0.0), CFG_BT, name)
            for name, weights in WEIGHTS.items()}


def phase_table(curves: Mapping[str, ResultadoBacktest], rf: pd.Series, n_trials: int) -> pd.DataFrame:
    """CAGR, vol, MDD and turnover of every curve; Sharpe over the CDI, IR, P[> BOVA11], DSR and MinTRL of the mixes
    (V = the variance of the two mixes' daily IRs)."""
    against_bova = tabela_do_periodo({name: curves[name].valores for name in MIXES}, curves["BOVA11"].valores, rf,
                                     n_trials)
    rows = {name: {"CAGR": cagr(curve.valores),
                   "Vol": float(retornos_de_valores(curve.valores).std() * np.sqrt(DIAS_UTEIS_ANO)),
                   "MDD": drawdown(curve.valores)[1], "Turnover/yr": curve.turnover_anual_medio}
            for name, curve in curves.items()}
    return pd.DataFrame(rows).T.join(against_bova)


def print_phase(phase: int, curves: Mapping[str, ResultadoBacktest], table: pd.DataFrame,
                by_subperiod: pd.DataFrame, p_log_ima_over_cdi: float) -> None:
    mix = curves[MIX_IMA]
    print(f"\n=== Phase {phase}: {mix.valores.index[1]:%d/%m/%Y} to {mix.valores.index[-1]:%d/%m/%Y}, "
          f"{len(mix.pesos_alvo)} rebalances ===")
    print(_tabela(table, ["CAGR", "Vol", "MDD", "Turnover/yr"]))
    print("CAGR excess over BOVA11 by subperiod:")
    print(_tabela(by_subperiod, list(SUBPERIODS)))
    print(f"P[IMA-B mix > CDI mix] in log: {p_log_ima_over_cdi:.3f}")
    values = pd.DataFrame({name: curve.valores for name, curve in curves.items()})
    for months in WINDOW_MONTHS:
        print(f"Rolling {months}-month windows against BOVA11:")
        print(_tabela(resumir_janelas(janelas_moveis(values, "BOVA11", months)), PORCENTO_JANELAS))


def main() -> int:
    started = time.perf_counter()
    prices, sha256 = load()
    n_trials = contar_tentativas()
    print(f"Files (sha256): {sha256}")
    print(f"Span {prices.index[0]:%d/%m/%Y} to {prices.index[-1]:%d/%m/%Y} ({len(prices)} sessions); "
          f"IMA-B net of {ETF_FEE:.2%} a.a.; DSR N = {n_trials}")
    p_log: dict[str, list[float]] = {name: [] for name in MIXES}
    excesses: dict[str, list[pd.Series]] = {name: [] for name in MIXES}
    p_log_ima_over_cdi = []
    for phase in FASES:
        curves = phase_curves(prices, phase)
        rf = market_data.serie_rf_diaria(curves[MIX_IMA].valores.index, prices["CDI"].pct_change().dropna(), 0.0)
        table = phase_table(curves, rf, n_trials)
        by_subperiod = pd.DataFrame({name: excessos_por_subperiodo(curves[name].valores, curves["BOVA11"].valores,
                                                                   SUBPERIODS) for name in MIXES}).T
        for name in MIXES:
            p_log[name].append(float(table.loc[name, "P log"]))
            excesses[name].append(by_subperiod.loc[name])
        p_log_ima_over_cdi.append(psr_da_diferenca(curves[MIX_IMA].valores, curves[MIX_CDI].valores, log=True))
        print_phase(phase, curves, table, by_subperiod, p_log_ima_over_cdi[-1])
        print(f"(phase {phase} done in {time.perf_counter() - started:.0f} s)", flush=True)

    verdicts = {name: decidir(p_log[name], excesses[name]) for name in MIXES}
    for name, verdict in verdicts.items():
        print(f"\n{name}: condition 1 (P log >= 0.95 in the 3 phases) {'met' if verdict.condicao_1 else 'not met'}; "
              f"condition 2 (positive excess in >= 2 of 3 subperiods, every phase) "
              f"{'met' if verdict.condicao_2 else 'not met'} -> "
              f"{'CANDIDATE' if verdict.candidato else 'not a candidate'}")
    leg = bond_leg(verdicts[MIX_IMA].candidato, verdicts[MIX_CDI].candidato, p_log_ima_over_cdi)
    print("Core: " + (f"BOVA11 50% + {leg} 50% (candidate to the final proof and paper-v2)" if leg
                      else "stays BOVA11"))
    print(f"(total {time.perf_counter() - started:.0f} s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
