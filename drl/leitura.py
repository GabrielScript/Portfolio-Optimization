"""The reading of a run: each strategy's test years stitched into one out-of-sample curve and measured over the CDI.

    .venv-drl/Scripts/python -m drl.leitura --lago <lake exports> --run <name> [--composite <curves.parquet>]

A strategy enters the table only with every year of the run present and adjoining: a missing cell is an error, never
a shorter curve. ``--composite`` adds the house engine's Composite curve (``docs/dados/comparacao-estrategias.parquet``)
cut to the same sessions and measured by the same function. It is a reference from another simulator (quarterly
rebalance, five-year eligibility window, the fixed rules), which the table says in the row's name; the first version
pasted its numbers from another report, with a different risk-free rate and a drawdown measured on yearly points.
"""

from __future__ import annotations

import argparse
import itertools
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from drl.dados import Panel, load_panel
from drl.executar import LAGO, SAIDAS, Cell, read_cell
from drl.periodo import emendar
from drl.relatorio import CAPM_FIELDS, FIELDS, capm, metrics


def stitch(cells: Sequence[Cell]) -> tuple[pd.Series, np.ndarray]:
    """One curve (indexed by session) and the turnover of every decision, from the cells of one strategy. Each cell
    must start on the session the previous one ended on."""
    ordenadas = sorted(cells, key=lambda cell: cell.year)
    for anterior, seguinte in itertools.pairwise(ordenadas):
        if seguinte.sessions[0] != anterior.sessions[-1]:
            raise ValueError(f"{anterior.strategy}: {seguinte.year} starts on {seguinte.sessions[0]}, but "
                             f"{anterior.year} ended on {anterior.sessions[-1]} (a year is missing)")
    sessoes = list(ordenadas[0].sessions) + [s for cell in ordenadas[1:] for s in cell.sessions[1:]]
    valores = emendar([np.asarray(cell.values) for cell in ordenadas])
    return (pd.Series(valores, index=pd.DatetimeIndex(sessoes)),
            np.concatenate([np.asarray(cell.turnover) for cell in ordenadas]))


def curve_metrics(curva: pd.Series, panel: Panel, turnover: np.ndarray | None = None) -> dict[str, float]:
    """``curva`` on consecutive sessions of the panel, over the CDI of those sessions."""
    inicio = int(panel.sessions.get_loc(curva.index[0]))
    esperado = panel.sessions[inicio:inicio + len(curva)]
    if not curva.index.equals(esperado):
        raise ValueError("the curve must cover consecutive sessions of the panel")
    return metrics(curva.to_numpy(), panel.cdi[inicio:inicio + len(curva) - 1], turnover)


def curve_capm(curva: pd.Series, benchmark: pd.Series, panel: Panel) -> dict[str, float]:
    """The CAPM line of ``curva`` against ``benchmark`` on the curve's sessions, over the CDI of those sessions."""
    referencia = benchmark.reindex(curva.index)
    if referencia.isna().any():
        raise ValueError("the benchmark does not cover every session of the curve")
    inicio = int(panel.sessions.get_loc(curva.index[0]))
    return capm(curva.to_numpy(), referencia.to_numpy(), panel.cdi[inicio:inicio + len(curva) - 1])


def yearly_returns(curvas: dict[str, pd.Series]) -> pd.DataFrame:
    """Calendar-year returns of each curve (rows: years); the first year is measured from the curve's first point."""
    colunas = {}
    for nome, curva in curvas.items():
        fechos = curva.groupby(curva.index.year).last()
        colunas[nome] = fechos / fechos.shift(1).fillna(curva.iloc[0]) - 1.0
    return pd.DataFrame(colunas)


def read_run(pasta: Path) -> dict[str, list[Cell]]:
    celulas: dict[str, list[Cell]] = {}
    for arquivo in sorted(pasta.glob("*.json")):
        cell = read_cell(arquivo)
        celulas.setdefault(cell.strategy, []).append(cell)
    if not celulas:
        raise SystemExit(f"no cell under {pasta}")
    return celulas


COMPOSITE_ROW = "Composite (house engine, reference)"


def curves(celulas: dict[str, list[Cell]], composite: pd.Series | None = None, benchmark: pd.Series | None = None
           ) -> tuple[dict[str, pd.Series], dict[str, np.ndarray], set[int]]:
    """Each strategy's stitched curve over the years every strategy of the run has, every curve ending where the
    shortest reference ends, and the Composite reference cut to the same sessions; each strategy's turnover; the
    years."""
    anos = set.intersection(*({cell.year for cell in grupo} for grupo in celulas.values()))
    if not anos:
        raise ValueError("the strategies of this run share no test year")
    fins = [serie.index[-1] for serie in (composite, benchmark) if serie is not None]
    curvas, giros = {}, {}
    for estrategia, grupo in celulas.items():
        curva, giros[estrategia] = stitch([cell for cell in grupo if cell.year in anos])
        curvas[estrategia] = curva.loc[:min(fins)] if fins else curva
    if composite is not None:
        recorte = composite.reindex(next(iter(curvas.values())).index)
        if recorte.isna().any():
            raise ValueError("the Composite curve does not cover every session of the run")
        curvas[COMPOSITE_ROW] = recorte
    return curvas, giros, anos


def table(celulas: dict[str, list[Cell]], panel: Panel, composite: pd.Series | None = None,
          benchmark: pd.Series | None = None) -> pd.DataFrame:
    """One row per strategy over the years every strategy of the run has; the Composite reference on the same dates;
    with a ``benchmark`` curve, the CAPM line of every row against it."""
    curvas, giros, anos = curves(celulas, composite, benchmark)
    linhas = {}
    for nome, curva in curvas.items():
        nomes = 15.0 if nome == COMPOSITE_ROW else float(np.mean([c.names for c in celulas[nome] if c.year in anos]))
        linhas[nome] = {**curve_metrics(curva, panel, giros.get(nome)), "years": len(anos), "names": nomes}
        if benchmark is not None:
            linhas[nome] |= curve_capm(curva, benchmark, panel)
    colunas = [*FIELDS, *(CAPM_FIELDS if benchmark is not None else ()), "years", "names"]
    return pd.DataFrame(linhas).T[colunas].sort_values("sharpe", ascending=False)


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--lago", type=Path, default=LAGO)
    p.add_argument("--run", required=True)
    p.add_argument("--composite", type=Path, default=None, help="parquet with the columns date and Composite")
    p.add_argument("--benchmark", default=None, help="a column of the same parquet (BOVA11) for the CAPM line")
    p.add_argument("--last-session", default=None)
    args = p.parse_args(argv)
    panel = load_panel(args.lago, args.last_session)
    composite = benchmark = None
    if args.composite is not None:
        referencias = pd.read_parquet(args.composite)
        referencias = referencias.set_index(pd.DatetimeIndex(pd.to_datetime(referencias.pop("date"))))
        composite = referencias["Composite"].dropna()
        benchmark = None if args.benchmark is None else referencias[args.benchmark].dropna()
    celulas = read_run(SAIDAS / args.run)
    tabela = table(celulas, panel, composite, benchmark)
    anuais = yearly_returns(curves(celulas, composite, benchmark)[0])[list(tabela.index)]
    texto = (tabela.to_string(float_format=lambda x: f"{x:.4f}") + "\n\nCalendar-year returns\n"
             + anuais.to_string(float_format=lambda x: f"{x:+.3f}"))
    (SAIDAS / args.run / "leitura.txt").write_text(texto + "\n", encoding="utf-8", newline="\n")
    print(texto)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
