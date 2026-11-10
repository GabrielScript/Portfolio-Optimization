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
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from drl.dados import Panel, load_panel
from drl.executar import LAGO, SAIDAS, Cell, read_cell
from drl.periodo import emendar
from drl.relatorio import FIELDS, metrics


def stitch(cells: Sequence[Cell]) -> tuple[pd.Series, np.ndarray]:
    """One curve (indexed by session) and the turnover of every decision, from the cells of one strategy. Each cell
    must start on the session the previous one ended on."""
    ordenadas = sorted(cells, key=lambda cell: cell.year)
    for anterior, seguinte in zip(ordenadas, ordenadas[1:]):
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


def read_run(pasta: Path) -> dict[str, list[Cell]]:
    celulas: dict[str, list[Cell]] = {}
    for arquivo in sorted(pasta.glob("*.json")):
        cell = read_cell(arquivo)
        celulas.setdefault(cell.strategy, []).append(cell)
    if not celulas:
        raise SystemExit(f"no cell under {pasta}")
    return celulas


def table(celulas: dict[str, list[Cell]], panel: Panel, composite: pd.Series | None = None) -> pd.DataFrame:
    """One row per strategy over the years every strategy of the run has; the Composite reference on the same dates."""
    anos = set.intersection(*(set(cell.year for cell in grupo) for grupo in celulas.values()))
    if not anos:
        raise ValueError("the strategies of this run share no test year")
    linhas, janela = {}, None
    for estrategia, grupo in celulas.items():
        curva, giro = stitch([cell for cell in grupo if cell.year in anos])
        janela = curva.index
        linhas[estrategia] = {**curve_metrics(curva, panel, giro), "years": len(anos),
                              "names": float(np.mean([cell.names for cell in grupo if cell.year in anos]))}
    if composite is not None and janela is not None:
        recorte = composite.reindex(janela)
        if recorte.isna().any():
            raise ValueError("the Composite curve does not cover every session of the run")
        linhas["Composite (house engine, reference)"] = {**curve_metrics(recorte, panel), "years": len(anos),
                                                         "names": 15.0}
    return pd.DataFrame(linhas).T[[*FIELDS, "years", "names"]].sort_values("sharpe", ascending=False)


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--lago", type=Path, default=LAGO)
    p.add_argument("--run", required=True)
    p.add_argument("--composite", type=Path, default=None)
    p.add_argument("--last-session", default=None)
    args = p.parse_args(argv)
    panel = load_panel(args.lago, args.last_session)
    composite = None
    if args.composite is not None:
        composite = pd.read_parquet(args.composite).set_index("date")["Composite"].dropna()
        composite.index = pd.DatetimeIndex(pd.to_datetime(composite.index))
    tabela = table(read_run(SAIDAS / args.run), panel, composite)
    texto = tabela.to_string(float_format=lambda x: f"{x:.4f}")
    (SAIDAS / args.run / "leitura.txt").write_text(texto + "\n", encoding="utf-8", newline="\n")
    print(texto)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
