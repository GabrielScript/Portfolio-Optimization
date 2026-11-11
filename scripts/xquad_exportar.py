"""Export every QUBO of attempt 56 (full universe, Composite score) for the XQuad solve (spec 2026-11-11-xquad-quip).

    ../Portfolio-Optimization/.venv/Scripts/python -X utf8 scripts/xquad_exportar.py

Drives the engine with attempt 56's selector and the project's annealer, weights set equal (the QUBO of a decision
depends only on the training window and the month's score, never on the weights or on earlier picks), and writes
``../quantum-lab/instancias/fase{0,21,42}.npz``: for decision i, ``q{i}`` (the QUBO), ``t{i}`` (tickers), ``g{i}``
(sectors), ``p{i}`` (attempt 56's picks, indices) and ``dates`` (ISO). Decisions without a QUBO (no more than k
candidates) are absent, as in the engine.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.backtest import Congelamento, walk_forward_dinamico  # noqa: E402
from core.qpo import K, qubo_selector  # noqa: E402
from core.qubo_universo import FULL_UNIVERSE, SECTOR_CAP, anneal_solver  # noqa: E402
from scripts import avaliar_conservadora, avaliar_robustez  # noqa: E402
from scripts.avaliar_robustez import FASES, _acoes  # noqa: E402
from scripts.avaliar_universo_dinamico import CFG_BT  # noqa: E402
from scripts.qpo_montar import FOLDER, LAST_SESSION  # noqa: E402
from scripts.qubo_universo_avaliar import _value_score  # noqa: E402

DESTINO = Path(__file__).resolve().parents[2] / "quantum-lab" / "instancias"


def main() -> int:
    inicio = time.perf_counter()
    avaliar_robustez.PASTA_PRECOS = avaliar_robustez.PASTA_UNIVERSO = FOLDER
    dados = avaliar_robustez.carregar()
    mensal = avaliar_conservadora.carregar_mensal(FOLDER)
    nota, _ = _value_score(mensal, com_crescimento=True)
    DESTINO.mkdir(parents=True, exist_ok=True)
    for fase in FASES:
        registro: dict = {}
        selecionar = qubo_selector(dados.setores, FULL_UNIVERSE, K, anneal_solver(dados.setores, SECTOR_CAP), registro)

        def estrategia(treino: pd.DataFrame) -> pd.Series:
            mes = treino.index[-1].to_period("M").to_timestamp()
            nomes = selecionar(treino.index[-1], treino, nota(mes, list(treino.columns)))
            return pd.Series(1.0 / len(nomes), index=nomes)

        acoes = _acoes(dados.precos.loc[:LAST_SESSION].iloc[fase:])
        walk_forward_dinamico(acoes, dados.universo, estrategia, CFG_BT, "export", Congelamento())
        datas = sorted(registro)
        arrays = {"dates": np.array([f"{d:%Y-%m-%d}" for d in datas])}
        for i, data in enumerate(datas):
            instancia, picks = registro[data]
            arrays[f"q{i}"] = instancia.q
            arrays[f"t{i}"] = np.array(instancia.tickers)
            arrays[f"g{i}"] = np.array([dados.setores[t] for t in instancia.tickers])
            arrays[f"p{i}"] = np.array(picks, dtype=int)
        np.savez_compressed(DESTINO / f"fase{fase}.npz", **arrays)
        tamanhos = [len(registro[d][0].tickers) for d in datas]
        print(f"phase {fase}: {len(datas)} QUBOs, candidates {min(tamanhos)}-{max(tamanhos)} "
              f"(median {int(np.median(tamanhos))}), {time.perf_counter() - inicio:.0f} s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
