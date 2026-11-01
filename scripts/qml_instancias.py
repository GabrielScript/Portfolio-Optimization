"""
qml_instancias.py - The solver bench's instances (docs/superpowers/specs/2026-11-01-qpo-hibrido-design.md, section 6): QUBO-giro's
real decisions in phase 0, reduced to the top-n names with k = n // 2, with the exact and annealing answers from the app's own
code, so the bench can cross-check its checker. Prints no return: the arm's curve is thrown away here.

    python -X utf8 scripts/qml_instancias.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.backtest import Congelamento, walk_forward_dinamico  # noqa: E402
from core.qpo import RISK_AVERSION, QuboInstance, build_qubo, solve_exact  # noqa: E402
from core.qubo_giro import COST_PER_SIDE, REBALANCES_PER_YEAR, TAU, estrategia_qubo_giro  # noqa: E402
from core.qubo_universo import SECTOR_CAP, anneal, anneal_solver  # noqa: E402
from qpo_kaggle.qubo_math import energy  # noqa: E402
from scripts import avaliar_conservadora, avaliar_robustez  # noqa: E402
from scripts.avaliar_robustez import _acoes  # noqa: E402
from scripts.avaliar_universo_dinamico import CFG_BT  # noqa: E402
from scripts.qml_ler_ic import PASTA  # noqa: E402
from scripts.qpo_montar import FOLDER, LAST_SESSION  # noqa: E402
from scripts.qubo_universo_avaliar import _value_score  # noqa: E402

TAMANHOS = (12, 16)
DATAS = 30
SEMENTES = 30


def reduzida(alpha: np.ndarray, cov: np.ndarray, h: np.ndarray, n: int) -> tuple[np.ndarray, np.ndarray]:
    """The top-n names with k = n // 2: (Q without turnover, Q with it), tau = f c / k as in the spec."""
    k = n // 2
    sem_giro = build_qubo(alpha[:n], cov[:n, :n], k, RISK_AVERSION)
    return sem_giro, sem_giro + REBALANCES_PER_YEAR * COST_PER_SIDE / k * np.diag(1.0 - 2.0 * h[:n])


def main() -> int:
    inicio = time.perf_counter()
    avaliar_robustez.PASTA_PRECOS = avaliar_robustez.PASTA_UNIVERSO = FOLDER
    dados = avaliar_robustez.carregar()
    mensal = avaliar_conservadora.carregar_mensal(FOLDER)
    nota, _ = _value_score(mensal, com_crescimento=True)
    guardadas: list[tuple[pd.Timestamp, np.ndarray, np.ndarray, np.ndarray]] = []
    recozimento = anneal_solver(dados.setores, SECTOR_CAP)

    def resolver(instancia: QuboInstance) -> tuple[int, ...]:
        base = build_qubo(instancia.alpha, instancia.cov, instancia.k, RISK_AVERSION)
        h = np.round((1.0 - np.diag(instancia.q - base) / TAU) / 2.0)
        topo = max(TAMANHOS)
        guardadas.append((instancia.date, instancia.alpha[:topo].copy(), instancia.cov[:topo, :topo].copy(), h[:topo]))
        return recozimento(instancia)

    acoes = _acoes(dados.precos.loc[:LAST_SESSION])
    walk_forward_dinamico(acoes, dados.universo, estrategia_qubo_giro(nota, dados.setores, resolver=resolver), CFG_BT,
                          "QUBO-giro", Congelamento())
    print(f"{len(guardadas)} real decisions, {time.perf_counter() - inicio:.0f} s", flush=True)
    posicoes = np.round(np.linspace(0, len(guardadas) - 1, DATAS)).astype(int)
    meio = len(guardadas) // 2
    casos = [("datas", int(posicao), 0) for posicao in posicoes] + [("sementes", meio, semente) for semente in range(SEMENTES)]
    saida: dict[str, np.ndarray] = {"eixo": np.array([caso[0] for caso in casos]), "semente": np.array([caso[2] for caso in casos]),
                                    "data": np.array([f"{guardadas[caso[1]][0]:%Y-%m-%d}" for caso in casos])}
    for n in TAMANHOS:
        sem, com, exatos, recozidos = [], [], [], []
        for _, posicao, semente in casos:
            _, alpha, cov, h = guardadas[posicao]
            q_sem, q_com = reduzida(alpha, cov, h, n)
            k = n // 2
            picks = anneal(q_com, k, None, seed=semente)
            sem.append(q_sem)
            com.append(q_com)
            exatos.append(solve_exact(q_com, k).energy)
            recozidos.append(float(energy(q_com, np.isin(np.arange(n), picks).astype(float))))
        saida |= {f"q_sem_{n}": np.array(sem), f"q_com_{n}": np.array(com), f"exato_{n}": np.array(exatos),
                  f"recozido_{n}": np.array(recozidos)}
        print(f"n = {n} done, {time.perf_counter() - inicio:.0f} s", flush=True)
    np.savez_compressed(PASTA / "b3" / "bancada_instancias.npz", **saida)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
