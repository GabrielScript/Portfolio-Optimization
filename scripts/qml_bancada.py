"""
qml_bancada.py - Runs the solver bench (docs/superpowers/specs/2026-11-01-qpo-hibrido-design.md, section 6) in ``.venv-qpo`` and
appends one row per (case, n, solver, raw or repaired), so a stopped run resumes. Fixed before any result: Adam, 200 steps, rate
0.05, p = 3, 3 layers, 4,096 shots.

    PYTHONPATH=. ../.venv-qpo/Scripts/python.exe -X utf8 scripts/qml_bancada.py D:/b3-datalake/exports/qml/b3
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from qpo_kaggle import qml_bancada as b

TIROS = 4_096
PASSOS = 200
TAXA = 0.05
P = 3
CAMADAS = 3
FIOS = 2
TOLERANCIA_DA_CONFERENCIA = 1e-9


def linhas_do_caso(q_sem: np.ndarray, q_com: np.ndarray, k: int, semente: int, exato_do_app: float,
                   recozido_do_app: float) -> list[dict]:
    gabarito = b.exato(q_com, k)
    if abs(gabarito.minimo - exato_do_app) > TOLERANCIA_DA_CONFERENCIA:
        raise SystemExit(f"the checker's optimum {gabarito.minimo} differs from the app's {exato_do_app}")
    n = len(q_com)
    estados = b.todos_os_estados(n)
    linhas = []

    def registrar(solver: str, etapa: str, amostras: np.ndarray, segundos: float) -> None:
        linhas.append({"solver": solver, "etapa": etapa, "segundos": segundos, **b.medir(amostras, q_com, k, gabarito)})

    def com_reparo(solver: str, amostras: np.ndarray, segundos: float) -> None:
        registrar(solver, "cru", amostras, segundos)
        registrar(solver, "reparado", b.reparar(q_com, amostras, k), segundos)

    registrar("Uniforme viável", "cru", b.uniforme_viavel(n, k, TIROS, semente), 0.0)
    com_reparo("Uniforme na base", b.uniforme_na_base(n, TIROS, semente), 0.0)
    linhas.append({"solver": "Recozimento", "etapa": "cru", "segundos": float("nan"), "viavel": 1.0,
                   "p_otimo": float(recozido_do_app <= gabarito.minimo + 1e-10), "razao_media": float("nan"),
                   "razao_melhor": b.approximation_ratio(recozido_do_app, gabarito.minimo, gabarito.maximo)})
    inicio = time.perf_counter()
    distribuicao, subespaco = b.treinar_qaoa_hard(q_com, k, P, PASSOS, TAXA)
    registrar("QAOA-hard", "cru", b.amostrar(distribuicao, subespaco, TIROS, semente), time.perf_counter() - inicio)

    penalizada = b.energias_com_penalidade(q_com, k, b.penalidade_de_hodson(q_com, k, semente))
    inicio = time.perf_counter()
    distribuicao = b.treinar_qaoa_soft(penalizada, P, PASSOS, TAXA)
    com_reparo("QAOA-soft", b.amostrar(distribuicao, estados, TIROS, semente), time.perf_counter() - inicio)

    inicio = time.perf_counter()
    distribuicao = b.treinar_vqe(penalizada, CAMADAS, PASSOS, TAXA, semente)
    com_reparo("VQE", b.amostrar(distribuicao, estados, TIROS, semente), time.perf_counter() - inicio)

    inicio = time.perf_counter()
    sem_giro = b.energias_com_penalidade(q_sem, k, b.penalidade_de_hodson(q_sem, k, semente))
    amostras = b.amostrar(b.treinar_vqe(sem_giro, CAMADAS, PASSOS, TAXA, semente), estados, TIROS, semente)
    if (amostras.sum(axis=1) == k).any():
        registrar("VQE-Constrained", "pós", b.vqe_restrito(amostras, q_sem, q_com, k)[None], time.perf_counter() - inicio)
    else:
        linhas.append({"solver": "VQE-Constrained", "etapa": "pós", "segundos": time.perf_counter() - inicio, "viavel": 0.0,
                       "p_otimo": 0.0, "razao_media": float("nan"), "razao_melhor": float("nan")})
    return linhas


def main(argv: list[str]) -> int:
    pasta = Path(argv[0])
    torch.set_num_threads(FIOS)
    saida = pasta / "bancada.csv"
    feitos = set()
    if saida.exists():
        feitos = set(map(tuple, pd.read_csv(saida)[["n", "caso"]].drop_duplicates().to_numpy().tolist()))
    with np.load(pasta / "bancada_instancias.npz") as dados:
        instancias = {nome: dados[nome] for nome in dados.files}
    for n in (12, 16):
        for caso in range(len(instancias["eixo"])):
            if (n, caso) in feitos:
                continue
            inicio = time.perf_counter()
            linhas = linhas_do_caso(instancias[f"q_sem_{n}"][caso], instancias[f"q_com_{n}"][caso], n // 2,
                                    int(instancias["semente"][caso]), float(instancias[f"exato_{n}"][caso]),
                                    float(instancias[f"recozido_{n}"][caso]))
            pd.DataFrame(linhas).assign(n=n, caso=caso, eixo=instancias["eixo"][caso], data=instancias["data"][caso]).to_csv(
                saida, mode="a", header=not saida.exists(), index=False)
            print(f"n {n} case {caso}: {time.perf_counter() - inicio:.0f} s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
