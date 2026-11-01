"""
qml_prever.py - Runs the forecast models over one market's package, in ``.venv-qpo``. Resumes where it stopped.

    PYTHONPATH=. ../.venv-qpo/Scripts/python.exe -X utf8 scripts/qml_prever.py D:/b3-datalake/exports/qml/b3
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import torch

from qpo_kaggle.qml_pacote import ler_pacote
from qpo_kaggle.qml_sinal import Mlp, Treino, Vqc, prever_fourier, prever_ridge, prever_torch

SEMENTES = (0, 1, 2)
FIOS = 3                # threads for torch: one core stays free for the batch and the machine


def main(argv: list[str]) -> int:
    pasta = Path(argv[0])
    torch.set_num_threads(FIOS)
    pacote = ler_pacote(pasta / "pacote.npz")
    print(f"versions: torch {torch.__version__}; months {len(pacote.previstos)}; rows {len(pacote.y):,}", flush=True)
    resultados = {}
    saida = pasta / "previsoes.npz"
    if saida.exists():
        with np.load(saida) as antigos:
            resultados = dict(antigos)
    for nome, funcao in (("Ridge", prever_ridge), ("Fourier", prever_fourier)):
        if nome not in resultados:
            inicio = time.perf_counter()
            resultados[nome] = funcao(pacote)
            np.savez_compressed(saida, **resultados)
            print(f"{nome}: {time.perf_counter() - inicio:.0f} s", flush=True)
    for nome, classe in (("MLP", Mlp), ("VQC", Vqc)):
        if nome in resultados:
            continue
        inicio = time.perf_counter()
        por_semente = [prever_torch(classe, pacote, semente, Treino(), pasta / f"estado_{nome}") for semente in SEMENTES]
        resultados[nome] = np.mean(por_semente, axis=0)
        for semente, previsao in zip(SEMENTES, por_semente, strict=True):
            resultados[f"{nome}_{semente}"] = previsao
        np.savez_compressed(saida, **resultados)
        print(f"{nome}: {time.perf_counter() - inicio:.0f} s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
