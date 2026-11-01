"""
qml_spike.py - Timing of one training step at the real batch size, on synthetic angles, with no market data: the number
that decides laptop or Kaggle (design, section 7).

    PYTHONPATH=. ../.venv-qpo/Scripts/python.exe -X utf8 qpo_kaggle/qml_spike.py
"""

from __future__ import annotations

import time

import numpy as np
import torch
from torch import nn

from qpo_kaggle.qml_sinal import N_ATRIBUTOS, Mlp, Treino, Vqc, atributos_de_fourier, ridge_gcv

PASSOS_DA_MEDIDA = 30
MESES = {"B3": 203, "EUA": 83}
SEMENTES = 3


def segundos_por_passo(modelo: nn.Module, lote: int) -> float:
    gerador = np.random.default_rng(0)
    theta = torch.tensor(gerador.uniform(-np.pi / 2, np.pi / 2, (lote, N_ATRIBUTOS)), dtype=torch.float64)
    y = torch.tensor(gerador.normal(size=lote), dtype=torch.float64)
    otimizador = torch.optim.Adam(modelo.parameters(), lr=0.02)
    inicio = time.perf_counter()
    for _ in range(PASSOS_DA_MEDIDA):
        otimizador.zero_grad(set_to_none=True)
        nn.functional.mse_loss(modelo(theta), y).backward()
        otimizador.step()
    return (time.perf_counter() - inicio) / PASSOS_DA_MEDIDA


def main() -> None:
    treino = Treino()
    print(f"torch threads {torch.get_num_threads()}")
    for nome, classe in (("VQC", Vqc), ("MLP", Mlp)):
        por_passo = segundos_por_passo(classe(semente=0), treino.lote)
        for mercado, meses in MESES.items():
            passos = (treino.passos_iniciais + (meses - 1) * treino.passos_seguintes) * SEMENTES
            print(f"{nome} {mercado}: {por_passo * 1000:.0f} ms a step, {passos:,} steps, {por_passo * passos / 3600:.2f} h")
    gerador = np.random.default_rng(0)
    x = atributos_de_fourier(gerador.uniform(-np.pi / 2, np.pi / 2, (20_000, N_ATRIBUTOS)))
    inicio = time.perf_counter()
    ridge_gcv(x, gerador.normal(size=20_000))
    fit = time.perf_counter() - inicio
    print(f"Fourier ridge, 20,000 rows: {fit:.1f} s a month, B3 {fit * MESES['B3'] / 60:.0f} min, {x.nbytes / 1e6:.0f} MB")


if __name__ == "__main__":
    main()
