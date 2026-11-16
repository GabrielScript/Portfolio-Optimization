"""
painel_amplo_prever.py - Ridge-29 and PLS-29 over the broad package (docs/superpowers/specs/2026-11-16-painel-amplo-design.md,
section 4), in ``.venv-qpo``. Writes ``painel_run/painel_previsoes.npz`` (the venv cannot write on D:).

    PYTHONPATH=. ../.venv-qpo/Scripts/python.exe -X utf8 scripts/painel_amplo_prever.py
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
from sklearn.cross_decomposition import PLSRegression

from qpo_kaggle.qml_pacote import Pacote, ler_pacote
from qpo_kaggle.qml_sinal import prever_ridge, ridge_gcv

PASTA = Path(__file__).resolve().parent.parent / "painel_run"
COMPONENTES = 3


def prever_pls(pacote: Pacote) -> np.ndarray:
    previsao = np.full(len(pacote.y), np.nan)
    for mes, linhas in zip(pacote.previstos, pacote.treinos, strict=True):
        modelo = PLSRegression(n_components=COMPONENTES, scale=True).fit(pacote.theta[linhas], pacote.y[linhas])
        alvo = pacote.mes == mes
        previsao[alvo] = modelo.predict(pacote.theta[alvo]).ravel()
    return previsao


def main() -> int:
    inicio = time.perf_counter()
    pacote = ler_pacote(PASTA / "pacote_amplo.npz")
    print(f"rows {len(pacote.y):,}, features {pacote.theta.shape[1]}, months {len(pacote.previstos)}", flush=True)
    colunas = {"Ridge29": prever_ridge(pacote)}
    print(f"Ridge29 done in {time.perf_counter() - inicio:.0f} s", flush=True)
    colunas["PLS29"] = prever_pls(pacote)
    print(f"PLS29 done in {time.perf_counter() - inicio:.0f} s", flush=True)
    ultimo = ridge_gcv(pacote.theta[pacote.treinos[-1]], pacote.y[pacote.treinos[-1]])
    colunas["Ridge29_coeficientes"] = ultimo.coeficientes
    np.savez_compressed(PASTA / "painel_previsoes.npz", **colunas)
    print("written")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
