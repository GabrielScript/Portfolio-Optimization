"""
qml_pacote.py - The panel as it crosses from the app's environment to ``.venv-qpo``: numpy only, anonymous, one file. The
training rows travel with it, so the purge is built and checked once, on the app's side.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class Pacote:
    theta: np.ndarray               # (rows, 6)
    y: np.ndarray                   # (rows,), NaN where the target has not closed
    mes: np.ndarray                 # (rows,) month number, year * 12 + month - 1
    previstos: np.ndarray           # the months to forecast, ascending
    treinos: Sequence[np.ndarray]   # per forecast month, the row positions that may train it


def guardar_pacote(arquivo: Path, pacote: Pacote) -> None:
    np.savez_compressed(arquivo, theta=pacote.theta, y=pacote.y, mes=pacote.mes, previstos=pacote.previstos,
                        linhas=np.concatenate(pacote.treinos), cortes=np.cumsum([len(t) for t in pacote.treinos]))


def ler_pacote(arquivo: Path) -> Pacote:
    with np.load(arquivo) as dados:
        return Pacote(dados["theta"], dados["y"], dados["mes"], dados["previstos"],
                      np.split(dados["linhas"], dados["cortes"][:-1]))
