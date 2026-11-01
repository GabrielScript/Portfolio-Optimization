"""
pbo.py - The probability of backtest overfitting by combinatorially symmetric cross-validation (Bailey, Borwein, Lopez de Prado and Zhu,
2017). Splits the rows into S blocks; in each of the C(S, S/2) splits, the configuration with the best Sharpe in one half is ranked in
the other. PBO is how often it lands in the bottom half. A description of a set of curves, not a gate: it sees only the curves given.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np
import pandas as pd
from scipy.stats import rankdata


@dataclass(frozen=True)
class ResultadoCSCV:
    pbo: float
    divisoes: int
    mais_escolhida: str
    logits: np.ndarray


def cscv(retornos: pd.DataFrame, blocos: int = 16) -> ResultadoCSCV:
    """``retornos``: one row per period, one column per configuration, no gaps. Sums per block, so a split costs O(S x N)."""
    if blocos % 2:
        raise ValueError(f"blocos must be even: {blocos}")
    if retornos.isna().any(axis=None):
        raise ValueError("The rows have a gap: align the curves first")
    partes = np.array_split(retornos.to_numpy(), blocos)
    linhas = np.array([len(parte) for parte in partes], dtype=float)
    somas = np.array([parte.sum(axis=0) for parte in partes])
    quadrados = np.array([(parte ** 2).sum(axis=0) for parte in partes])

    def sharpe(indices: list[int]) -> np.ndarray:
        n = linhas[indices].sum()
        media = somas[indices].sum(axis=0) / n
        return media / np.sqrt((quadrados[indices].sum(axis=0) - n * media ** 2) / (n - 1.0))

    colunas = retornos.shape[1]
    logits, escolhas = [], np.zeros(colunas, dtype=int)
    for dentro in combinations(range(blocos), blocos // 2):
        fora = [i for i in range(blocos) if i not in dentro]
        melhor = int(np.argmax(sharpe(list(dentro))))
        posicao = rankdata(sharpe(fora))[melhor] / (colunas + 1)
        logits.append(np.log(posicao / (1.0 - posicao)))
        escolhas[melhor] += 1
    valores = np.array(logits)
    return ResultadoCSCV(float((valores <= 0.0).mean()), len(valores), str(retornos.columns[int(escolhas.argmax())]), valores)
