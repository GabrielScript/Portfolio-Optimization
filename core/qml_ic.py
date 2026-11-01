"""
qml_ic.py - The reading of the hybrid forecast (docs/superpowers/specs/2026-11-01-qpo-hibrido-design.md, section 3): monthly
Spearman IC against the 3-month target, its Newey-West t and the gate of S2 and DECISOES 161.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

IC_MINIMO = 0.02
T_MINIMO = 2.0
LAGS = 2                # the 3-month target overlaps across consecutive months


def t_newey_west(serie: pd.Series, lags: int = LAGS) -> float:
    """t of the mean with Bartlett weights over ``lags`` autocovariances."""
    valores = serie.dropna().to_numpy()
    desvios = valores - valores.mean()
    variancia = float(desvios @ desvios) / len(valores)
    for lag in range(1, lags + 1):
        variancia += 2.0 * (1.0 - lag / (lags + 1)) * float(desvios[lag:] @ desvios[:-lag]) / len(valores)
    return float(valores.mean() / np.sqrt(variancia / len(valores)))


def passa_no_ic(ic: float, t: float) -> bool:
    return ic >= IC_MINIMO and t >= T_MINIMO


def ic_do_alvo(tabela: pd.DataFrame, previsao: str, alvo: str) -> pd.Series:
    """Spearman IC of each month with a closed target, over the rows that have both. Columns: ``month``, the two given."""
    validas = tabela.dropna(subset=[previsao, alvo])
    return validas.groupby("month").apply(lambda parte: parte[previsao].rank().corr(parte[alvo].rank()),
                                          include_groups=False)
