"""
volatilidade.py - Annualized volatility and downside volatility of a value curve: the measures of the Conservadora vs
BOVA11 rerun (`scripts/rerun_conservadora.py`) and of paper-v2's secondary criterion (`scripts/paper_v2.py`).

Kept out of `core/metrics.py`, which is in paper-v1's guarded path (`scripts/paper_trading.py`).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from core.config import DIAS_UTEIS_ANO
from core.metrics import downside_deviation, retornos_de_valores

_RAIZ_ANO = np.sqrt(DIAS_UTEIS_ANO)


@dataclass(frozen=True)
class Volatilidades:
    anual: float        # standard deviation of the daily simple returns x sqrt(252)
    de_baixa: float     # sqrt(mean(min(r, 0)^2)) over all days x sqrt(252): target 0, not the CDI


def volatilidades(valores: pd.Series) -> Volatilidades:
    """Both measures on the daily simple returns of ``valores`` (missing values dropped first)."""
    retornos = retornos_de_valores(valores.dropna())
    return Volatilidades(float(retornos.std() * _RAIZ_ANO), downside_deviation(retornos) * float(_RAIZ_ANO))
