"""Annualized volatility and downside volatility of a value curve (``core/volatilidade.py``): the measures of the
Conservadora vs BOVA11 rerun and of paper-v2's secondary criterion."""

import numpy as np
import pandas as pd
import pytest


def test_both_measures_annualize_the_daily_simple_returns_of_the_curve():
    from core.volatilidade import volatilidades
    retornos = np.array([0.01, -0.02, 0.01, -0.01])
    valores = pd.Series(100.0 * np.cumprod(np.r_[1.0, 1.0 + retornos]), index=pd.bdate_range("2026-01-05", periods=5))
    medidas = volatilidades(valores)
    assert medidas.anual == pytest.approx(np.std(retornos, ddof=1) * np.sqrt(252))
    assert medidas.de_baixa == pytest.approx(np.sqrt(np.mean(np.minimum(retornos, 0.0) ** 2)) * np.sqrt(252))
