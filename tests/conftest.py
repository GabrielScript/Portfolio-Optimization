import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


@pytest.fixture
def retornos_sinteticos() -> pd.DataFrame:
    """6 ativos, 800 pregões, retornos normais correlacionados (semente fixa)."""
    gerador = np.random.default_rng(42)
    n_ativos, n_dias = 6, 800
    medias = np.array([0.0002, 0.0004, 0.0006, 0.0008, 0.0010, 0.0003])
    vols = np.array([0.010, 0.014, 0.018, 0.022, 0.030, 0.012])
    correlacao = 0.3 * np.ones((n_ativos, n_ativos)) + 0.7 * np.eye(n_ativos)
    cov = np.outer(vols, vols) * correlacao
    dados = gerador.multivariate_normal(medias, cov, size=n_dias)
    datas = pd.bdate_range("2020-01-02", periods=n_dias)
    return pd.DataFrame(dados, index=datas, columns=[f"A{i}.SA" for i in range(n_ativos)])
