"""Minimum variance inside the house rules (spec 2026-11-16-minima-variancia)."""

import numpy as np
import pandas as pd
import pytest

from core.config import ConfigOtimizacao, Objetivo, limites_por_setor
from core.minvar import estrategia_minima_variancia, minima_variancia, percentil_da_nota

NOMES = [f"S{setor}N{i}3" for setor in range(6) for i in range(5)]
SETORES = {nome: nome[:2] for nome in NOMES}


def _cov(semente: int = 0) -> pd.DataFrame:
    """A one-factor covariance: the first sector has the lowest betas and residual variances."""
    gerador = np.random.default_rng(semente)
    betas = np.linspace(0.3, 1.5, len(NOMES))
    residuo = np.linspace(0.01, 0.09, len(NOMES)) * gerador.uniform(0.8, 1.2, len(NOMES))
    return pd.DataFrame(0.04 * np.outer(betas, betas) + np.diag(residuo), index=NOMES, columns=NOMES)


def _cfg(nomes: int | None = 15) -> ConfigOtimizacao:
    return ConfigOtimizacao(objetivo=Objetivo.MIN_VOL, peso_maximo=0.20, n_ativos_max=nomes, vol_maxima=0.40,
                            grupos=limites_por_setor(SETORES, 0.30))


def test_caps_and_count_hold():
    pesos = minima_variancia(_cov(), _cfg(nomes=8))
    assert pesos.sum() == pytest.approx(1.0)
    assert (pesos > 0).sum() <= 8 and pesos.max() <= 0.20 + 1e-9
    assert pesos.groupby(SETORES).sum().max() <= 0.30 + 1e-9
    assert pesos.groupby(SETORES).sum()["S0"] == pytest.approx(0.30, abs=1e-6)      # the low-risk sector fills its cap


def test_matches_inverse_variance_when_no_cap_binds():
    nomes = [f"S{i}N03" for i in range(8)]
    variancias = np.linspace(0.040, 0.052, 8)
    cfg = ConfigOtimizacao(objetivo=Objetivo.MIN_VOL, peso_maximo=0.20, n_ativos_max=15, vol_maxima=0.40,
                           grupos=limites_por_setor({nome: nome[:2] for nome in nomes}, 0.30))
    pesos = minima_variancia(pd.DataFrame(np.diag(variancias), index=nomes, columns=nomes), cfg)
    esperado = (1.0 / variancias) / (1.0 / variancias).sum()
    assert np.allclose(pesos.to_numpy(), esperado, atol=1e-5)


def test_score_floor_lifts_the_mean_percentile_at_a_variance_cost():
    cov = _cov()
    percentil = percentil_da_nota(pd.Series(np.arange(len(NOMES), dtype=float), index=NOMES).iloc[5:], cov.index)
    livre = minima_variancia(cov, _cfg())
    com_piso = minima_variancia(cov, _cfg(), percentil, 0.70)
    assert float(percentil @ livre) < 0.60                      # the low-risk names carry the low scores
    assert float(percentil @ com_piso) >= 0.70 - 1e-6
    assert com_piso @ cov @ com_piso > livre @ cov @ livre
    assert (com_piso > 0).sum() <= 15 and com_piso.groupby(SETORES).sum().max() <= 0.30 + 1e-9


def test_percentile_is_the_rank_among_scored_names_and_zero_without_a_score():
    percentil = percentil_da_nota(pd.Series({"A": 3.0, "B": 1.0, "C": 2.0, "D": 4.0}), pd.Index(["A", "B", "C", "D", "E"]))
    assert percentil.to_dict() == {"A": 0.75, "B": 0.25, "C": 0.5, "D": 1.0, "E": 0.0}


def test_strategy_leaves_out_names_without_a_sector_and_keeps_the_rules():
    gerador = np.random.default_rng(1)
    treino = pd.DataFrame(gerador.multivariate_normal(np.zeros(len(NOMES)), _cov().to_numpy() / 252, size=600),
                          index=pd.bdate_range("2020-01-01", periods=600), columns=NOMES)
    setores = {**SETORES, NOMES[0]: ""}                         # the lowest-risk name loses its sector
    pesos = estrategia_minima_variancia(setores, nomes=8)(treino)
    assert NOMES[0] not in pesos.index and pesos.sum() == pytest.approx(1.0)
    assert len(pesos) <= 8 and pesos.max() <= 0.20 + 1e-6 and pesos.groupby(SETORES).sum().max() <= 0.30 + 1e-6
