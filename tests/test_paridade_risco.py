"""core.paridade_risco: the equal risk contribution (ERC) portfolio, long only, fully invested."""

import numpy as np
import pandas as pd
import pytest

from core.paridade_risco import contribuicoes_de_risco, otimizar_paridade_de_risco


def test_uncorrelated_assets_get_weights_inverse_to_their_volatility():
    vols = np.array([0.1, 0.2, 0.4])
    cov = pd.DataFrame(np.diag(vols ** 2), index=list("ABC"), columns=list("ABC"))
    pesos = otimizar_paridade_de_risco(cov)
    esperado = (1 / vols) / (1 / vols).sum()
    assert pesos.to_numpy() == pytest.approx(esperado, rel=1e-6)
    assert list(pesos.index) == list("ABC")


def test_every_asset_contributes_the_same_risk_on_a_correlated_covariance():
    gerador = np.random.default_rng(3)
    fatores = gerador.normal(size=(500, 8)) @ gerador.normal(size=(8, 8)) * 0.01
    cov = pd.DataFrame(np.cov(fatores, rowvar=False), index=[f"P{i}" for i in range(8)],
                       columns=[f"P{i}" for i in range(8)])
    pesos = otimizar_paridade_de_risco(cov)
    assert pesos.sum() == pytest.approx(1.0) and (pesos > 0).all()
    partes = contribuicoes_de_risco(pesos, cov)
    assert partes.to_numpy() == pytest.approx(np.full(8, 1 / 8), abs=1e-6)


def test_a_single_asset_takes_everything():
    cov = pd.DataFrame([[0.04]], index=["A"], columns=["A"])
    assert otimizar_paridade_de_risco(cov).tolist() == [1.0]
