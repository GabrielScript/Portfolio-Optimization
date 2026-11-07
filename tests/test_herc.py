"""HERC-CDaR (core/herc.py), docs/superpowers/specs/2026-11-07-herc-design.md and its addendum. The weights before the
caps are held to skfolio's ``HierarchicalEqualRiskContribution`` (run with ``../.venv-skfolio``, DECISOES 168)."""

import numpy as np
import pandas as pd
import pytest

from core.herc import aplicar_tetos, cdar, pesos_herc_cdar


def _retornos(linhas: int = 600, nomes: int = 15, semente: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(semente)
    fatores = rng.standard_normal((linhas, 3)) * 0.01
    cargas = rng.uniform(0.0, 1.5, (3, nomes))
    proprios = rng.standard_normal((linhas, nomes)) * rng.uniform(0.005, 0.03, nomes)
    return pd.DataFrame(fatores @ cargas + proprios, columns=[f"N{i:02d}" for i in range(nomes)])


def test_cdar_is_the_mean_of_the_worst_5_percent_of_drawdowns_of_uncompounded_returns():
    retornos = np.array([0.1, -0.2, 0.05, -0.1, 0.3] * 8)
    acumulado = np.cumsum(retornos)
    quedas = np.maximum.accumulate(np.concatenate([[0.0], acumulado]))[1:] - acumulado
    esperado = np.sort(quedas)[::-1][: int(np.ceil(0.05 * len(quedas)))].mean()
    assert cdar(retornos) == pytest.approx(esperado, rel=0.05)


def test_weights_are_long_only_and_sum_to_one():
    pesos = pesos_herc_cdar(_retornos())
    assert pesos.sum() == pytest.approx(1.0) and (pesos >= 0).all()
    assert list(pesos.index) == list(_retornos().columns)


@pytest.mark.parametrize("nomes,semente", [(15, 0), (15, 3), (40, 1)])
def test_the_weights_match_skfolio_herc_cdar(nomes, semente):
    skfolio = pytest.importorskip("skfolio")
    from skfolio import RiskMeasure
    from skfolio.optimization import HierarchicalEqualRiskContribution
    retornos = _retornos(nomes=nomes, semente=semente)
    referencia = HierarchicalEqualRiskContribution(risk_measure=RiskMeasure.CDAR).fit(retornos).weights_
    assert skfolio.__version__
    np.testing.assert_allclose(pesos_herc_cdar(retornos).to_numpy(), referencia, atol=1e-6)


def test_the_caps_cut_a_name_at_20_percent_and_a_sector_at_30_percent():
    pesos = pd.Series({"A": 0.40, "B": 0.10, "C": 0.10, "D": 0.10, "E": 0.10, "F": 0.10, "G": 0.10})
    setores = {"A": "S1", "B": "S1", "C": "S2", "D": "S2", "E": "S3", "F": "S4", "G": "S5"}
    cortados = aplicar_tetos(pesos, setores, nome=0.20, setor=0.30)
    assert cortados.sum() == pytest.approx(1.0)
    assert cortados.max() <= 0.20 + 1e-9
    assert cortados.groupby(setores).sum().max() <= 0.30 + 1e-9
    assert cortados["E"] > 0.10                                       # the excess went to names with room
