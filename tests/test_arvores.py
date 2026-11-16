"""The tree forecasts of spec 2026-11-16-arvores (run in ``.venv-qpo``)."""

import numpy as np
import pytest

from qpo_kaggle.arvores import MODELOS, prever_arvore
from qpo_kaggle.qml_pacote import Pacote


def _pacote(semente: int = 0) -> Pacote:
    gerador = np.random.default_rng(semente)
    meses = np.repeat(np.arange(6), 300)
    theta = gerador.uniform(-1.5, 1.5, size=(len(meses), 6))
    y = np.sin(2 * theta[:, 0]) + 0.5 * theta[:, 1] * theta[:, 2] + 0.3 * gerador.standard_normal(len(meses))
    previstos = np.array([3, 4, 5])
    treinos = [np.flatnonzero(meses < m - 1) for m in previstos]     # a purge of one month, as the package's
    return Pacote(theta, y, meses, previstos, treinos)


@pytest.mark.parametrize("nome", ["RF", "GBR", "LGBM"])
def test_forecasts_only_the_forecast_months_and_learns_the_signal(nome):
    pacote = _pacote()
    previsao = prever_arvore(MODELOS[nome], pacote, semente=0)
    alvo = np.isin(pacote.mes, pacote.previstos)
    assert np.isnan(previsao[~alvo]).all() and np.isfinite(previsao[alvo]).all()
    assert np.corrcoef(previsao[alvo], pacote.y[alvo])[0, 1] > 0.5


@pytest.mark.parametrize("nome", ["RF", "GBR", "LGBM"])
def test_the_forecast_month_and_the_purged_month_never_reach_the_fit(nome):
    pacote = _pacote()
    mexido = pacote.y.copy()
    mexido[pacote.mes >= 4] = 99.0                     # month 4 is forecast 4's own and month 5's purge
    alterado = Pacote(pacote.theta, mexido, pacote.mes, pacote.previstos, pacote.treinos)
    antes = prever_arvore(MODELOS[nome], pacote, semente=0)
    depois = prever_arvore(MODELOS[nome], alterado, semente=0)
    meses_4_5 = np.isin(pacote.mes, [4, 5])
    np.testing.assert_array_equal(antes[meses_4_5], depois[meses_4_5])


@pytest.mark.parametrize("nome", ["RF", "GBR", "LGBM"])
def test_same_seed_same_forecast(nome):
    pacote = _pacote()
    np.testing.assert_array_equal(prever_arvore(MODELOS[nome], pacote, semente=1),
                                  prever_arvore(MODELOS[nome], pacote, semente=1))


def test_settings_are_the_pre_registered_ones():
    rf, gbr = MODELOS["RF"](0).get_params(), MODELOS["GBR"](0).get_params()
    assert (rf["n_estimators"], rf["min_samples_leaf"], rf["max_features"], rf["max_samples"]) == (300, 100, 2, 0.5)
    assert (gbr["n_estimators"], gbr["learning_rate"], gbr["max_depth"], gbr["subsample"], gbr["min_samples_leaf"]) \
        == (200, 0.05, 3, 0.8, 100)
    lgbm = MODELOS["LGBM"](0).get_params()
    if "num_leaves" in lgbm:
        assert (lgbm["n_estimators"], lgbm["learning_rate"], lgbm["num_leaves"], lgbm["min_child_samples"],
                lgbm["subsample"], lgbm["subsample_freq"], lgbm["reg_lambda"]) == (200, 0.05, 8, 100, 0.8, 1, 1.0)
    else:                                              # the pre-registered fallback, HistGradientBoostingRegressor
        assert (lgbm["max_iter"], lgbm["learning_rate"], lgbm["max_leaf_nodes"], lgbm["min_samples_leaf"],
                lgbm["l2_regularization"], lgbm["early_stopping"]) == (200, 0.05, 8, 100, 1.0, False)
