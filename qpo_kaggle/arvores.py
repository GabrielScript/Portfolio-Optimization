"""
arvores.py - The tree forecasts of spec ``docs/superpowers/specs/2026-11-16-arvores-design.md`` (section 3): Random
Forest, scikit-learn Gradient Boosting and LightGBM, each refit from scratch every forecast month on that month's
purged training rows of the 213 package. Runs in ``.venv-qpo``; inputs and predictions cross as files.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Protocol

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor, HistGradientBoostingRegressor, RandomForestRegressor

from qpo_kaggle.qml_pacote import Pacote

FIOS = 3
FOLHA_MINIMA = 100

try:
    from lightgbm import LGBMRegressor
    LIGHTGBM = True
except ImportError:                                   # the spec's fallback: the same histogram boosting in scikit-learn
    LIGHTGBM = False


class Regressor(Protocol):
    def fit(self, x: np.ndarray, y: np.ndarray) -> "Regressor": ...
    def predict(self, x: np.ndarray) -> np.ndarray: ...


def random_forest(semente: int) -> RandomForestRegressor:
    return RandomForestRegressor(n_estimators=300, min_samples_leaf=FOLHA_MINIMA, max_features=2, max_samples=0.5,
                                 n_jobs=FIOS, random_state=semente)


def gradient_boosting(semente: int) -> GradientBoostingRegressor:
    return GradientBoostingRegressor(n_estimators=200, learning_rate=0.05, max_depth=3, subsample=0.8,
                                     min_samples_leaf=FOLHA_MINIMA, random_state=semente)


def lightgbm(semente: int) -> Regressor:
    if LIGHTGBM:
        return LGBMRegressor(n_estimators=200, learning_rate=0.05, num_leaves=8, min_child_samples=FOLHA_MINIMA,
                             subsample=0.8, subsample_freq=1, reg_lambda=1.0, deterministic=True,
                             force_col_wise=True, n_jobs=FIOS, random_state=semente, verbose=-1)
    return HistGradientBoostingRegressor(max_iter=200, learning_rate=0.05, max_leaf_nodes=8,
                                         min_samples_leaf=FOLHA_MINIMA, l2_regularization=1.0, early_stopping=False,
                                         random_state=semente)


MODELOS: dict[str, Callable[[int], Regressor]] = {"RF": random_forest, "GBR": gradient_boosting, "LGBM": lightgbm}


def prever_arvore(fabrica: Callable[[int], Regressor], pacote: Pacote, semente: int,
                  ao_terminar_mes: Callable[[int], None] | None = None) -> np.ndarray:
    """(rows,) forecasts: each forecast month's rows from a model fitted on that month's training rows only; NaN on
    the rows of months not forecast. ``ao_terminar_mes(posicao)`` is called after each month (progress)."""
    previsao = np.full(len(pacote.y), np.nan)
    for posicao, (mes, linhas) in enumerate(zip(pacote.previstos, pacote.treinos, strict=True)):
        modelo = fabrica(semente).fit(pacote.theta[linhas], pacote.y[linhas])
        if isinstance(modelo, RandomForestRegressor):
            modelo.set_params(n_jobs=1)               # threads sum the trees in a varying order: last-bit differences
        alvo = pacote.mes == mes
        previsao[alvo] = modelo.predict(pacote.theta[alvo])
        if ao_terminar_mes is not None:
            ao_terminar_mes(posicao)
    return previsao
