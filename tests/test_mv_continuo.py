"""The continuous arm of the hybrid QPO (docs/superpowers/specs/2026-11-01-qpo-hibrido-design.md, Prop. 3 and section 4)."""

import numpy as np
import pandas as pd
import pytest

from core.estimation import estimar_momentos
from core.qpo import qubo_alphas


def _treino(colunas: list[str], pregoes: int = 300) -> pd.DataFrame:
    gerador = np.random.default_rng(2)
    return pd.DataFrame(gerador.normal(0.0005, 0.015, (pregoes, len(colunas))), columns=colunas,
                        index=pd.bdate_range("2019-01-01", periods=pregoes))


def _nota(mes: pd.Timestamp, candidatos: list[str]) -> pd.Series:
    return pd.Series(np.arange(len(candidatos), 0, -1, dtype=float), index=candidatos)


COLUNAS = [f"AA{i:02d}3" for i in range(20)]
SETORES = {papel: f"S{i % 5}" for i, papel in enumerate(COLUNAS)}


def test_the_weights_respect_every_cap_and_are_fully_invested():
    from core.mv_continuo import estrategia_mv_continua

    pesos = estrategia_mv_continua(_nota, SETORES)(_treino(COLUNAS))
    assert pesos.sum() == pytest.approx(1.0, abs=1e-6) and pesos.min() >= -1e-9 and pesos.max() <= 0.20 + 1e-6
    assert pesos.groupby(SETORES).sum().max() <= 0.30 + 1e-6


def test_it_keeps_one_class_of_each_issuer_the_higher_score():
    from core.mv_continuo import estrategia_mv_continua

    colunas = [*COLUNAS, "AA00" + "4"]
    setores = {**SETORES, "AA004": "S0"}

    def nota(mes, candidatos):
        return _nota(mes, COLUNAS).reindex(candidatos).fillna(0.5)

    pesos = estrategia_mv_continua(nota, setores)(_treino(colunas))
    assert "AA004" not in pesos.index and "AA003" in pesos.index


def test_a_name_without_sector_or_score_is_out():
    from core.mv_continuo import estrategia_mv_continua

    setores = {papel: setor for papel, setor in SETORES.items() if papel != "AA003"}

    def nota(mes, candidatos):
        return _nota(mes, candidatos).drop("AA013")

    pesos = estrategia_mv_continua(nota, setores)(_treino(COLUNAS))
    assert "AA003" not in pesos.index and "AA013" not in pesos.index


def test_it_is_the_optimum_of_the_composite_alpha_with_lambda_one():
    from core.mv_continuo import estrategia_mv_continua

    treino = _treino(COLUNAS)
    pesos = estrategia_mv_continua(_nota, SETORES)(treino)
    cov = estimar_momentos(treino, shrink_mu=False).cov
    alpha = qubo_alphas(_nota(treino.index[-1], COLUNAS), cov)

    def utilidade(w: pd.Series) -> float:
        return float(alpha @ w - 0.5 * w @ cov @ w)

    otimo = utilidade(pesos.reindex(COLUNAS).fillna(0.0))
    gerador = np.random.default_rng(3)
    for _ in range(200):                         # feasible perturbations cannot beat it
        direcao = gerador.normal(size=len(COLUNAS))
        direcao -= direcao.mean()
        tentativa = (pesos.reindex(COLUNAS).fillna(0.0) + 0.01 * direcao).clip(0.0, 0.20)
        tentativa /= tentativa.sum()
        if tentativa.max() <= 0.20 and tentativa.groupby(SETORES).sum().max() <= 0.30:
            assert utilidade(tentativa) <= otimo + 1e-7


def test_a_portfolio_outside_the_fixed_rules_is_refused():
    from dataclasses import replace

    from core.backtest import CarteiraRecusada
    from core.mv_continuo import estrategia_mv_continua
    from core.risco import LIMITES

    with pytest.raises(CarteiraRecusada):
        estrategia_mv_continua(_nota, SETORES, limites=replace(LIMITES, peso_maximo_acao=0.01))(_treino(COLUNAS))
