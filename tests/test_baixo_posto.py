"""The low-rank form of the engine's Ledoit-Wolf covariance and the optimizer on it: same answers as the dense ones, for universes
where the dense form costs minutes per solve (the US test: about 3,000 names on 36 rows)."""

import numpy as np
import pandas as pd
import pytest

from core.estimation import estimar_momentos


def _retornos(linhas: int, colunas: int, semente: int = 0) -> pd.DataFrame:
    gerador = np.random.default_rng(semente)
    mercado = gerador.normal(0.0005, 0.008, linhas)[:, None]
    dados = mercado * gerador.uniform(0.5, 1.5, colunas) + gerador.normal(0.0, 0.015, (linhas, colunas))
    return pd.DataFrame(dados, columns=[f"S{i}" for i in range(colunas)],
                        index=pd.bdate_range("2019-01-01", periods=linhas))


@pytest.mark.parametrize(("linhas", "colunas"), [(36, 200), (36, 36), (300, 20)])
def test_it_rebuilds_the_engines_annual_covariance(linhas, colunas):
    from core.baixo_posto import ledoit_wolf_baixo_posto

    retornos = _retornos(linhas, colunas)
    baixo = ledoit_wolf_baixo_posto(retornos)
    densa = estimar_momentos(retornos, shrink_mu=False).cov
    assert baixo.nomes == list(retornos.columns)
    assert baixo.diagonal * np.eye(colunas) + baixo.fator.T @ baixo.fator == pytest.approx(densa.to_numpy(), abs=1e-12)


def _caso(cfg_extra: dict, setorizado: bool = False, colunas: int = 40):
    from core.config import ConfigOtimizacao, Objetivo, limites_por_setor

    retornos = _retornos(36, colunas)
    mu = pd.Series(np.random.default_rng(1).normal(0.10, 0.08, colunas), index=retornos.columns)
    grupos = limites_por_setor({papel: f"G{i % 5}" for i, papel in enumerate(retornos.columns)}, 0.30) if setorizado else ()
    base = {"objetivo": Objetivo.KELLY, "peso_maximo": 0.20, "n_ativos_max": None, "vol_maxima": 0.40, "aversao_risco": 1.0}
    return retornos, mu, ConfigOtimizacao(**{**base, **cfg_extra}, grupos=grupos)


@pytest.mark.parametrize(("extra", "setorizado"), [({}, False), ({}, True), ({"n_ativos_max": 10}, False),
                                                   ({"aversao_risco": 4.0}, True), ({"vol_maxima": 0.12}, False)])
def test_kelly_gives_the_dense_optimizers_weights(extra, setorizado):
    from core.baixo_posto import ledoit_wolf_baixo_posto, otimizar_baixo_posto
    from core.optimization import otimizar

    retornos, mu, cfg = _caso(extra, setorizado)
    densa = otimizar(mu, estimar_momentos(retornos, shrink_mu=False).cov, cfg).pesos
    pesos = otimizar_baixo_posto(mu, ledoit_wolf_baixo_posto(retornos), cfg)
    assert pesos.index.equals(densa.index)
    assert pesos.to_numpy() == pytest.approx(densa.to_numpy(), abs=2e-4)


def test_an_impossible_volatility_cap_falls_back_to_the_minimum_variance():
    from core.baixo_posto import ledoit_wolf_baixo_posto, otimizar_baixo_posto
    from core.optimization import otimizar

    retornos, mu, cfg = _caso({"vol_maxima": 0.001})
    densa = otimizar(mu, estimar_momentos(retornos, shrink_mu=False).cov, cfg)
    assert densa.status == "fallback"
    assert otimizar_baixo_posto(mu, ledoit_wolf_baixo_posto(retornos), cfg).to_numpy() == pytest.approx(densa.pesos.to_numpy(), abs=2e-4)


def test_it_refuses_what_it_cannot_solve_instead_of_answering_wrongly():
    from core.baixo_posto import ledoit_wolf_baixo_posto, otimizar_baixo_posto
    from core.config import Objetivo
    from dataclasses import replace

    retornos, mu, cfg = _caso({})
    sigma = ledoit_wolf_baixo_posto(retornos)
    with pytest.raises(ValueError, match="only Kelly"):
        otimizar_baixo_posto(mu, sigma, replace(cfg, objetivo=Objetivo.MAX_SHARPE))
    with pytest.raises(ValueError, match="same names"):
        otimizar_baixo_posto(mu.iloc[::-1], sigma, cfg)
    with pytest.raises(ValueError, match="NaN"):
        ledoit_wolf_baixo_posto(retornos.mask(retornos == retornos.iloc[0, 0]))


def test_the_large_universe_arm_gives_the_mv_continuo_weights():
    from core.mv_continuo import estrategia_mv_continua, estrategia_mv_continua_grande

    retornos = _retornos(36, 40)
    setores = {papel: f"G{i % 5}" for i, papel in enumerate(retornos.columns)}

    def nota(mes, candidatos):
        return pd.Series(np.arange(len(candidatos), 0, -1, dtype=float), index=candidatos)

    grande = estrategia_mv_continua_grande(nota, setores)(retornos)
    assert grande.to_numpy() == pytest.approx(estrategia_mv_continua(nota, setores)(retornos).to_numpy(), abs=2e-4)
