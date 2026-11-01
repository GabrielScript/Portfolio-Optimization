"""The recorded strategies on the US's monthly grid (docs/superpowers/specs/2026-11-01-todas-as-estrategias-design.md, sections 2 and 4):
the same optimum as the dense code they stand in for, on monthly moments the dense annualization would get wrong."""

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from core.backtest import CarteiraRecusada, estrategia_hrp
from core.config import ConfigOtimizacao, Objetivo, limites_por_setor
from core.estimation import covariancia_ledoit_wolf, shrinkage_jorion
from core.optimization import otimizar
from core.reamostragem import ConfigReamostragem, otimizar_reamostrado
from core.risco import Limites
from core.us_teste import no_grid_mensal
from core.views import Sinal, estrategia_com_views

KELLY = ConfigOtimizacao(objetivo=Objetivo.KELLY, peso_maximo=0.20, n_ativos_max=10, vol_maxima=0.40, aversao_risco=2.0)


def _mensal(linhas: int = 36, colunas: int = 24, semente: int = 0) -> pd.DataFrame:
    gerador = np.random.default_rng(semente)
    dados = gerador.normal(0.008, 0.03, linhas)[:, None] * gerador.uniform(0.5, 1.5, colunas) + gerador.normal(0.004, 0.06, (linhas, colunas))
    return pd.DataFrame(dados, columns=[f"N{i:03d}" for i in range(colunas)],
                        index=pd.date_range("2020-01-31", periods=linhas, freq="ME"))


def _setores(retornos: pd.DataFrame) -> dict[str, str]:
    return {papel: f"S{i % 6}" for i, papel in enumerate(retornos.columns)}


def _momentos(retornos: pd.DataFrame, jorion: bool) -> tuple[pd.Series, pd.DataFrame]:
    cov = covariancia_ledoit_wolf(retornos)
    media = retornos.mean().to_numpy()
    if jorion:
        media = shrinkage_jorion(media, cov.to_numpy(), len(retornos))[0]
    return pd.Series(media * 12, index=retornos.columns), cov * 12


@pytest.mark.parametrize("objetivo", [Objetivo.KELLY, Objetivo.MAX_SHARPE, Objetivo.MIN_VOL])
def test_the_large_arm_is_the_dense_optimum_on_monthly_moments(objetivo):
    from core.estrategias_todas_us import estrategia_grande

    treino = _mensal()
    mu, cov = _momentos(treino, jorion=True)
    cfg = replace(KELLY, objetivo=objetivo)
    esperado = otimizar(mu, cov, cfg).pesos
    assert estrategia_grande(cfg)(treino).to_numpy() == pytest.approx(esperado.to_numpy(), abs=2e-3)


def test_with_many_more_names_than_rows_the_mean_is_raw_as_the_spec_says():
    from core.estrategias_todas_us import estrategia_grande

    treino = _mensal(36, 60)
    mu, cov = _momentos(treino, jorion=False)
    esperado = otimizar(mu, cov, KELLY).pesos
    assert estrategia_grande(KELLY)(treino).to_numpy() == pytest.approx(esperado.to_numpy(), abs=2e-3)


def test_max_sharpe_takes_the_t_bill_of_the_last_complete_month():
    from core.estrategias_todas_us import estrategia_grande

    treino = _mensal()
    data = treino.index[-1]
    rf = pd.Series({data - pd.offsets.MonthEnd(1): 0.002, data: 0.05})     # the decision month's own rate is not known yet
    mu, cov = _momentos(treino, jorion=True)
    cfg = replace(KELLY, objetivo=Objetivo.MAX_SHARPE)
    esperado = otimizar(mu, cov, replace(cfg, taxa_livre_risco=0.024)).pesos
    assert estrategia_grande(cfg, rf)(treino).to_numpy() == pytest.approx(esperado.to_numpy(), abs=2e-3)


def test_with_rules_only_names_with_a_sector_enter_and_the_sector_cap_holds():
    from core.estrategias_todas_us import estrategia_grande

    treino = _mensal()
    setores = _setores(treino)
    del setores["N000"]
    pesos = estrategia_grande(KELLY, setores=setores)(treino)
    assert "N000" not in pesos.index
    assert pesos.groupby(setores).sum().max() <= 0.30 + 1e-6


def test_with_rules_a_violation_is_a_refusal():
    from core.estrategias_todas_us import estrategia_grande

    treino = _mensal()
    with pytest.raises(CarteiraRecusada):
        estrategia_grande(replace(KELLY, vol_maxima=None), setores=_setores(treino), limites=Limites(vol_maxima=0.0001))(treino)


def test_the_resampled_arm_holds_to_the_dense_one():
    from core.estrategias_todas_us import estrategia_reamostrada_grande

    gerador = np.random.default_rng(1)
    retornos = pd.DataFrame(gerador.normal(0.0006, 0.01, (300, 1)) + gerador.normal(0.0003, 0.015, (300, 12)),
                            columns=[f"N{i:03d}" for i in range(12)])
    config = ConfigReamostragem(8, 21, 0)
    esperado = otimizar_reamostrado(retornos, KELLY, config=config).resultado.pesos
    assert estrategia_reamostrada_grande(KELLY, config, periodos=252)(retornos).to_numpy() == pytest.approx(esperado.to_numpy(), abs=2e-3)


def test_the_large_hrp_is_the_dense_hrp():
    from core.estrategias_todas_us import hrp_grande
    from core.hierarchical import otimizar_hrp

    treino = _mensal(36, 30)
    assert hrp_grande(otimizar_hrp)(treino).sort_index().to_numpy() == pytest.approx(estrategia_hrp()(treino).sort_index().to_numpy(), abs=1e-9)


def test_the_views_arm_holds_to_the_dense_one_on_the_engines_scale():
    from core.estrategias_todas_us import estrategia_views_grande

    treino = _mensal(36, 24)
    setores = _setores(treino)
    colunas = list(treino.columns)
    nota = pd.Series(np.linspace(-1.0, 1.0, 24), index=colunas)
    sinais = [Sinal(lambda mes, candidatos: nota.reindex(candidatos), pd.Series(0.05, index=pd.date_range("2018-01-01", periods=60, freq="MS")))]
    mes = (treino.index[-1]).to_period("M").to_timestamp()
    capital = pd.DataFrame({"issuer": [papel[:4] for papel in colunas], "month": mes, "market_cap": np.arange(1.0, 25.0)})
    denso = estrategia_com_views(KELLY, setores, sinais, capital)
    esperado = no_grid_mensal(denso)(treino.set_axis(treino.index - pd.offsets.MonthEnd(1)))
    obtido = estrategia_views_grande(KELLY, setores, sinais, capital)(treino)
    assert obtido.to_numpy() == pytest.approx(esperado.to_numpy(), abs=2e-3)


def test_the_selection_reads_the_covariance_on_the_engines_scale_and_the_weights_read_real_monthly_means():
    from core.estrategias_todas_us import selecao_e_max_sharpe_mensal

    treino = _mensal()
    setores = _setores(treino)
    nomes = list(treino.columns[:12])
    visto = {}

    def selecionar(data, recebido, nota):
        visto["cov"] = covariancia_ledoit_wolf(recebido)
        visto["data"] = data
        return nomes

    cfg = ConfigOtimizacao(objetivo=Objetivo.MAX_SHARPE, peso_maximo=0.20, n_ativos_max=None, vol_maxima=0.40)
    grupos = limites_por_setor({papel: setores[papel] for papel in nomes}, 0.30)
    rf = pd.Series({treino.index[-2]: 0.002})
    mu, cov = _momentos(treino[nomes], jorion=True)
    esperado = otimizar(mu, cov, replace(cfg, taxa_livre_risco=0.024, grupos=grupos)).pesos

    pesos = selecao_e_max_sharpe_mensal(selecionar, lambda mes, candidatos: pd.Series(0.0, index=candidatos), setores, cfg, rf)(treino)

    assert visto["data"] == treino.index[-1]
    assert visto["cov"].to_numpy() == pytest.approx(covariancia_ledoit_wolf(treino).to_numpy() / 21, rel=1e-9)
    assert pesos.reindex(esperado.index).to_numpy() == pytest.approx(esperado.to_numpy(), abs=1e-6)
