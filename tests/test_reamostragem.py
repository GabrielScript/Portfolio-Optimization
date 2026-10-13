"""Reamostragem (bagging de Michaud) com bootstrap circular em blocos."""

import numpy as np
import pandas as pd
import pytest

from core import reamostragem
from core.config import ConfigOtimizacao, Objetivo
from core.estimation import estimar_momentos
from core.optimization import ErroOtimizacao, otimizar
from core.reamostragem import ConfigReamostragem, indices_bootstrap_circular, otimizar_reamostrado


def _kelly(**kwargs) -> ConfigOtimizacao:
    base = dict(objetivo=Objetivo.KELLY, peso_maximo=0.35, n_ativos_max=None, vol_maxima=None,
                aversao_risco=2.0, taxa_livre_risco=0.0)
    base.update(kwargs)
    return ConfigOtimizacao(**base)


def _retornos(n_ativos: int, n_dias: int, semente: int, media: float = 0.0005) -> pd.DataFrame:
    gerador = np.random.default_rng(semente)
    correlacao = 0.3 * np.ones((n_ativos, n_ativos)) + 0.7 * np.eye(n_ativos)
    cov = (0.02**2) * correlacao
    dados = gerador.multivariate_normal(np.full(n_ativos, media), cov, size=n_dias)
    return pd.DataFrame(dados, columns=[f"A{i}.SA" for i in range(n_ativos)])


def _identidade(n_obs, tamanho_bloco, gerador):
    return np.arange(n_obs)


def test_indices_andam_de_um_em_um_dentro_do_bloco():
    idx = indices_bootstrap_circular(10, 4, np.random.default_rng(0))
    assert len(idx) == 10 and idx.min() >= 0 and idx.max() <= 9
    for j in range(9):
        if (j + 1) % 4 != 0:
            assert idx[j + 1] == (idx[j] + 1) % 10


def test_bloco_do_tamanho_da_amostra_e_uma_rotacao_circular():
    idx = indices_bootstrap_circular(10, 10, np.random.default_rng(3))
    assert sorted(idx) == list(range(10))


def test_bloco_maior_que_a_amostra_e_rejeitado(retornos_sinteticos):
    config = ConfigReamostragem(n_reamostras=2, tamanho_bloco=len(retornos_sinteticos) + 1)
    with pytest.raises(ValueError):
        otimizar_reamostrado(retornos_sinteticos, _kelly(), config=config)


def test_hrp_nao_e_suportado(retornos_sinteticos):
    with pytest.raises(ValueError):
        otimizar_reamostrado(retornos_sinteticos, _kelly(objetivo=Objetivo.HRP),
                             config=ConfigReamostragem(n_reamostras=2))


def test_bootstrap_identidade_reproduz_a_otimizacao_da_amostra_inteira(retornos_sinteticos, monkeypatch):
    """Oráculo: se toda reamostra é a amostra original, a média é a carteira original."""
    monkeypatch.setattr(reamostragem, "indices_bootstrap_circular", _identidade)
    cfg = _kelly()
    momentos = estimar_momentos(retornos_sinteticos, shrink_mu=True)
    esperado = otimizar(momentos.mu, momentos.cov, cfg).pesos

    res = otimizar_reamostrado(retornos_sinteticos, cfg, config=ConfigReamostragem(n_reamostras=3))

    np.testing.assert_allclose(res.resultado.pesos.values, esperado.values, atol=1e-4)
    np.testing.assert_array_equal(res.frequencia.values, (esperado > 0.01).astype(float).values)
    assert res.reamostras_validas == 3
    assert res.resultado.status == "otimo"


def test_devolve_os_momentos_da_amostra_inteira_usados_na_projecao(retornos_sinteticos):
    """A UI desenha a fronteira com estes momentos; têm de ser os mesmos da reamostragem."""
    def mais_um_ponto(mu, cov):
        return mu + 0.01

    momentos = estimar_momentos(retornos_sinteticos, shrink_mu=True)
    res = otimizar_reamostrado(retornos_sinteticos, _kelly(), config=ConfigReamostragem(n_reamostras=2),
                               ajustar_mu=mais_um_ponto)
    pd.testing.assert_series_equal(res.mu, momentos.mu + 0.01)
    pd.testing.assert_frame_equal(res.cov, momentos.cov)
    assert res.intensidade_shrinkage == pytest.approx(momentos.intensidade_shrinkage)


def test_carteira_final_respeita_tetos_cardinalidade_e_vol():
    retornos = _retornos(12, 600, semente=5)
    cfg = _kelly(peso_maximo=0.25, n_ativos_max=5, vol_maxima=0.30)
    res = otimizar_reamostrado(retornos, cfg, config=ConfigReamostragem(n_reamostras=20))
    pesos = res.resultado.pesos
    assert pesos.sum() == pytest.approx(1.0, abs=1e-6)
    assert (pesos >= 0).all() and (pesos <= 0.25 + 1e-4).all()
    assert int((pesos > 0).sum()) <= 5
    assert res.resultado.volatilidade <= 0.30 + 1e-4


def test_reamostragem_diversifica_quando_as_medias_verdadeiras_sao_iguais():
    """Médias verdadeiras iguais: a carteira de uma amostra concentra no ruído; a média das reamostras espalha."""
    retornos = _retornos(10, 500, semente=11)
    cfg = _kelly(peso_maximo=0.5)
    momentos = estimar_momentos(retornos, shrink_mu=False)
    base = otimizar(momentos.mu, momentos.cov, cfg).pesos
    res = otimizar_reamostrado(retornos, cfg, shrink_mu=False, config=ConfigReamostragem(n_reamostras=50))

    def n_efetivo(pesos):
        return 1.0 / float((pesos**2).sum())

    # Medido: 2,99 → 5,03 (razão 1,68; entre 1,63 e 2,36 em outras sementes). Sem a média, a razão é 1.
    assert n_efetivo(res.resultado.pesos) >= 1.4 * n_efetivo(base)


def test_mesma_semente_repete_e_outra_semente_muda(retornos_sinteticos):
    cfg = _kelly()
    a = otimizar_reamostrado(retornos_sinteticos, cfg, config=ConfigReamostragem(n_reamostras=5, semente=1))
    b = otimizar_reamostrado(retornos_sinteticos, cfg, config=ConfigReamostragem(n_reamostras=5, semente=1))
    c = otimizar_reamostrado(retornos_sinteticos, cfg, config=ConfigReamostragem(n_reamostras=5, semente=2))
    np.testing.assert_array_equal(a.resultado.pesos.values, b.resultado.pesos.values)
    assert not np.allclose(a.resultado.pesos.values, c.resultado.pesos.values, atol=1e-6)


def test_reamostra_que_falha_e_descartada_e_contada(retornos_sinteticos, monkeypatch):
    chamadas = []

    def falha_na_primeira(mu, cov, cfg):
        chamadas.append(1)
        if len(chamadas) == 1:
            raise ErroOtimizacao("solver falhou")
        return otimizar(mu, cov, cfg)

    monkeypatch.setattr(reamostragem, "otimizar", falha_na_primeira)
    res = otimizar_reamostrado(retornos_sinteticos, _kelly(), config=ConfigReamostragem(n_reamostras=4))
    assert len(chamadas) == 4
    assert res.reamostras_validas == 3
    assert res.resultado.status == "impreciso"
    assert res.resultado.pesos.sum() == pytest.approx(1.0, abs=1e-6)


def test_todas_as_reamostras_falhando_levanta_erro(retornos_sinteticos, monkeypatch):
    def sempre_falha(mu, cov, cfg):
        raise ErroOtimizacao("solver falhou")

    monkeypatch.setattr(reamostragem, "otimizar", sempre_falha)
    with pytest.raises(ErroOtimizacao):
        otimizar_reamostrado(retornos_sinteticos, _kelly(), config=ConfigReamostragem(n_reamostras=3))


def test_ajuste_de_mu_vale_dentro_de_cada_reamostra(retornos_sinteticos):
    """Um μ muito maior para A0 o leva ao teto em TODA reamostra (o tilt growth entra assim)."""
    def favorece_a0(mu, cov):
        return mu + pd.Series(5.0, index=["A0.SA"]).reindex(mu.index).fillna(0.0)

    res = otimizar_reamostrado(retornos_sinteticos, _kelly(), config=ConfigReamostragem(n_reamostras=5),
                               ajustar_mu=favorece_a0)
    assert res.resultado.pesos["A0.SA"] == pytest.approx(0.35, abs=1e-4)
    assert res.frequencia["A0.SA"] == 1.0
