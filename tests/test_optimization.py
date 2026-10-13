import numpy as np
import pandas as pd
import pytest

from core.config import ConfigOtimizacao, LimiteGrupo, Objetivo
from core.estimation import estimar_momentos
from core.optimization import ErroInviavel, fronteira_eficiente, otimizar, preparar_covariancia


def _momentos(retornos):
    m = estimar_momentos(retornos, shrink_mu=False)
    return m.mu, m.cov


@pytest.mark.parametrize("objetivo", list(Objetivo))
def test_pesos_validos_em_todos_objetivos(retornos_sinteticos, objetivo):
    mu, cov = _momentos(retornos_sinteticos)
    cfg = ConfigOtimizacao(objetivo=objetivo, peso_maximo=0.35, n_ativos_max=None,
                           vol_maxima=None, taxa_livre_risco=0.02)
    res = otimizar(mu, cov, cfg)
    assert res.pesos.sum() == pytest.approx(1.0, abs=1e-6)
    assert (res.pesos >= -1e-9).all()
    if objetivo is not Objetivo.HRP:
        assert (res.pesos <= 0.35 + 1e-4).all()


def test_max_sharpe_exato_domina_grade_densa(retornos_sinteticos):
    """Charnes-Cooper tem que ser >= qualquer ponto da fronteira (o método antigo era uma grade)."""
    mu, cov = _momentos(retornos_sinteticos)
    cfg = ConfigOtimizacao(objetivo=Objetivo.MAX_SHARPE, peso_maximo=0.4, n_ativos_max=None,
                           vol_maxima=None, taxa_livre_risco=0.03)
    res = otimizar(mu, cov, cfg)
    fronteira = fronteira_eficiente(mu, cov, cfg, n_pontos=200)
    assert res.status == "otimo"
    assert res.sharpe >= fronteira["sharpe"].max() - 1e-4


def test_max_sharpe_sem_restricao_ativa_bate_tangente_analitica():
    """Sem tetos ativos, w* proporcional a Sigma^-1 (mu - rf) (Markowitz/Tobin)."""
    mu = pd.Series([0.10, 0.12, 0.15], index=list("abc"))
    cov = pd.DataFrame([[0.04, 0.006, 0.004], [0.006, 0.05, 0.01], [0.004, 0.01, 0.06]],
                       index=list("abc"), columns=list("abc"))
    rf = 0.02
    analitico = np.linalg.solve(cov.values, mu.values - rf)
    analitico /= analitico.sum()
    assert (analitico > 0).all()
    cfg = ConfigOtimizacao(objetivo=Objetivo.MAX_SHARPE, peso_maximo=1.0, n_ativos_max=None,
                           vol_maxima=None, taxa_livre_risco=rf)
    res = otimizar(mu, cov, cfg)
    np.testing.assert_allclose(res.pesos.values, analitico, atol=1e-5)


def test_max_sharpe_cai_para_min_vol_quando_rf_domina(retornos_sinteticos):
    mu, cov = _momentos(retornos_sinteticos)
    cfg = ConfigOtimizacao(objetivo=Objetivo.MAX_SHARPE, peso_maximo=0.4, n_ativos_max=None,
                           vol_maxima=None, taxa_livre_risco=float(mu.max()) + 0.5)
    res = otimizar(mu, cov, cfg)
    min_vol = otimizar(mu, cov, ConfigOtimizacao(objetivo=Objetivo.MIN_VOL, peso_maximo=0.4,
                                                 n_ativos_max=None, vol_maxima=None))
    assert res.status == "fallback"
    np.testing.assert_allclose(res.pesos.values, min_vol.pesos.values, atol=1e-5)


def test_kelly_sem_restricao_ativa_bate_solucao_fechada():
    """max mu'w - (l/2) w'Sw s.a. 1'w=1: w = S^-1(mu - g 1)/l com g escolhido para somar 1."""
    mu = pd.Series([0.10, 0.12, 0.15], index=list("abc"))
    cov = pd.DataFrame(np.diag([0.04, 0.05, 0.06]), index=list("abc"), columns=list("abc"))
    lam = 3.0
    s_inv = np.linalg.inv(cov.values)
    uns = np.ones(3)
    gama = (uns @ s_inv @ mu.values - lam) / (uns @ s_inv @ uns)
    fechado = s_inv @ (mu.values - gama) / lam
    assert (fechado > 0).all()
    cfg = ConfigOtimizacao(objetivo=Objetivo.KELLY, peso_maximo=1.0, n_ativos_max=None,
                           vol_maxima=None, aversao_risco=lam)
    np.testing.assert_allclose(otimizar(mu, cov, cfg).pesos.values, fechado, atol=1e-5)


def test_kelly_invariante_a_anualizacao(retornos_sinteticos):
    """Claim 2 (verificado): f_A = h f_d, mesmo argmax."""
    mu, cov = _momentos(retornos_sinteticos)
    cfg = ConfigOtimizacao(objetivo=Objetivo.KELLY, peso_maximo=0.5, n_ativos_max=None,
                           vol_maxima=None, aversao_risco=1.0)
    anual = otimizar(mu, cov, cfg).pesos
    diario = otimizar(mu / 252, cov / 252, cfg).pesos
    np.testing.assert_allclose(anual.values, diario.values, atol=1e-4)


def test_teto_de_vol_respeitado_e_fallback_quando_inviavel(retornos_sinteticos):
    mu, cov = _momentos(retornos_sinteticos)
    cfg = ConfigOtimizacao(objetivo=Objetivo.MAX_RETORNO, peso_maximo=0.5, n_ativos_max=None,
                           vol_maxima=0.20)
    res = otimizar(mu, cov, cfg)
    assert res.volatilidade <= 0.20 + 1e-4
    inviavel = otimizar(mu, cov, ConfigOtimizacao(objetivo=Objetivo.KELLY, peso_maximo=0.5,
                                                  n_ativos_max=None, vol_maxima=0.01))
    assert inviavel.status == "fallback" and "inviável" in inviavel.mensagem


def test_cardinalidade(retornos_sinteticos):
    mu, cov = _momentos(retornos_sinteticos)
    cfg = ConfigOtimizacao(objetivo=Objetivo.MIN_VOL, peso_maximo=0.5, n_ativos_max=2,
                           vol_maxima=None)
    res = otimizar(mu, cov, cfg)
    assert int((res.pesos > 0).sum()) <= 2
    assert res.pesos.sum() == pytest.approx(1.0)


def test_teto_de_grupo(retornos_sinteticos):
    mu, cov = _momentos(retornos_sinteticos)
    grupo = LimiteGrupo("cripto", frozenset({"A4.SA", "A3.SA"}), 0.10)
    cfg = ConfigOtimizacao(objetivo=Objetivo.MAX_RETORNO, peso_maximo=0.5, n_ativos_max=None,
                           vol_maxima=None, grupos=(grupo,))
    res = otimizar(mu, cov, cfg)
    assert res.pesos[["A3.SA", "A4.SA"]].sum() <= 0.10 + 1e-5


def test_inviavel_explicito():
    mu = pd.Series([0.1, 0.2], index=["a", "b"])
    cov = pd.DataFrame(np.eye(2) * 0.04, index=["a", "b"], columns=["a", "b"])
    with pytest.raises(ErroInviavel):
        otimizar(mu, cov, ConfigOtimizacao(peso_maximo=0.3, n_ativos_max=None))


def test_config_rejeita_cardinalidade_impossivel():
    with pytest.raises(ValueError):
        ConfigOtimizacao(peso_maximo=0.2, n_ativos_max=4)


def test_preparar_covariancia_simetrica_definida_positiva():
    cov = pd.DataFrame([[1.0, 1.0], [1.0, 1.0]])  # singular
    matriz = preparar_covariancia(cov)
    np.linalg.cholesky(matriz)
    np.testing.assert_allclose(matriz, matriz.T)
