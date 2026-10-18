import numpy as np
import pandas as pd
import pytest

from core.config import ConfigOtimizacao, LimiteGrupo, Objetivo, limites_por_setor
from core.estimation import estimar_momentos
from core.optimization import (ErroInviavel, ajustar_aos_tetos, combinar_nucleo_satelite, fronteira_eficiente,
                               otimizar, preparar_covariancia, projetar_pesos)


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
    grupo = LimiteGrupo("setor", frozenset({"A4.SA", "A3.SA"}), 0.10)
    cfg = ConfigOtimizacao(objetivo=Objetivo.MAX_RETORNO, peso_maximo=0.5, n_ativos_max=None,
                           vol_maxima=None, grupos=(grupo,))
    res = otimizar(mu, cov, cfg)
    assert res.pesos[["A3.SA", "A4.SA"]].sum() <= 0.10 + 1e-5


def test_limites_por_setor_criam_um_grupo_por_setor_com_o_mesmo_teto():
    grupos = limites_por_setor({"a": "Energia", "b": "Bancos", "c": "Energia"}, 0.30)
    assert grupos == (LimiteGrupo("Bancos", frozenset({"b"}), 0.30),
                      LimiteGrupo("Energia", frozenset({"a", "c"}), 0.30))


def test_inviavel_explicito():
    mu = pd.Series([0.1, 0.2], index=["a", "b"])
    cov = pd.DataFrame(np.eye(2) * 0.04, index=["a", "b"], columns=["a", "b"])
    with pytest.raises(ErroInviavel):
        otimizar(mu, cov, ConfigOtimizacao(peso_maximo=0.3, n_ativos_max=None))


def test_teto_de_grupo_inviavel_diz_quanto_os_ativos_somam_no_maximo():
    """3 ativos no mesmo setor, teto de 30% no setor e 50% por ativo: no máximo 30% < 100%."""
    mu = pd.Series([0.10, 0.20, 0.15], index=list("abc"))
    grupos = limites_por_setor({"a": "Energia", "b": "Energia", "c": "Energia"}, 0.30)
    cfg = ConfigOtimizacao(peso_maximo=0.5, n_ativos_max=None, vol_maxima=None, grupos=grupos)
    with pytest.raises(ErroInviavel, match="somam no máximo 30%"):
        otimizar(mu, _cov_diagonal([0.04] * 3, list("abc")), cfg)


def test_config_rejeita_cardinalidade_impossivel():
    with pytest.raises(ValueError):
        ConfigOtimizacao(peso_maximo=0.2, n_ativos_max=4)


def _cov_diagonal(variancias, tickers):
    return pd.DataFrame(np.diag(variancias), index=tickers, columns=tickers)


def test_projecao_mantem_alvo_que_ja_e_viavel():
    alvo = pd.Series([0.3, 0.3, 0.2, 0.2], index=list("abcd"))
    cfg = ConfigOtimizacao(peso_maximo=0.4, n_ativos_max=None, vol_maxima=None)
    proj = projetar_pesos(alvo, _cov_diagonal([0.04] * 4, list("abcd")), cfg)
    np.testing.assert_allclose(proj.pesos.values, alvo.values, atol=1e-5)
    assert proj.status == "otimo"


def test_projecao_fica_com_os_k_maiores_e_redistribui_por_igual():
    """Simplex com caixa: (0,30; 0,25; 0,20) somam 0,75 e cada um recebe +0,25/3."""
    alvo = pd.Series([0.30, 0.25, 0.20, 0.15, 0.10], index=list("abcde"))
    cfg = ConfigOtimizacao(peso_maximo=0.5, n_ativos_max=3, vol_maxima=None)
    proj = projetar_pesos(alvo, _cov_diagonal([0.04] * 5, list("abcde")), cfg)
    np.testing.assert_allclose(proj.pesos.values, [0.383333, 0.333333, 0.283333, 0.0, 0.0], atol=1e-5)


def test_projecao_respeita_teto_de_grupo():
    """a+b <= 0,5 ativa: a e b cedem 0,15 cada para c."""
    alvo = pd.Series([0.4, 0.4, 0.2], index=list("abc"))
    grupo = LimiteGrupo("ab", frozenset({"a", "b"}), 0.5)
    cfg = ConfigOtimizacao(peso_maximo=1.0, n_ativos_max=None, vol_maxima=None, grupos=(grupo,))
    proj = projetar_pesos(alvo, _cov_diagonal([0.04] * 3, list("abc")), cfg)
    np.testing.assert_allclose(proj.pesos.values, [0.25, 0.25, 0.5], atol=1e-5)


def test_projecao_sem_cardinalidade_quando_o_top_k_nao_cabe_no_teto_de_grupo():
    """Top-2 = {a, b} com a <= 0,35 e teto de 0,5: no máximo 0,85 < 1. Em todos os ativos o alvo já é viável."""
    alvo = pd.Series([0.30, 0.26, 0.24, 0.20], index=list("abcd"))
    grupo = LimiteGrupo("a", frozenset({"a"}), 0.35)
    cfg = ConfigOtimizacao(peso_maximo=0.5, n_ativos_max=2, vol_maxima=None, grupos=(grupo,))
    proj = projetar_pesos(alvo, _cov_diagonal([0.04] * 4, list("abcd")), cfg)
    np.testing.assert_allclose(proj.pesos.values, alvo.values, atol=1e-5)
    assert proj.status == "fallback" and "Limite de 2 ativos não aplicado" in proj.mensagem


def test_projecao_impoe_teto_de_vol_da_covariancia_dada():
    """sigma 20% e 50%, sem correlação, teto 20%: 0,29a² − 0,5a + 0,21 = 0 → a = 0,42/0,58."""
    alvo = pd.Series([0.5, 0.5], index=["a", "b"])
    cfg = ConfigOtimizacao(peso_maximo=1.0, n_ativos_max=None, vol_maxima=0.20)
    proj = projetar_pesos(alvo, _cov_diagonal([0.04, 0.25], ["a", "b"]), cfg)
    assert proj.pesos["a"] == pytest.approx(0.42 / 0.58, abs=1e-4)
    assert proj.status == "otimo"


def test_projecao_sem_o_teto_de_vol_quando_ele_e_inviavel():
    """Vol mínima possível com 20%/50% é ~18,6% > 10%: projeta só nas restrições lineares."""
    alvo = pd.Series([0.5, 0.5], index=["a", "b"])
    cfg = ConfigOtimizacao(peso_maximo=1.0, n_ativos_max=None, vol_maxima=0.10)
    proj = projetar_pesos(alvo, _cov_diagonal([0.04, 0.25], ["a", "b"]), cfg)
    np.testing.assert_allclose(proj.pesos.values, [0.5, 0.5], atol=1e-5)
    assert proj.status == "fallback" and "inviável" in proj.mensagem


def test_preparar_covariancia_simetrica_definida_positiva():
    cov = pd.DataFrame([[1.0, 1.0], [1.0, 1.0]])  # singular
    matriz = preparar_covariancia(cov)
    np.linalg.cholesky(matriz)
    np.testing.assert_allclose(matriz, matriz.T)


def test_nucleo_satelite_combina_os_pesos_na_fracao_do_satelite():
    nucleo = pd.Series({"A": 0.5, "B": 0.5})
    satelite = pd.Series({"B": 0.4, "C": 0.6})
    pesos = combinar_nucleo_satelite(nucleo, satelite, fracao_satelite=0.3)
    assert pesos.to_dict() == pytest.approx({"A": 0.35, "B": 0.47, "C": 0.18})


def test_ruido_do_solver_acima_do_teto_volta_ao_teto_sem_perder_a_soma():
    # A e B passam do teto de 20% em 3e-6 (renormalização depois de zerar pesos < 1e-4); E tem a folga exata
    pesos = pd.Series({"A": 0.200003, "B": 0.200003, "C": 0.2, "D": 0.2, "E": 0.199994})
    cfg = ConfigOtimizacao(peso_maximo=0.20, n_ativos_max=None, vol_maxima=None,
                           grupos=(LimiteGrupo("Setor", frozenset({"A", "B"}), 0.40),))
    ajustados = ajustar_aos_tetos(pesos, cfg)
    assert ajustados.to_dict() == pytest.approx({"A": 0.2, "B": 0.2, "C": 0.2, "D": 0.2, "E": 0.2}, abs=1e-12)
