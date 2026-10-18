import numpy as np
import pandas as pd
import pytest

from core.config import taxa_anual_para_aritmetica
from core.estimation import (aplicar_tilt_growth, covariancia_ewma, covariancia_ledoit_wolf, estimar_momentos,
                             pesos_ewma, retornos_simples, shrinkage_jorion)
from core.hierarchical import otimizar_hrp, otimizar_hrp_arvore
from core import metrics


def test_pesos_ewma_dobram_a_cada_meia_vida_rumo_ao_presente_e_somam_um():
    # meia-vida de 1 pregão: pesos crus 1/4, 1/2, 1 (do mais antigo ao mais recente), soma 7/4
    assert pesos_ewma(3, meia_vida=1) == pytest.approx([1 / 7, 2 / 7, 4 / 7])


@pytest.mark.parametrize("meia_vida", [0, -21])
def test_pesos_ewma_rejeitam_meia_vida_nao_positiva(meia_vida):
    with pytest.raises(ValueError, match="meia_vida"):
        pesos_ewma(3, meia_vida=meia_vida)


def test_covariancia_ewma_com_meia_vida_infinita_iguala_ledoit_wolf(retornos_sinteticos):
    """Pesos iguais: a EWMA encolhida tem que ser a Ledoit-Wolf amostral (mesmo centro, mesma escala)."""
    ewma = covariancia_ewma(retornos_sinteticos, meia_vida=1e12)
    amostral = covariancia_ledoit_wolf(retornos_sinteticos)
    np.testing.assert_allclose(ewma.values, amostral.values, rtol=1e-6)
    assert list(ewma.index) == list(ewma.columns) == list(retornos_sinteticos.columns)


def test_covariancia_ewma_pesa_mais_o_regime_recente():
    """
    950 pregões com vol 1% e os 50 últimos com 3%. Meia-vida 21: os 50 últimos levam
    1 - 0,5^(50/21) ~ 81% do peso, variância esperada ~7,5e-4; na amostral, ~1,4e-4.
    """
    gerador = np.random.default_rng(7)
    vols = np.r_[np.full(950, 0.01), np.full(50, 0.03)]
    retornos = pd.DataFrame(gerador.standard_normal((1000, 3)) * vols[:, None], columns=["a", "b", "c"])
    variancia_ewma = np.diag(covariancia_ewma(retornos, meia_vida=21).values)
    variancia_amostral = np.diag(covariancia_ledoit_wolf(retornos).values)
    assert (variancia_ewma > 3 * variancia_amostral).all()


def test_estimar_momentos_com_meia_vida_usa_a_ewma_no_sigma_e_no_jorion(retornos_sinteticos):
    momentos = estimar_momentos(retornos_sinteticos, shrink_mu=True, meia_vida_cov=63)
    cov_diaria = covariancia_ewma(retornos_sinteticos, meia_vida=63).values
    _, intensidade = shrinkage_jorion(retornos_sinteticos.mean().values, cov_diaria, len(retornos_sinteticos))
    np.testing.assert_allclose(momentos.cov.values, cov_diaria * 252)
    assert momentos.intensidade_shrinkage == pytest.approx(intensidade)


def test_jorion_invariante_a_anualizacao(retornos_sinteticos):
    """Claim 3 (verificado): calcular em unidades diárias ou anuais (com T/h) dá o mesmo w."""
    m = retornos_sinteticos.mean().values
    s = retornos_sinteticos.cov().values
    t, n = retornos_sinteticos.shape
    _, w_diario = shrinkage_jorion(m, s, t)
    h = 252.0
    mu, sigma = h * m, h * s
    uns = np.ones(n)
    mu0 = uns @ np.linalg.solve(sigma, mu) / (uns @ np.linalg.solve(sigma, uns))
    d = mu - mu0
    w_anual = (n + 2) / ((n + 2) + (t / h) * d @ np.linalg.solve(sigma, d))
    assert w_diario == pytest.approx(w_anual, rel=1e-10)
    assert 0.0 <= w_diario <= 1.0


def test_shrinkage_reduz_dispersao_de_mu(retornos_sinteticos):
    bruto = estimar_momentos(retornos_sinteticos, shrink_mu=False).mu
    encolhido = estimar_momentos(retornos_sinteticos, shrink_mu=True).mu
    assert encolhido.std() < bruto.std()


def test_retornos_simples_amostra_comum():
    precos = pd.DataFrame({"a": [10, 11, 12, 13.0], "b": [np.nan, 5, 5.5, 6.0]})
    r = retornos_simples(precos)
    assert len(r) == 2 and not r.isna().any().any()


def test_tilt_ignora_ativos_sem_score():
    mu = pd.Series([0.1, 0.1, 0.1], index=["a", "b", "sem_score"])
    cov = pd.DataFrame(np.eye(3) * 0.04, index=mu.index, columns=mu.index)
    scores = pd.Series({"a": 90.0, "b": 10.0})
    ajustado = aplicar_tilt_growth(mu, cov, scores, 0.5)
    assert ajustado["sem_score"] == pytest.approx(0.1)
    assert ajustado["a"] > 0.1 > ajustado["b"]


def test_sortino_contraexemplo_do_codigo_antigo():
    """Claim 5: desvio do subconjunto negativo = 0 com perdas reais; o correto é finito."""
    x = pd.Series([-0.01, -0.01, 0.03])
    assert x[x < 0].std() == 0.0
    dd = metrics.downside_deviation(x)
    assert dd == pytest.approx(np.sqrt(0.0002 / 3))
    assert np.isfinite(metrics.sortino(x)) and metrics.sortino(x) > 0


def test_drawdown_e_duracao():
    valores = pd.Series([100, 110, 99, 105, 111, 90, 95.0])
    serie, mdd, duracao = metrics.drawdown(valores)
    assert mdd == pytest.approx(90 / 111 - 1)
    assert duracao == 2


def test_conversao_selic_para_aritmetica():
    assert taxa_anual_para_aritmetica(0.1475) == pytest.approx(0.1377, abs=5e-4)


def test_alfa_por_regressao_recupera_parametros():
    gerador = np.random.default_rng(0)
    datas = pd.bdate_range("2021-01-01", periods=1000)
    r_b = pd.Series(gerador.normal(0.0004, 0.012, 1000), index=datas)
    rf = pd.Series(0.0003, index=datas)
    r_p = rf + 0.0002 + 1.3 * (r_b - rf) + gerador.normal(0, 0.002, 1000)
    valores_p = 100 * (1 + r_p).cumprod()
    valores_b = 100 * (1 + r_b).cumprod()
    comp = metrics.comparar_benchmark(valores_p, valores_b, rf)
    assert comp.beta == pytest.approx(1.3, abs=0.03)
    assert comp.alfa_anual == pytest.approx(0.0002 * 252, abs=0.03)


@pytest.mark.parametrize("alocador", [otimizar_hrp, otimizar_hrp_arvore])
def test_hrp_pesos_positivos_somam_um_sem_normalizar(retornos_sinteticos, alocador):
    """Claim 4 (prova corrigida): produto de fatores em (0,1) já soma 1."""
    pesos = alocador(retornos_sinteticos.cov())
    assert (pesos > 0).all()
    assert pesos.sum() == pytest.approx(1.0, abs=1e-12)


def test_hrp_dois_ativos_e_inverso_variancia():
    cov = pd.DataFrame([[0.04, 0.0], [0.0, 0.01]], index=["a", "b"], columns=["a", "b"])
    pesos = otimizar_hrp(cov)
    assert pesos["a"] == pytest.approx(0.2) and pesos["b"] == pytest.approx(0.8)


def _t_alfa_newey_west_de_referencia(y: np.ndarray, x: np.ndarray) -> float:
    """t do intercepto por MQO com variância de Newey-West (Bartlett, L = ⌊4 (T/100)^(2/9)⌋), escrita com laços."""
    t = len(y)
    X = np.column_stack([np.ones(t), x])
    beta = np.linalg.solve(X.T @ X, X.T @ y)
    e = y - X @ beta
    defasagens = int(np.floor(4 * (t / 100) ** (2 / 9)))
    S = sum(e[i] ** 2 * np.outer(X[i], X[i]) for i in range(t))
    for lag in range(1, defasagens + 1):
        peso = 1 - lag / (defasagens + 1)
        for i in range(lag, t):
            cruzado = e[i] * e[i - lag] * np.outer(X[i], X[i - lag])
            S = S + peso * (cruzado + cruzado.T)
    inversa = np.linalg.inv(X.T @ X)
    return float(beta[0] / np.sqrt((inversa @ S @ inversa)[0, 0]))


def test_t_do_alfa_usa_erro_padrao_de_newey_west():
    # resíduo AR(1) com rho 0,6: o t de MQO comum exagera a significância do alfa
    gerador = np.random.default_rng(11)
    r_b = gerador.normal(0.0004, 0.012, 600)
    ruido = np.zeros(600)
    for i in range(1, 600):
        ruido[i] = 0.6 * ruido[i - 1] + gerador.normal(0.0, 0.004)
    r_p = 0.0003 + 0.9 * r_b + ruido
    datas = pd.bdate_range("2022-01-03", periods=601)
    valores = pd.Series(np.r_[1.0, np.cumprod(1 + r_p)], index=datas)
    referencia = pd.Series(np.r_[1.0, np.cumprod(1 + r_b)], index=datas)
    comparacao = metrics.comparar_benchmark(valores, referencia, pd.Series(0.0, index=datas))
    assert comparacao.t_stat_alfa == pytest.approx(_t_alfa_newey_west_de_referencia(r_p, r_b), rel=1e-6)
