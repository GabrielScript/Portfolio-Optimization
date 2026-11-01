"""The books' portfolios (docs/superpowers/specs/2026-11-01-todas-as-estrategias-design.md, section 3), each held to its definition
computed independently of the code that builds it."""

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
from scipy.optimize import minimize

from core.baixo_posto import densa, ledoit_wolf_baixo_posto
from core.config import ConfigOtimizacao, Objetivo, limites_por_setor
from core.estimation import estimar_momentos
from core.optimization import otimizar


def _retornos(linhas: int = 60, colunas: int = 12, semente: int = 0) -> pd.DataFrame:
    gerador = np.random.default_rng(semente)
    mercado = gerador.normal(0.005, 0.03, linhas)[:, None]
    dados = mercado * gerador.uniform(0.4, 1.6, colunas) + gerador.normal(0.002, 0.05, (linhas, colunas)) * gerador.uniform(0.5, 2.0, colunas)
    return pd.DataFrame(dados, columns=[f"S{i:02d}" for i in range(colunas)])


def _cfg(n_ativos_max: int | None = None, setorizado: bool = False, colunas: int = 12) -> ConfigOtimizacao:
    grupos = limites_por_setor({f"S{i:02d}": f"G{i % 4}" for i in range(colunas)}, 0.30) if setorizado else ()
    return ConfigOtimizacao(objetivo=Objetivo.KELLY, peso_maximo=0.20, n_ativos_max=n_ativos_max, vol_maxima=0.40, grupos=grupos)


def _carteiras_ao_acaso(colunas: int, quantas: int = 4000, semente: int = 3) -> np.ndarray:
    """Random long-only portfolios under the 20% cap, the brute-force check of an LP's optimum."""
    pesos = np.random.default_rng(semente).dirichlet(np.ones(colunas), quantas)
    return pesos[pesos.max(axis=1) <= 0.20]


def test_inverse_variance_and_inverse_volatility_follow_the_diagonal():
    from core.carteiras_livros import pesos_ivar, pesos_ivol

    sigma = ledoit_wolf_baixo_posto(_retornos())
    variancias = np.diag(densa(sigma).to_numpy())
    assert pesos_ivar(sigma).to_numpy() == pytest.approx((1 / variancias) / (1 / variancias).sum())
    assert pesos_ivol(sigma).to_numpy() == pytest.approx((variancias ** -0.5) / (variancias ** -0.5).sum())


def test_equal_risk_contribution_equalizes_each_names_share_of_risk():
    from core.carteiras_livros import pesos_rpp

    for linhas, colunas in ((60, 12), (36, 150)):
        sigma = ledoit_wolf_baixo_posto(_retornos(linhas, colunas))
        pesos = pesos_rpp(sigma).to_numpy()
        cov = densa(sigma).to_numpy()
        contribuicoes = pesos * (cov @ pesos) / (pesos @ cov @ pesos)
        assert pesos.sum() == pytest.approx(1.0) and pesos.min() > 0
        assert contribuicoes == pytest.approx(np.full(colunas, 1 / colunas), abs=1e-7)


def test_max_decorrelation_is_the_minimum_variance_on_the_correlation_divided_by_the_volatilities():
    from core.carteiras_livros import pesos_mdec

    sigma = ledoit_wolf_baixo_posto(_retornos())
    cov = densa(sigma)
    vol = np.sqrt(np.diag(cov))
    correlacao = cov / np.outer(vol, vol)
    cfg = ConfigOtimizacao(objetivo=Objetivo.MIN_VOL, peso_maximo=1.0, n_ativos_max=None, vol_maxima=None)
    na_correlacao = otimizar(pd.Series(0.0, index=cov.index), correlacao, cfg).pesos.to_numpy()
    esperado = (na_correlacao / vol) / (na_correlacao / vol).sum()
    assert pesos_mdec(sigma).to_numpy() == pytest.approx(esperado, abs=1e-3)


def test_herc_is_one_over_n_inside_and_equal_risk_across_two_separate_blocks():
    from core.carteiras_livros import pesos_herc

    gerador = np.random.default_rng(1)
    bloco_a, bloco_b = gerador.normal(0, 0.02, (120, 1)), gerador.normal(0, 0.08, (120, 1))
    ruido = gerador.normal(0, 0.002, (120, 8))
    retornos = pd.DataFrame(np.hstack([bloco_a + ruido[:, :4], bloco_b + ruido[:, 4:]]), columns=[f"S{i}" for i in range(8)])
    sigma = ledoit_wolf_baixo_posto(retornos)
    pesos = pesos_herc(sigma, clusters=2).to_numpy()
    assert pesos.sum() == pytest.approx(1.0)
    assert pesos[:4] == pytest.approx(np.full(4, pesos[0])) and pesos[4:] == pytest.approx(np.full(4, pesos[4]))
    cov = densa(sigma).to_numpy()
    um_a, um_b = np.r_[np.full(4, 0.25), np.zeros(4)], np.r_[np.zeros(4), np.full(4, 0.25)]
    # two groups: equal risk contribution means weight times volatility is the same on both sides, whatever their correlation
    assert pesos[:4].sum() * np.sqrt(um_a @ cov @ um_a) == pytest.approx(pesos[4:].sum() * np.sqrt(um_b @ cov @ um_b))


def test_max_diversification_is_max_sharpe_with_mu_the_volatilities():
    from core.carteiras_livros import pesos_mdiv

    retornos = _retornos()
    sigma = ledoit_wolf_baixo_posto(retornos)
    cov = densa(sigma)
    cfg = _cfg(5)
    esperado = otimizar(pd.Series(np.sqrt(np.diag(cov)), index=cov.index), cov, replace(cfg, objetivo=Objetivo.MAX_SHARPE)).pesos
    assert pesos_mdiv(sigma, cfg).to_numpy() == pytest.approx(esperado.to_numpy(), abs=5e-4)


def _cvar_empirico(perdas: np.ndarray, alfa: float) -> float:
    """Rockafellar-Uryasev's value at the sample's own tail, computed from the sorted losses."""
    ordenadas = np.sort(perdas)
    limiar = ordenadas[int(np.ceil(alfa * len(perdas))) - 1]
    return float(limiar + np.clip(perdas - limiar, 0.0, None).sum() / ((1 - alfa) * len(perdas)))


def test_mean_cvar_beats_every_random_portfolio_on_its_own_objective():
    from core.carteiras_livros import pesos_cvar

    retornos = _retornos()
    cenarios, mu = retornos.to_numpy(), retornos.mean().to_numpy()
    pesos = pesos_cvar(retornos, retornos.mean(), ledoit_wolf_baixo_posto(retornos, 12), _cfg(), alfa=0.90).to_numpy()

    def valor(w: np.ndarray) -> float:
        return float(mu @ w - _cvar_empirico(-cenarios @ w, 0.90))

    assert pesos.sum() == pytest.approx(1.0) and pesos.max() <= 0.20 + 1e-6
    assert valor(pesos) >= max(valor(w) for w in _carteiras_ao_acaso(12)) - 1e-6


def _drawdowns(cenarios: np.ndarray, w: np.ndarray) -> np.ndarray:
    acumulado = np.cumsum(cenarios @ w)
    return np.maximum.accumulate(np.maximum(acumulado, 0.0)) - acumulado


def test_mean_cdar_beats_every_random_portfolio_on_its_own_objective():
    from core.carteiras_livros import pesos_cdar

    retornos = _retornos()
    cenarios, mu = retornos.to_numpy(), retornos.mean().to_numpy()
    pesos = pesos_cdar(retornos, retornos.mean(), ledoit_wolf_baixo_posto(retornos, 12), _cfg(), alfa=0.90).to_numpy()

    def valor(w: np.ndarray) -> float:
        return float(mu @ w - _cvar_empirico(_drawdowns(cenarios, w), 0.90))

    assert pesos.sum() == pytest.approx(1.0) and pesos.max() <= 0.20 + 1e-6
    assert valor(pesos) >= max(valor(w) for w in _carteiras_ao_acaso(12)) - 1e-6


def test_mvsk_improves_its_objective_on_the_equal_weight_start_within_the_caps():
    from core.carteiras_livros import pesos_mvsk

    retornos = _retornos()
    sigma = ledoit_wolf_baixo_posto(retornos, 12)
    cenarios = retornos.to_numpy()
    media = cenarios.mean(axis=0)
    centrados = cenarios - media
    cov_periodo = densa(sigma).to_numpy() / 12

    def perda(w: np.ndarray) -> float:
        p = centrados @ w
        return float(-media @ w + 0.5 * w @ cov_periodo @ w - (p ** 3).mean() / 3 + (p ** 4).mean() / 4)

    pesos = pesos_mvsk(retornos, retornos.mean(), sigma, 12, _cfg()).to_numpy()
    assert pesos.sum() == pytest.approx(1.0) and pesos.min() >= -1e-9 and pesos.max() <= 0.20 + 1e-6
    assert perda(pesos) < perda(np.full(12, 1 / 12))
    inicios = [np.full(12, 1 / 12), *np.random.default_rng(5).dirichlet(np.full(12, 8.0), 6).clip(0.0, 0.2)]
    oraculo = min(minimize(perda, inicio / inicio.sum(), method="SLSQP", bounds=[(0.0, 0.20)] * 12, options={"maxiter": 500, "ftol": 1e-12},
                           constraints={"type": "eq", "fun": lambda w: w.sum() - 1.0}).fun for inicio in inicios)
    assert perda(pesos) <= oraculo + 2e-6


def test_mvsk_follows_the_higher_moments_when_the_means_are_equal_and_no_cap_binds():
    """Crash-prone names against lottery-like ones with the same mean and variance: only skewness tells them apart. The linearized steps
    must land on the optimum an independent solver finds on the full objective."""
    from core.carteiras_livros import pesos_mvsk

    gerador = np.random.default_rng(7)
    salto = (gerador.random((240, 6)) < 0.05) * 0.60
    choque = gerador.normal(0.0, 0.05, (240, 6))
    dados = choque + np.hstack([-salto[:, :3], salto[:, 3:]])
    dados = (dados - dados.mean(axis=0)) / dados.std(axis=0) * 0.15
    colunas = [f"S{i:02d}" for i in range(6)]
    retornos = pd.DataFrame(dados, columns=colunas)
    sigma = ledoit_wolf_baixo_posto(retornos, 12)
    cov_periodo = densa(sigma).to_numpy() / 12

    def perda(w: np.ndarray) -> float:
        p = dados @ w
        return float(0.5 * w @ cov_periodo @ w - (p ** 3).mean() / 3 + (p ** 4).mean() / 4)

    cfg = ConfigOtimizacao(objetivo=Objetivo.KELLY, peso_maximo=0.90, n_ativos_max=None, vol_maxima=None, grupos=())
    pesos = pesos_mvsk(retornos, retornos.mean(), sigma, 12, cfg).to_numpy()
    oraculo = minimize(perda, np.full(6, 1 / 6), method="SLSQP", bounds=[(0.0, 0.90)] * 6, options={"maxiter": 1000, "ftol": 1e-13},
                       constraints={"type": "eq", "fun": lambda w: w.sum() - 1.0})
    assert pesos[3:].sum() > pesos[:3].sum()
    assert perda(pesos) == pytest.approx(oraculo.fun, abs=1e-7)
    assert pesos == pytest.approx(oraculo.x, abs=5e-3)


def test_robust_mean_variance_with_kappa_zero_is_kelly_with_lambda_one():
    from core.carteiras_livros import pesos_robusto

    retornos = _retornos()
    sigma = ledoit_wolf_baixo_posto(retornos, 12)
    mu = retornos.mean() * 12
    esperado = otimizar(mu, densa(sigma), replace(_cfg(), aversao_risco=1.0)).pesos
    assert pesos_robusto(mu, sigma, replace(_cfg(), aversao_risco=1.0), 5.0, kappa=0.0).to_numpy() == pytest.approx(esperado.to_numpy(), abs=5e-4)


def test_a_larger_kappa_lowers_the_portfolios_volatility():
    from core.carteiras_livros import pesos_robusto

    retornos = _retornos()
    sigma = ledoit_wolf_baixo_posto(retornos, 12)
    mu = retornos.mean() * 12
    cov = densa(sigma).to_numpy()

    def vol(kappa: float) -> float:
        w = pesos_robusto(mu, sigma, replace(_cfg(), aversao_risco=1.0), 5.0, kappa=kappa).to_numpy()
        return float(np.sqrt(w @ cov @ w))

    assert vol(40.0) < vol(0.0)


def test_the_robust_portfolio_does_not_depend_on_the_unit_of_time():
    """S = Sigma/T holds per period. Annualizing multiplies the whole objective by the same factor, so the portfolio cannot move."""
    from core.carteiras_livros import pesos_robusto

    retornos = _retornos()
    cfg = ConfigOtimizacao(objetivo=Objetivo.KELLY, peso_maximo=0.40, n_ativos_max=None, vol_maxima=None, aversao_risco=1.0)

    def pesos(periodos: int) -> np.ndarray:
        return pesos_robusto(retornos.mean() * periodos, ledoit_wolf_baixo_posto(retornos, periodos), cfg, 60 / periodos, kappa=1.0).to_numpy()

    assert pesos(12) == pytest.approx(pesos(1), abs=2e-4)


def test_the_cardinality_holds_and_the_sector_caps_hold_on_the_concentrated_arms():
    from core.carteiras_livros import pesos_cvar

    retornos = _retornos()
    cfg = _cfg(6, setorizado=True)
    pesos = pesos_cvar(retornos, retornos.mean(), ledoit_wolf_baixo_posto(retornos, 12), cfg, alfa=0.90)
    assert (pesos > 1e-9).sum() <= 6 and pesos.sum() == pytest.approx(1.0)
    setores = {f"S{i:02d}": f"G{i % 4}" for i in range(12)}
    assert pesos.groupby(setores).sum().max() <= 0.30 + 1e-6


def test_the_period_mean_follows_the_engines_jorion_where_it_applies_and_is_raw_where_it_turns_itself_off():
    from core.carteiras_livros import media_por_periodo

    muitos_dados = _retornos(300, 12)
    sigma = ledoit_wolf_baixo_posto(muitos_dados)
    assert media_por_periodo(muitos_dados, sigma, 252).to_numpy() == pytest.approx(estimar_momentos(muitos_dados).mu.to_numpy() / 252)
    poucos_dados = _retornos(36, 200)
    assert media_por_periodo(poucos_dados, ledoit_wolf_baixo_posto(poucos_dados, 12), 12).to_numpy() == pytest.approx(poucos_dados.mean().to_numpy())
