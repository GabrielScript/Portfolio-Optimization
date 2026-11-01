"""The second batch's portfolios (docs/superpowers/specs/2026-11-01-lote2-livro-design.md), each held to a result computed
independently of the code that builds it: SciPy's optimizers, a linear program, or the book's closed form."""

import math

import cvxpy as cp
import numpy as np
import pandas as pd
import pytest
from scipy.optimize import linprog, minimize

from core.baixo_posto import densa, ledoit_wolf_baixo_posto
from core.config import ConfigOtimizacao, Objetivo, limites_por_setor
from core.optimization import otimizar


def _retornos(linhas: int = 60, colunas: int = 12, semente: int = 0, escala: float = 1.0) -> pd.DataFrame:
    gerador = np.random.default_rng(semente)
    mercado = gerador.normal(0.005, 0.03, linhas)[:, None]
    dados = mercado * gerador.uniform(0.4, 1.6, colunas) + gerador.normal(0.002, 0.05, (linhas, colunas)) * gerador.uniform(0.5, 2.0, colunas)
    return pd.DataFrame(dados * escala, columns=[f"S{i:02d}" for i in range(colunas)])


def _cfg(n_ativos_max: int | None = None, vol_maxima: float | None = None, aversao: float = 1.0, peso_maximo: float = 0.20,
         grupos: tuple = ()) -> ConfigOtimizacao:
    return ConfigOtimizacao(objetivo=Objetivo.KELLY, peso_maximo=peso_maximo, n_ativos_max=n_ativos_max, vol_maxima=vol_maxima,
                            aversao_risco=aversao, grupos=grupos)


def _oraculo(perda, colunas: int, teto: float = 0.20, restricoes: tuple = (), partidas: int = 8) -> np.ndarray:
    """SciPy's SLSQP from several random starts, the best one that keeps every constraint."""
    melhor, valor = None, np.inf
    gerador = np.random.default_rng(7)
    for _ in range(partidas):
        inicio = np.clip(gerador.dirichlet(np.ones(colunas)), 0.0, teto)
        saida = minimize(perda, inicio / inicio.sum(), method="SLSQP", bounds=[(0.0, teto)] * colunas,
                         constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1.0}, *restricoes],
                         options={"maxiter": 2000, "ftol": 1e-13})
        if saida.success and saida.fun < valor:
            melhor, valor = saida.x, saida.fun
    assert melhor is not None
    return melhor


@pytest.mark.parametrize("escala, periodos", [(1.0, 12), (0.2, 252)])
def test_exact_kelly_is_the_maximum_of_the_mean_log(escala, periodos):
    from core.carteiras_lote2 import LOG, pesos_utilidade

    retornos = _retornos(escala=escala)
    pesos = pesos_utilidade(retornos, ledoit_wolf_baixo_posto(retornos, periodos), _cfg(), LOG)
    cenarios = retornos.to_numpy()
    oraculo = _oraculo(lambda w: -np.log1p(cenarios @ w).mean(), 12)
    assert np.log1p(cenarios @ pesos.to_numpy()).mean() == pytest.approx(np.log1p(cenarios @ oraculo).mean(), abs=1e-7)
    assert pesos.to_numpy() == pytest.approx(oraculo, abs=2e-2)


def test_root_utility_is_the_maximum_of_the_mean_root():
    from core.carteiras_lote2 import RAIZ, pesos_utilidade

    retornos = _retornos()
    pesos = pesos_utilidade(retornos, ledoit_wolf_baixo_posto(retornos, 12), _cfg(), RAIZ)
    cenarios = retornos.to_numpy()
    oraculo = _oraculo(lambda w: -np.sqrt(1.0 + cenarios @ w).mean(), 12)
    assert np.sqrt(1.0 + cenarios @ pesos.to_numpy()).mean() == pytest.approx(np.sqrt(1.0 + cenarios @ oraculo).mean(), abs=1e-7)


def test_exact_kelly_keeps_a_binding_volatility_cap():
    from core.carteiras_lote2 import LOG, pesos_utilidade

    retornos = _retornos()
    sigma = ledoit_wolf_baixo_posto(retornos, 12)
    cov = densa(sigma).to_numpy()
    livre = pesos_utilidade(retornos, sigma, _cfg(), LOG).to_numpy()
    teto = 0.9 * float(np.sqrt(livre @ cov @ livre))
    pesos = pesos_utilidade(retornos, sigma, _cfg(vol_maxima=teto), LOG).to_numpy()
    cenarios = retornos.to_numpy()
    oraculo = _oraculo(lambda w: -np.log1p(cenarios @ w).mean(), 12,
                       restricoes=({"type": "ineq", "fun": lambda w: teto ** 2 - w @ cov @ w},))
    assert np.sqrt(pesos @ cov @ pesos) <= teto + 1e-5
    assert np.log1p(cenarios @ pesos).mean() == pytest.approx(np.log1p(cenarios @ oraculo).mean(), abs=1e-6)


def test_exact_kelly_takes_the_two_stage_cardinality():
    from core.carteiras_lote2 import LOG, pesos_utilidade

    retornos = _retornos(linhas=100, colunas=30)
    pesos = pesos_utilidade(retornos, ledoit_wolf_baixo_posto(retornos, 12), _cfg(n_ativos_max=6), LOG)
    assert (pesos > 0).sum() <= 6
    assert pesos.sum() == pytest.approx(1.0)
    assert pesos.max() <= 0.20 + 1e-9


def test_exact_kelly_survives_a_name_that_lost_everything():
    from core.carteiras_lote2 import LOG, pesos_utilidade

    retornos = _retornos()
    retornos.iloc[10, 3] = -1.0
    pesos = pesos_utilidade(retornos, ledoit_wolf_baixo_posto(retornos, 12), _cfg(), LOG)
    assert np.isfinite(pesos.to_numpy()).all()
    assert pesos.sum() == pytest.approx(1.0)


@pytest.mark.parametrize("alfa", [2, 3])
def test_downside_is_the_optimum_of_its_smooth_problem(alfa):
    from core.carteiras_lote2 import pesos_downside

    retornos = _retornos()
    cenarios, media = retornos.to_numpy(), retornos.to_numpy().mean(axis=0)
    pesos = pesos_downside(retornos, ledoit_wolf_baixo_posto(retornos, 12), _cfg(), alfa).to_numpy()

    def perda(w: np.ndarray) -> float:
        return float(-(media @ w) + (np.clip(-(cenarios @ w), 0.0, None) ** alfa).mean())

    assert perda(pesos) == pytest.approx(perda(_oraculo(perda, 12)), abs=1e-6)


def test_downside_of_order_one_is_the_optimum_of_the_linear_program():
    from core.carteiras_lote2 import pesos_downside

    retornos = _retornos()
    cenarios, media = retornos.to_numpy(), retornos.to_numpy().mean(axis=0)
    linhas, colunas = cenarios.shape
    custo = np.concatenate([-media, np.full(linhas, 1.0 / linhas)])
    saida = linprog(custo, A_ub=np.hstack([-cenarios, -np.eye(linhas)]), b_ub=np.zeros(linhas),
                    A_eq=np.concatenate([np.ones(colunas), np.zeros(linhas)])[None], b_eq=[1.0],
                    bounds=[(0.0, 0.20)] * colunas + [(0.0, None)] * linhas)
    pesos = pesos_downside(retornos, ledoit_wolf_baixo_posto(retornos, 12), _cfg(), 1).to_numpy()
    assert -(media @ pesos) + np.clip(-(cenarios @ pesos), 0.0, None).mean() == pytest.approx(saida.fun, abs=1e-6)


def test_downside_reads_the_raw_mean_and_not_jorions():
    """With more rows than names ``media_por_periodo`` would shrink: the spec has these arms read the training rows directly."""
    from core.carteiras_lote2 import pesos_downside

    retornos = _retornos(linhas=120, colunas=6)
    cenarios, media = retornos.to_numpy(), retornos.to_numpy().mean(axis=0)
    pesos = pesos_downside(retornos, ledoit_wolf_baixo_posto(retornos, 12), _cfg(peso_maximo=0.40), 2).to_numpy()

    def perda(w: np.ndarray) -> float:
        return float(-(media @ w) + (np.clip(-(cenarios @ w), 0.0, None) ** 2).mean())

    assert perda(pesos) == pytest.approx(perda(_oraculo(perda, 6, teto=0.40)), abs=1e-6)


@pytest.mark.parametrize("ranking", ["mu", "mu_sigma", "mu_sigma2"])
def test_quintile_holds_the_top_fifth_equally(ranking):
    from core.carteiras_lote2 import pesos_quintil

    retornos = _retornos(colunas=15)
    sigma = ledoit_wolf_baixo_posto(retornos, 12)
    media = retornos.mean()
    variancias = pd.Series(np.diag(densa(sigma).to_numpy()), index=retornos.columns)
    nota = {"mu": media, "mu_sigma": media / np.sqrt(variancias), "mu_sigma2": media / variancias}[ranking]
    topo = nota.sort_values(ascending=False).index[:3]
    pesos = pesos_quintil(media, sigma, _cfg(peso_maximo=0.40), ranking)
    assert pesos[topo].to_numpy() == pytest.approx(np.full(3, 1.0 / 3.0))
    assert pesos.drop(topo).abs().max() == 0.0


def test_quintile_is_cut_to_the_caps_and_only_its_own_names_take_the_cut():
    """30 names, k = 6, three of the top in one sector: that sector is cut to 30%, the others rise to the 20% cap, and what is left is
    cash, because "cut to the caps" gives no name outside the quintile any money."""
    from core.carteiras_lote2 import pesos_quintil

    retornos = _retornos(linhas=80, colunas=30)
    media = retornos.mean()
    topo = list(media.sort_values(ascending=False).index[:6])
    grupos = limites_por_setor({nome: ("S0" if nome in topo[:3] else "S1" if nome in topo else f"F{i}") for i, nome in enumerate(media.index)}, 0.70)
    grupos = tuple(grupo if "S0" not in grupo.nome else type(grupo)(grupo.nome, grupo.tickers, 0.30) for grupo in grupos)
    pesos = pesos_quintil(media, ledoit_wolf_baixo_posto(retornos, 12), _cfg(grupos=grupos), "mu")
    assert pesos[topo[:3]].to_numpy() == pytest.approx(np.full(3, 0.10))
    assert pesos[topo[3:]].to_numpy() == pytest.approx(np.full(3, 0.20))
    assert pesos.drop(topo).abs().max() == 0.0
    assert pesos.sum() == pytest.approx(0.90)


def test_l1_robust_mean_is_the_quintile_portfolio():
    """Palomar's Example 14.3: max w'mu - eps ||w||_inf holds the top k equally, so it is no arm of its own."""
    from core.carteiras_lote2 import pesos_quintil

    retornos = _retornos(colunas=15)
    media = retornos.mean()
    ordenada = media.sort_values(ascending=False).to_numpy()
    # 1/3 on the top three beats 1/2 on two and 1/4 on four exactly between these two sizes
    menor = 6.0 * (ordenada[:2].mean() - ordenada[:3].mean())
    maior = 12.0 * (ordenada[:3].mean() - ordenada[:4].mean())
    assert menor < maior
    w = cp.Variable(15)
    cp.Problem(cp.Maximize(media.to_numpy() @ w - ((menor + maior) / 2.0) * cp.norm(w, "inf")), [cp.sum(w) == 1, w >= 0]).solve(solver="CLARABEL")
    pesos = pesos_quintil(media, ledoit_wolf_baixo_posto(retornos, 12), _cfg(peso_maximo=0.40), "mu")
    assert w.value == pytest.approx(pesos.to_numpy(), abs=1e-5)


def test_box_robust_mean_is_the_optimum_with_each_mean_cut_by_its_error():
    from core.carteiras_lote2 import pesos_box

    retornos = _retornos()
    sigma = ledoit_wolf_baixo_posto(retornos, 12)
    cov = densa(sigma).to_numpy()
    mu = retornos.mean() * 12
    anos = 5.0
    corte = np.sqrt(np.diag(cov)) / np.sqrt(anos)
    pesos = pesos_box(mu, sigma, _cfg(aversao=8.0, peso_maximo=0.60), anos).to_numpy()

    def perda(w: np.ndarray) -> float:
        return float(-(mu.to_numpy() - corte) @ w + 4.0 * w @ cov @ w)

    oraculo = _oraculo(perda, 12, teto=0.60)
    assert oraculo.max() < 0.60 - 1e-3     # inside the cap: the width of the box shows in the weights
    assert pesos == pytest.approx(oraculo, abs=2e-3)


def test_return_constraint_is_the_minimum_variance_above_the_mean_of_the_names():
    from core.carteiras_lote2 import pesos_retorno_minimo

    retornos = _retornos()
    sigma = ledoit_wolf_baixo_posto(retornos, 12)
    cov = densa(sigma).to_numpy()
    mu = (retornos.mean() * 12).to_numpy()
    beta = mu.mean()
    pesos = pesos_retorno_minimo(pd.Series(mu, index=retornos.columns), sigma, _cfg()).to_numpy()
    oraculo = _oraculo(lambda w: w @ cov @ w, 12, restricoes=({"type": "ineq", "fun": lambda w: mu @ w - beta},))
    assert pesos @ mu >= beta - 1e-6
    assert pesos @ cov @ pesos == pytest.approx(oraculo @ cov @ oraculo, abs=1e-6)


def test_return_constraint_falls_back_to_the_minimum_variance_when_nothing_meets_it():
    from core.carteiras_lote2 import pesos_retorno_minimo

    retornos = _retornos()
    sigma = ledoit_wolf_baixo_posto(retornos, 12)
    cov = densa(sigma).to_numpy()
    pesos = pesos_retorno_minimo(retornos.mean() * 12, sigma, _cfg(vol_maxima=0.001)).to_numpy()
    oraculo = _oraculo(lambda w: w @ cov @ w, 12)
    assert pesos @ cov @ pesos == pytest.approx(oraculo @ cov @ oraculo, abs=1e-6)


def test_subset_resampling_is_the_average_of_kelly_on_random_subsets():
    """Three draws by hand with the dense optimizer, the average then projected: what the spec's row 82 defines."""
    from core.carteiras_lote2 import ConfigSubconjunto, pesos_subconjunto
    from core.carteiras_livros import media_por_periodo
    from core.optimization import projetar_pesos

    retornos = _retornos(linhas=80, colunas=20)
    cfg = _cfg(n_ativos_max=8, vol_maxima=0.60)
    tamanho = math.ceil(20 ** 0.7)
    gerador = np.random.default_rng(0)
    soma = pd.Series(0.0, index=retornos.columns)
    for _ in range(3):
        nomes = list(retornos.columns[np.sort(gerador.choice(20, tamanho, replace=False))])
        sigma = ledoit_wolf_baixo_posto(retornos[nomes], 12)
        mu = media_por_periodo(retornos[nomes], sigma, 12) * 12
        soma = soma.add(otimizar(mu, densa(sigma), _cfg(vol_maxima=0.60, aversao=2.0)).pesos, fill_value=0.0)
    esperado = projetar_pesos(soma / 3.0, densa(ledoit_wolf_baixo_posto(retornos, 12)), cfg).pesos
    pesos = pesos_subconjunto(retornos, 12, cfg, ConfigSubconjunto(reamostras=3))
    assert pesos.to_numpy() == pytest.approx(esperado.reindex(pesos.index).fillna(0.0).to_numpy(), abs=2e-3)


def test_subset_bagging_also_resamples_the_rows():
    from core.carteiras_lote2 import ConfigSubconjunto, pesos_subconjunto

    retornos = _retornos(linhas=80, colunas=20)
    cfg = _cfg(n_ativos_max=8, vol_maxima=0.60)
    sem_linhas = pesos_subconjunto(retornos, 12, cfg, ConfigSubconjunto(reamostras=5))
    com_linhas = pesos_subconjunto(retornos, 12, cfg, ConfigSubconjunto(reamostras=5, bloco=1))
    assert not np.allclose(sem_linhas.to_numpy(), com_linhas.to_numpy(), atol=1e-3)
    assert com_linhas.sum() == pytest.approx(1.0)
    assert (com_linhas > 0).sum() <= 8


def test_subset_drops_a_draw_that_has_no_feasible_portfolio():
    """Under a 30% sector cap a draw whose names sit in too few sectors has no portfolio; it is left out of the average."""
    from core.carteiras_lote2 import ConfigSubconjunto, pesos_subconjunto

    retornos = _retornos(linhas=80, colunas=20)
    grupos = limites_por_setor({nome: ("S0" if i < 15 else f"S{i}") for i, nome in enumerate(retornos.columns)}, 0.30)
    pesos = pesos_subconjunto(retornos, 12, _cfg(grupos=grupos), ConfigSubconjunto(reamostras=5))
    assert pesos.sum() == pytest.approx(1.0)
    assert pesos[retornos.columns[:15]].sum() <= 0.30 + 1e-6


def _momentos(cenarios: np.ndarray, media: np.ndarray, cov: np.ndarray, w: np.ndarray) -> np.ndarray:
    centrados = (cenarios - cenarios.mean(axis=0)) @ w
    return np.array([media @ w, w @ cov @ w, (centrados ** 3).mean(), (centrados ** 4).mean()])


def _melhora(momentos: np.ndarray, base: np.ndarray) -> float:
    """The book's delta: the smallest proportional improvement over the four moments."""
    return float(min((momentos[0] - base[0]) / abs(base[0]), (base[1] - momentos[1]) / base[1],
                     (momentos[2] - base[2]) / abs(base[2]), (base[3] - momentos[3]) / base[3]))


@pytest.mark.parametrize("semente", range(6))
def test_tilting_reaches_the_improvement_of_an_independent_optimizer(semente):
    from core.carteiras_lote2 import pesos_tilting

    retornos = _retornos(semente=semente)
    sigma = ledoit_wolf_baixo_posto(retornos, 12)
    cov, cenarios, media = densa(sigma).to_numpy() / 12, retornos.to_numpy(), retornos.mean()
    base = pd.Series(np.r_[np.full(10, 0.1), np.zeros(2)], index=retornos.columns)
    pesos = pesos_tilting(retornos, base, media, sigma, 12, _cfg()).to_numpy()
    no_base = _momentos(cenarios, media.to_numpy(), cov, base.to_numpy())
    herfindahl = 1.01 * float(base @ base)

    def perda(x: np.ndarray) -> float:
        return -x[-1]

    def folgas(x: np.ndarray) -> np.ndarray:
        m = _momentos(cenarios, media.to_numpy(), cov, x[:-1])
        return np.array([(m[0] - no_base[0]) / abs(no_base[0]), (no_base[1] - m[1]) / no_base[1],
                         (m[2] - no_base[2]) / abs(no_base[2]), (no_base[3] - m[3]) / no_base[3]]) - x[-1]

    saida = minimize(perda, np.r_[base.to_numpy(), 0.0], method="SLSQP", bounds=[(0.0, 0.20)] * 12 + [(None, None)],
                     constraints=[{"type": "eq", "fun": lambda x: x[:-1].sum() - 1.0}, {"type": "ineq", "fun": folgas},
                                  {"type": "ineq", "fun": lambda x: herfindahl - x[:-1] @ x[:-1]}],
                     options={"maxiter": 2000, "ftol": 1e-12})
    # SLSQP's line search may stall (status 8) at a good point: what counts is the true delta of the point it stopped at, if feasible
    oraculo = saida.x[:-1]
    assert abs(oraculo.sum() - 1.0) < 1e-6 and oraculo @ oraculo <= herfindahl + 1e-6
    alcancado = _melhora(_momentos(cenarios, media.to_numpy(), cov, pesos), no_base)
    assert alcancado > 0.0
    assert alcancado >= _melhora(_momentos(cenarios, media.to_numpy(), cov, oraculo), no_base) - 0.01
    assert pesos @ pesos <= herfindahl + 1e-6
    assert pesos.max() <= 0.20 + 1e-9


def test_tilting_keeps_the_reference_when_it_cannot_improve_all_four():
    """The reference at the top of the caps with the best mean: raising the mean is impossible, so delta stays 0."""
    from core.carteiras_lote2 import pesos_tilting

    retornos = _retornos()
    sigma = ledoit_wolf_baixo_posto(retornos, 12)
    media = retornos.mean()
    melhores = media.sort_values(ascending=False).index[:5]
    base = pd.Series(0.0, index=retornos.columns)
    base[melhores] = 0.20
    pesos = pesos_tilting(retornos, base, media, sigma, 12, _cfg())
    assert pesos.to_numpy() == pytest.approx(base.to_numpy(), abs=1e-6)


def test_subset_keeps_the_order_of_names_that_are_not_sorted():
    from core.carteiras_lote2 import ConfigSubconjunto, pesos_subconjunto

    retornos = _retornos(linhas=80, colunas=20)
    retornos = retornos[list(retornos.columns[::-1])]
    pesos = pesos_subconjunto(retornos, 12, _cfg(n_ativos_max=8, vol_maxima=0.60), ConfigSubconjunto(reamostras=3))
    assert list(pesos.index) == list(retornos.columns)


@pytest.mark.parametrize("nome", ["LOG", "RAIZ"])
def test_each_utilitys_derivatives_are_its_own(nome):
    import core.carteiras_lote2 as modulo

    utilidade = getattr(modulo, nome)
    y, passo = np.array([0.7, 1.0, 1.3]), 1e-5
    assert utilidade.primeira(y) == pytest.approx((utilidade.valor(y + passo) - utilidade.valor(y - passo)) / (2 * passo), rel=1e-6)
    assert utilidade.segunda(y) == pytest.approx((utilidade.primeira(y + passo) - utilidade.primeira(y - passo)) / (2 * passo), rel=1e-6)
    assert utilidade.valor(np.array([4.0]))[0] == pytest.approx({"LOG": np.log(4.0), "RAIZ": 2.0}[nome])
