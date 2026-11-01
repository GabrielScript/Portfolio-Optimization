"""
carteiras_livros.py - The portfolios of Palomar's book and of López de Prado's chapter 16 that the project had not run
(docs/superpowers/specs/2026-11-01-todas-as-estrategias-design.md, section 3). All work on the covariance's low-rank form, from
hundreds of names on thousands of sessions (the B3) to thousands of names on 36 months (the US). New module: the paper guards'
paths stay as they are.

The whole-universe arms (IVarP, IVolP, MDecP, RPP, HERC) ignore the mean. The concentrated ones (MDivP, Mean-CVaR, Mean-CDaR, MVSK,
Robust-MV) take the app's two-stage cardinality and caps, and raise ``ErroOtimizacao`` as ``otimizar`` does.
"""

from __future__ import annotations

import heapq
from collections.abc import Callable
from dataclasses import replace

import cvxpy as cp
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import ClusterNode, to_tree

from core.baixo_posto import CovarianciaBaixoPosto, _min_vol, _risco_e_desvio, densa, otimizar_baixo_posto
from core.config import ConfigOtimizacao, Objetivo
from core.estimation import shrinkage_jorion
from core.hierarchical import _arvore
from core.optimization import (ErroInviavel, ErroOtimizacao, _checar_capacidade, _checar_capacidade_grupos, _limpar_pesos,
                               _resolver, _restricoes, ajustar_aos_tetos)

ALFA = 0.90                 # 0.95 on 36 rows would rest on 2 scenarios (spec, section 3)
CLUSTERS = 12
PASSOS_MVSK = 20
PARADA_MVSK = 1e-5
RECUOS_MVSK = (1.0, 0.5, 0.25, 0.125)
VOLTAS_RPP = 5_000
PARADA_RPP = 1e-11


def media_por_periodo(retornos: pd.DataFrame, sigma: CovarianciaBaixoPosto, periodos: int) -> pd.Series:
    """The mean per period as ``estimar_momentos`` takes it: Jorion's shrinkage, which turns itself off with fewer rows than names
    plus two, so then the raw mean (the US case), without building the matrix it would not use."""
    linhas, colunas = retornos.shape
    media = retornos.mean().to_numpy()
    if linhas <= colunas + 2:
        return pd.Series(media, index=retornos.columns)
    return pd.Series(shrinkage_jorion(media, densa(sigma).to_numpy() / periodos, linhas)[0], index=retornos.columns)


def _variancias(sigma: CovarianciaBaixoPosto) -> np.ndarray:
    return sigma.diagonal + (sigma.fator ** 2).sum(axis=0)


def _normalizado(pesos: np.ndarray, nomes: list[str]) -> pd.Series:
    return pd.Series(pesos / pesos.sum(), index=nomes)


def pesos_ivar(sigma: CovarianciaBaixoPosto) -> pd.Series:
    return _normalizado(1.0 / _variancias(sigma), sigma.nomes)


def pesos_ivol(sigma: CovarianciaBaixoPosto) -> pd.Series:
    return _normalizado(_variancias(sigma) ** -0.5, sigma.nomes)


def pesos_rpp(sigma: CovarianciaBaixoPosto) -> pd.Series:
    """Equal risk contribution by cyclical coordinate descent on the correlation matrix (Palomar ch. 11): each x_i in closed form,
    keeping F x so that a step costs O(T). The result does not depend on the scale, so it is divided by the volatilities at the end."""
    variancias = _variancias(sigma)
    fator = sigma.fator / np.sqrt(variancias)
    colunas = fator.shape[1]
    orcamento = 1.0 / colunas
    normas = (fator ** 2).sum(axis=0)
    x = np.full(colunas, 1.0 / np.sqrt(colunas))
    produto = fator @ x
    for _ in range(VOLTAS_RPP):
        anterior = x.copy()
        for i in range(colunas):
            fora_da_diagonal = float(fator[:, i] @ produto) - normas[i] * x[i]
            novo = (-fora_da_diagonal + np.sqrt(fora_da_diagonal ** 2 + 4.0 * orcamento)) / 2.0
            produto += fator[:, i] * (novo - x[i])
            x[i] = novo
        if np.abs(x - anterior).max() <= PARADA_RPP * x.max():
            return _normalizado(x / np.sqrt(variancias), sigma.nomes)
    raise ErroOtimizacao(f"RPP did not converge in {VOLTAS_RPP} sweeps")


def pesos_mdec(sigma: CovarianciaBaixoPosto) -> pd.Series:
    """Minimum variance on the correlation matrix, long-only, divided by the volatilities (Palomar ch. 6)."""
    variancias = _variancias(sigma)
    w = cp.Variable(len(variancias))
    risco = cp.sum_squares(cp.multiply(np.sqrt(sigma.diagonal / variancias), w)) + cp.sum_squares((sigma.fator / np.sqrt(variancias)) @ w)
    limpos = _limpar_pesos(_resolver(cp.Problem(cp.Minimize(risco), [cp.sum(w) == 1, w >= 0]), w)[0])
    return _normalizado(limpos / np.sqrt(variancias), sigma.nomes)


def _vol_de_um_sobre_n(sigma: CovarianciaBaixoPosto, membros: list[int]) -> float:
    return float(np.sqrt(sigma.diagonal / len(membros) + (sigma.fator[:, membros].mean(axis=1) ** 2).sum()))


def _cortar(raiz: ClusterNode, clusters: int) -> set[int]:
    """The tree's nodes that stay whole: splits the highest node until there are ``clusters`` of them."""
    fila = [(-raiz.dist, raiz.id, raiz)]
    folhas: set[int] = set()
    while fila and len(fila) + len(folhas) < clusters:
        _, _, no = heapq.heappop(fila)
        if no.is_leaf():
            folhas.add(no.id)
            continue
        for filho in (no.left, no.right):
            heapq.heappush(fila, (-filho.dist, filho.id, filho))
    return folhas | {no.id for _, _, no in fila}


def pesos_herc(sigma: CovarianciaBaixoPosto, clusters: int = CLUSTERS) -> pd.Series:
    """Raffinot's HERC (Palomar ch. 12): Ward on the correlation distance, cut at ``clusters``, 1/N inside. Each split shares its
    capital by equal risk contribution: for two groups that means inversely to their volatilities, whatever their correlation."""
    nomes = sigma.nomes
    raiz = to_tree(_arvore(densa(sigma), "ward"))
    inteiros = _cortar(raiz, clusters)
    pesos = np.zeros(len(nomes))
    pilha = [(raiz, 1.0)]
    while pilha:
        no, capital = pilha.pop()
        if no.id in inteiros:
            membros = no.pre_order()
            pesos[membros] = capital / len(membros)
            continue
        vol_esquerda = _vol_de_um_sobre_n(sigma, no.left.pre_order())
        vol_direita = _vol_de_um_sobre_n(sigma, no.right.pre_order())
        fracao = vol_direita / (vol_esquerda + vol_direita)
        pilha += [(no.left, capital * fracao), (no.right, capital * (1.0 - fracao))]
    return _normalizado(pesos, nomes)


def pesos_mdiv(sigma: CovarianciaBaixoPosto, cfg: ConfigOtimizacao) -> pd.Series:
    """The most diversified portfolio: max Sharpe with mu = the volatilities and rf 0 (Palomar ch. 6)."""
    return otimizar_baixo_posto(pd.Series(np.sqrt(_variancias(sigma)), index=sigma.nomes), sigma,
                                replace(cfg, objetivo=Objetivo.MAX_SHARPE, taxa_livre_risco=0.0))


def _parte(sigma: CovarianciaBaixoPosto, indices: np.ndarray) -> CovarianciaBaixoPosto:
    return CovarianciaBaixoPosto(sigma.diagonal, sigma.fator[:, indices], [sigma.nomes[i] for i in indices])


def _duas_etapas(sigma: CovarianciaBaixoPosto, cfg: ConfigOtimizacao, resolver: Callable[[np.ndarray], np.ndarray]) -> pd.Series:
    """The app's cardinality: solve on all, keep the largest ``n_ativos_max``, solve again on them; if those do not fit the caps, the
    cardinality stays unapplied, as in ``otimizar``. ``resolver`` takes the names' positions and gives their weights."""
    _checar_capacidade(len(sigma.nomes), cfg)
    _checar_capacidade_grupos(sigma.nomes, cfg)
    pesos = _limpar_pesos(resolver(np.arange(len(sigma.nomes))))
    if cfg.n_ativos_max is not None and int((pesos > 0).sum()) > cfg.n_ativos_max:
        selecionados = np.argsort(pesos)[::-1][:cfg.n_ativos_max]
        try:
            finais = np.zeros(len(pesos))
            finais[selecionados] = _limpar_pesos(resolver(selecionados))
            pesos = finais
        except ErroOtimizacao:
            pass
    return ajustar_aos_tetos(pd.Series(pesos, index=sigma.nomes), cfg)


def _maximizar_com_teto(objetivo: cp.Expression, w: cp.Variable, sigma: CovarianciaBaixoPosto, cfg: ConfigOtimizacao,
                        extras: list) -> np.ndarray:
    """The volatility cap as Kelly's: if no portfolio meets it, the minimum variance."""
    restricoes = [*_restricoes(w, sigma.nomes, cfg), *extras]
    if cfg.vol_maxima is not None:
        restricoes.append(_risco_e_desvio(w, sigma)[1] <= cfg.vol_maxima)
    try:
        return _resolver(cp.Problem(cp.Maximize(objetivo), restricoes), w)[0]
    except ErroInviavel:
        if cfg.vol_maxima is None:
            raise
    return _min_vol(sigma, cfg)


def pesos_cvar(retornos: pd.DataFrame, media: pd.Series, sigma: CovarianciaBaixoPosto, cfg: ConfigOtimizacao,
               alfa: float = ALFA) -> pd.Series:
    """max w'mu - CVaR(alfa), both per period, Rockafellar-Uryasev's LP on the training rows (Palomar ch. 10)."""
    cenarios = retornos.to_numpy()

    def resolver(indices: np.ndarray) -> np.ndarray:
        w, tau, folga = cp.Variable(len(indices)), cp.Variable(), cp.Variable(len(cenarios), nonneg=True)
        objetivo = media.to_numpy()[indices] @ w - (tau + cp.sum(folga) / ((1.0 - alfa) * len(cenarios)))
        return _maximizar_com_teto(objetivo, w, _parte(sigma, indices), cfg, [folga >= -cenarios[:, indices] @ w - tau])
    return _duas_etapas(sigma, cfg, resolver)


def pesos_cdar(retornos: pd.DataFrame, media: pd.Series, sigma: CovarianciaBaixoPosto, cfg: ConfigOtimizacao,
               alfa: float = ALFA) -> pd.Series:
    """max w'mu - CDaR(alfa), both per period, on uncompounded drawdowns from a starting peak of zero, an LP (Palomar ch. 10)."""
    acumulados = np.cumsum(retornos.to_numpy(), axis=0)
    linhas = len(acumulados)

    def resolver(indices: np.ndarray) -> np.ndarray:
        w, tau = cp.Variable(len(indices)), cp.Variable()
        pico, folga = cp.Variable(linhas, nonneg=True), cp.Variable(linhas, nonneg=True)
        carteira = acumulados[:, indices] @ w
        objetivo = media.to_numpy()[indices] @ w - (tau + cp.sum(folga) / ((1.0 - alfa) * linhas))
        extras = [pico >= carteira, pico[1:] >= pico[:-1], folga >= pico - carteira - tau]
        return _maximizar_com_teto(objetivo, w, _parte(sigma, indices), cfg, extras)
    return _duas_etapas(sigma, cfg, resolver)


def pesos_robusto(mu: pd.Series, sigma: CovarianciaBaixoPosto, cfg: ConfigOtimizacao, anos: float, kappa: float = 1.0) -> pd.Series:
    """max w'mu - kappa ||S^(1/2) w|| - (lambda/2) w'Sigma w with S = Sigma/T: the worst case of mu in an ellipsoid (Palomar ch. 14).
    ``mu`` and ``sigma`` are annual, so T is in years: with T in rows the penalty would shrink by the root of the periods per year."""
    def resolver(indices: np.ndarray) -> np.ndarray:
        parte = _parte(sigma, indices)
        w = cp.Variable(len(indices))
        risco, desvio = _risco_e_desvio(w, parte)
        objetivo = mu.to_numpy()[indices] @ w - kappa * desvio / np.sqrt(anos) - (cfg.aversao_risco / 2.0) * risco
        return _maximizar_com_teto(objetivo, w, parte, cfg, [])
    return _duas_etapas(sigma, cfg, resolver)


def pesos_mvsk(retornos: pd.DataFrame, media: pd.Series, sigma: CovarianciaBaixoPosto, periodos: int,
               cfg: ConfigOtimizacao) -> pd.Series:
    """min -phi1 + phi2/2 - phi3/3 + phi4/4, the CRRA weights with gamma 1, moments per period from the scenarios (Palomar ch. 9). Each
    step keeps mean and variance and linearizes skewness and kurtosis, a QP; a step that does not improve is halved."""
    centrados = retornos.to_numpy() - retornos.to_numpy().mean(axis=0)

    def resolver(indices: np.ndarray) -> np.ndarray:
        parte, cenarios, m = _parte(sigma, indices), centrados[:, indices], media.to_numpy()[indices]

        def perda(w: np.ndarray) -> float:
            carteira = cenarios @ w
            variancia = (parte.diagonal * (w @ w) + ((parte.fator @ w) ** 2).sum()) / periodos
            return float(-m @ w + variancia / 2.0 - (carteira ** 3).mean() / 3.0 + (carteira ** 4).mean() / 4.0)

        def _passo(alvo: Callable[[cp.Variable], cp.Expression]) -> np.ndarray:
            w = cp.Variable(len(indices))
            return _maximizar_com_teto(alvo(w), w, parte, cfg, [])

        uniforme = np.full(len(indices), 1.0 / len(indices))
        atual = _passo(lambda w: -cp.sum_squares(w - uniforme))     # 1/N, or the nearest portfolio within the caps
        for _ in range(PASSOS_MVSK):
            carteira = cenarios @ atual
            gradiente = (-(cenarios.T @ carteira ** 2) + (cenarios.T @ carteira ** 3)) / len(cenarios)
            proximo = _passo(lambda w: periodos * ((m - gradiente) @ w) - _risco_e_desvio(w, parte)[0] / 2.0)
            melhor = min((atual + recuo * (proximo - atual) for recuo in RECUOS_MVSK), key=perda)
            if perda(melhor) >= perda(atual) or np.abs(melhor - atual).max() <= PARADA_MVSK:
                break
            atual = melhor
        return atual
    return _duas_etapas(sigma, cfg, resolver)
