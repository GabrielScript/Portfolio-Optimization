"""
carteiras_lote2.py - The second batch of Palomar's formulations (docs/superpowers/specs/2026-11-01-lote2-livro-design.md). As
``carteiras_livros``: on the covariance's low-rank form, with the app's two-stage cardinality and caps, raising ``ErroOtimizacao`` as
``otimizar`` does. New module: the paper guards' paths stay as they are.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, replace

import cvxpy as cp
import numpy as np
import pandas as pd

from core.baixo_posto import CovarianciaBaixoPosto, _min_vol, _risco_e_desvio, densa, ledoit_wolf_baixo_posto, otimizar_baixo_posto
from core.carteiras_livros import _duas_etapas, _maximizar_com_teto, _parte, _variancias, media_por_periodo
from core.config import ConfigOtimizacao
from core.optimization import ErroInviavel, ErroOtimizacao, _limpar_pesos, _resolver, _restricoes, ajustar_aos_tetos, projetar_pesos
from core.reamostragem import indices_bootstrap_circular

PASSOS_NEWTON = 25
RECUOS = (1.0, 0.5, 0.25, 0.125)
MELHORA_MINIMA = 1e-10
PISO_DA_RIQUEZA = 1e-6      # a portfolio that lost everything in one row: its log stays finite, and very negative
EXPOENTE_DO_SUBCONJUNTO = 0.7
AVERSAO_DO_SUBCONJUNTO = 2.0
PASSOS_TILTING = 20
PARADA_TILTING = 1e-6
FOLGA_HERFINDAHL = 1.01     # the book's kappa = 0.01 xi(w0)
SENTIDOS = np.array([1.0, -1.0, 1.0, -1.0])     # mean and skewness up, variance and kurtosis down
ESCALA_MINIMA = 1e-12


@dataclass(frozen=True)
class Utilidade:
    """A concave utility of the wealth 1 + x and its two derivatives."""

    valor: Callable[[np.ndarray], np.ndarray]
    primeira: Callable[[np.ndarray], np.ndarray]
    segunda: Callable[[np.ndarray], np.ndarray]


LOG = Utilidade(np.log, lambda y: 1.0 / y, lambda y: -1.0 / y ** 2)
RAIZ = Utilidade(np.sqrt, lambda y: 0.5 / np.sqrt(y), lambda y: -0.25 / y ** 1.5)


def pesos_utilidade(retornos: pd.DataFrame, sigma: CovarianciaBaixoPosto, cfg: ConfigOtimizacao, utilidade: Utilidade) -> pd.Series:
    """max (1/T) sum U(1 + w'r_t) on the training rows (Palomar 7.3): Newton steps on the sample objective, each a QP under the caps,
    halved if it does not improve. Every step stays feasible, because the feasible set is convex and the start is in it."""
    cenarios = retornos.to_numpy()

    def resolver(indices: np.ndarray) -> np.ndarray:
        parte, linhas = _parte(sigma, indices), cenarios[:, indices]

        def riqueza(w: np.ndarray) -> np.ndarray:
            return np.maximum(1.0 + linhas @ w, PISO_DA_RIQUEZA)

        def media(w: np.ndarray) -> float:
            return float(utilidade.valor(riqueza(w)).mean())

        def _passo(alvo: Callable[[cp.Variable], cp.Expression]) -> np.ndarray:
            w = cp.Variable(len(indices))
            return _maximizar_com_teto(alvo(w), w, parte, cfg, [])

        uniforme = np.full(len(indices), 1.0 / len(indices))
        atual = _passo(lambda w: -cp.sum_squares(w - uniforme))     # 1/N, or the nearest portfolio within the caps
        for _ in range(PASSOS_NEWTON):
            y = riqueza(atual)
            gradiente = linhas.T @ utilidade.primeira(y) / len(linhas)
            curvatura = np.sqrt(-utilidade.segunda(y) / len(linhas))
            proximo = _passo(lambda w: gradiente @ (w - atual) - 0.5 * cp.sum_squares(cp.multiply(curvatura, linhas @ (w - atual))))
            melhor = max((atual + recuo * (proximo - atual) for recuo in RECUOS), key=media)
            if media(melhor) <= media(atual) + MELHORA_MINIMA:
                break
            atual = melhor
        return atual
    return _duas_etapas(sigma, cfg, resolver)


def pesos_downside(retornos: pd.DataFrame, sigma: CovarianciaBaixoPosto, cfg: ConfigOtimizacao, alfa: int) -> pd.Series:
    """max w'mu - (1/T) sum ((0 - w'r_t)+)^alfa, both per period, on the training rows' raw mean (Palomar 10.3.1, with tau = 0)."""
    cenarios = retornos.to_numpy()
    media = cenarios.mean(axis=0)
    linhas = len(cenarios)

    def resolver(indices: np.ndarray) -> np.ndarray:
        w, folga = cp.Variable(len(indices)), cp.Variable(linhas, nonneg=True)
        objetivo = media[indices] @ w - cp.sum(cp.power(folga, alfa)) / linhas
        return _maximizar_com_teto(objetivo, w, _parte(sigma, indices), cfg, [folga >= -cenarios[:, indices] @ w])
    return _duas_etapas(sigma, cfg, resolver)


def pesos_quintil(media: pd.Series, sigma: CovarianciaBaixoPosto, cfg: ConfigOtimizacao, ranking: str) -> pd.Series:
    """1/k on the top k = ceil(N/5) by mu, mu/sigma or mu/sigma^2 (Palomar 6.4.4), cut to the caps."""
    variancias = pd.Series(_variancias(sigma), index=sigma.nomes)
    nota = {"mu": media, "mu_sigma": media / np.sqrt(variancias), "mu_sigma2": media / variancias}[ranking]
    k = math.ceil(len(nota) / 5)
    pesos = pd.Series(0.0, index=nota.index)
    pesos[nota.sort_values(ascending=False, kind="stable").index[:k]] = 1.0 / k
    return ajustar_aos_tetos(pesos, cfg)


def pesos_box(mu: pd.Series, sigma: CovarianciaBaixoPosto, cfg: ConfigOtimizacao, anos: float) -> pd.Series:
    """The worst case of mu in a box of half-width sigma_i / sqrt(T) (Palomar 14.2.2). Annual inputs, so T is in years."""
    return otimizar_baixo_posto(mu - np.sqrt(_variancias(sigma)) / np.sqrt(anos), sigma, cfg)


def pesos_retorno_minimo(mu: pd.Series, sigma: CovarianciaBaixoPosto, cfg: ConfigOtimizacao) -> pd.Series:
    """min w'Sigma w with w'mu >= the eligible names' mean of mu, 1/N's (Palomar 7.1). If no portfolio meets it, the minimum variance:
    at each stage, so on the 10 names kept the floor may give way."""
    beta = float(mu.mean())
    medias = mu.to_numpy()

    def resolver(indices: np.ndarray) -> np.ndarray:
        parte = _parte(sigma, indices)
        w = cp.Variable(len(indices))
        risco, desvio = _risco_e_desvio(w, parte)
        restricoes = [*_restricoes(w, parte.nomes, cfg), medias[indices] @ w >= beta]
        if cfg.vol_maxima is not None:
            restricoes.append(desvio <= cfg.vol_maxima)
        try:
            return _resolver(cp.Problem(cp.Minimize(risco), restricoes), w)[0]
        except ErroInviavel:
            return _min_vol(parte, cfg)
    return _duas_etapas(sigma, cfg, resolver)


@dataclass(frozen=True)
class ConfigSubconjunto:
    """``bloco``: sessions of the circular bootstrap of the rows, which makes it subset bagging; None keeps the rows."""

    reamostras: int = 100
    bloco: int | None = None
    semente: int = 0


def pesos_subconjunto(retornos: pd.DataFrame, periodos: int, cfg: ConfigOtimizacao, config: ConfigSubconjunto = ConfigSubconjunto()) -> pd.Series:
    """Kelly lambda 2 on ceil(N^0.7) random names (Palomar 14.3.2), no cardinality inside, zeros filled, averaged, projected on the caps
    and the cardinality as ``otimizar_reamostrado``. A draw that fails is dropped: a random subset may fit no portfolio under the sector caps."""
    colunas = retornos.shape[1]
    tamanho = math.ceil(colunas ** EXPOENTE_DO_SUBCONJUNTO)
    cfg_do_sorteio = replace(cfg, aversao_risco=AVERSAO_DO_SUBCONJUNTO, n_ativos_max=None)
    gerador = np.random.default_rng(config.semente)
    soma, validas = pd.Series(0.0, index=retornos.columns), 0
    for _ in range(config.reamostras):
        nomes = retornos.columns[np.sort(gerador.choice(colunas, tamanho, replace=False))]
        amostra = retornos[nomes]
        if config.bloco is not None:
            amostra = amostra.iloc[indices_bootstrap_circular(len(retornos), config.bloco, gerador)]
        try:
            sigma = ledoit_wolf_baixo_posto(amostra, periodos)
            soma[nomes] += otimizar_baixo_posto(media_por_periodo(amostra, sigma, periodos) * periodos, sigma, cfg_do_sorteio).to_numpy()
        except (ErroOtimizacao, ValueError, np.linalg.LinAlgError):
            continue
        validas += 1
    if validas == 0:
        raise ErroOtimizacao(f"All {config.reamostras} subsets failed.")
    return projetar_pesos(soma / validas, densa(ledoit_wolf_baixo_posto(retornos, periodos)), cfg).pesos


def pesos_tilting(retornos: pd.DataFrame, referencia: pd.Series, media: pd.Series, sigma: CovarianciaBaixoPosto, periodos: int,
                  cfg: ConfigOtimizacao) -> pd.Series:
    """Palomar 9.3.3 from ``referencia``: max delta with each of the four moments moved by delta x |its value at the reference|, and a
    Herfindahl within 1% of the reference's. Each step linearizes skewness and kurtosis and is halved if the true delta does not rise;
    delta 0 at the reference is always feasible, so a failed step keeps what it has. No cardinality: all names may enter."""
    inicial = referencia.reindex(retornos.columns).fillna(0.0).to_numpy()
    inicial = inicial / inicial.sum()
    centrados = retornos.to_numpy() - retornos.to_numpy().mean(axis=0)
    medias = media.to_numpy()

    def momentos(w: np.ndarray) -> np.ndarray:
        carteira = centrados @ w
        return np.array([medias @ w, sigma.diagonal * (w @ w) + ((sigma.fator @ w) ** 2).sum(), (carteira ** 3).mean(), (carteira ** 4).mean()])

    base = momentos(inicial)
    escala = np.maximum(np.abs(base), ESCALA_MINIMA)
    herfindahl = FOLGA_HERFINDAHL * float(inicial @ inicial)

    def melhora(w: np.ndarray) -> float:
        return float(((momentos(w) - base) * SENTIDOS / escala).min())

    atual = inicial
    for _ in range(PASSOS_TILTING):
        carteira = centrados @ atual
        no_atual = momentos(atual)
        terceiro = 3.0 * (centrados.T @ carteira ** 2) / len(centrados)
        quarto = 4.0 * (centrados.T @ carteira ** 3) / len(centrados)
        w, delta = cp.Variable(len(atual)), cp.Variable()
        risco, desvio = _risco_e_desvio(w, sigma)
        restricoes = [*_restricoes(w, sigma.nomes, cfg), cp.sum_squares(w) <= herfindahl,
                      (medias @ w - base[0]) / escala[0] >= delta,
                      (base[1] - risco) / escala[1] >= delta,
                      (no_atual[2] + terceiro @ (w - atual) - base[2]) / escala[2] >= delta,
                      (base[3] - no_atual[3] - quarto @ (w - atual)) / escala[3] >= delta]
        if cfg.vol_maxima is not None:
            restricoes.append(desvio <= cfg.vol_maxima)
        try:
            proximo = _resolver(cp.Problem(cp.Maximize(delta), restricoes), w)[0]
        except ErroOtimizacao:
            break
        melhor = max((atual + recuo * (proximo - atual) for recuo in RECUOS), key=melhora)
        if melhora(melhor) <= melhora(atual) + PARADA_TILTING:
            break
        atual = melhor
    if atual is inicial:
        return pd.Series(inicial, index=retornos.columns)
    return ajustar_aos_tetos(pd.Series(_limpar_pesos(atual), index=retornos.columns), cfg)
