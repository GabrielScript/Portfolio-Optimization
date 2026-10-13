"""
hrp.py - Hierarchical Risk Parity (HRP) e Hierarchical Equal Risk Contribution (HERC)
App: Otimizador Growth Agressivo B3
Autor: Gabriel Estrela Lopes

Benchmarks hierárquicos robustos (López de Prado, 2016; Raffinot, 2018):

HRP  - clusteriza ativos por distância de correlação, quasi-diagonaliza a
       covariância e aloca por bissecção recursiva com inverso-variância.
       Não inverte matriz nem estima mu -> muito robusto out-of-sample.

HERC - variante que aloca risco IGUAL entre clusters (equal risk contribution
       no nível dos clusters), inverso-variância dentro de cada cluster.

Papel no app: benchmarks honestos contra a carteira growth otimizada.
São métodos risk-based (ignoram retorno esperado) — se a carteira agressiva
não os bate em Calmar/Sortino, é sinal de alerta. Rodá-los sobre o universo
JÁ filtrado por growth score dá o híbrido "seleção agressiva + alocação robusta".

Usa apenas scipy (dependência existente).
"""

import logging
from typing import Dict, List

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, leaves_list, to_tree
from scipy.spatial.distance import squareform

logger = logging.getLogger(__name__)


def _matriz_correlacao(matriz_cov: pd.DataFrame) -> pd.DataFrame:
    """Converte covariância em correlação."""
    sigma = np.sqrt(np.diag(matriz_cov.values))
    corr = matriz_cov.values / np.outer(sigma, sigma)
    corr = np.clip(corr, -1.0, 1.0)
    return pd.DataFrame(corr, index=matriz_cov.index, columns=matriz_cov.columns)


def _distancia_correlacao(corr: pd.DataFrame) -> np.ndarray:
    """
    Distância de López de Prado: d_ij = sqrt(0.5 * (1 - rho_ij)).
    Retorna forma condensada (exigida pelo linkage do scipy).
    """
    dist = np.sqrt(0.5 * (1.0 - corr.values))
    np.fill_diagonal(dist, 0.0)
    # simetriza contra ruído de ponto flutuante antes do squareform
    dist = (dist + dist.T) / 2.0
    return squareform(dist, checks=False)


def _variancia_cluster(matriz_cov: pd.DataFrame, membros: List[str]) -> float:
    """
    Variância do cluster usando pesos inverso-variância internos
    (definição padrão do HRP, López de Prado cap. 16).
    """
    cov = matriz_cov.loc[membros, membros].values
    ivp = 1.0 / np.diag(cov)
    ivp /= ivp.sum()
    return float(ivp @ cov @ ivp)


def otimizar_hrp(matriz_cov: pd.DataFrame,
                 metodo_linkage: str = "single") -> Dict[str, float]:
    """
    Hierarchical Risk Parity (López de Prado, 2016).

    Etapas:
    1. Árvore hierárquica sobre a distância de correlação
    2. Quasi-diagonalização (ordena ativos pela árvore -> similares vizinhos)
    3. Bissecção recursiva: divide a lista ordenada ao meio e aloca capital
       entre as metades inversamente proporcional à variância de cada metade

    Args:
        matriz_cov: covariância anualizada (DataFrame ticker x ticker)
        metodo_linkage: 'single' (original LdP), 'ward' ou 'average'

    Returns:
        Dict {ticker: peso}, pesos somam 1, todos >= 0.
    """
    tickers = matriz_cov.index.tolist()
    if len(tickers) == 1:
        return {tickers[0]: 1.0}

    corr = _matriz_correlacao(matriz_cov)
    dist_condensada = _distancia_correlacao(corr)
    arvore = linkage(dist_condensada, method=metodo_linkage)

    # Quasi-diagonalização: ordem das folhas da árvore
    ordem = [tickers[i] for i in leaves_list(arvore)]

    # Bissecção recursiva
    pesos = pd.Series(1.0, index=ordem)
    grupos = [ordem]
    while grupos:
        novos_grupos = []
        for grupo in grupos:
            if len(grupo) <= 1:
                continue
            meio = len(grupo) // 2
            esquerda, direita = grupo[:meio], grupo[meio:]

            var_esq = _variancia_cluster(matriz_cov, esquerda)
            var_dir = _variancia_cluster(matriz_cov, direita)

            # alocação inverso-variância entre as duas metades
            alpha = 1.0 - var_esq / (var_esq + var_dir)
            pesos[esquerda] *= alpha
            pesos[direita] *= (1.0 - alpha)

            novos_grupos.extend([esquerda, direita])
        grupos = novos_grupos

    pesos /= pesos.sum()
    return pesos.to_dict()


def _coletar_folhas(no) -> List[int]:
    """Índices das folhas sob um nó da árvore do scipy."""
    if no.is_leaf():
        return [no.id]
    return _coletar_folhas(no.get_left()) + _coletar_folhas(no.get_right())


def otimizar_herc(matriz_cov: pd.DataFrame,
                  metodo_linkage: str = "ward") -> Dict[str, float]:
    """
    Hierarchical Equal Risk Contribution (Raffinot, 2018).

    Diferença para o HRP: em vez de bissecção pela ORDEM quasi-diagonal,
    desce a ÁRVORE de clusterização de verdade. Em cada nó, divide o capital
    entre os dois sub-clusters de forma inversamente proporcional ao risco
    (variância) de cada um -> contribuição de risco igualada nó a nó.
    Dentro do cluster final, alocação inverso-variância.

    Args:
        matriz_cov: covariância anualizada (DataFrame ticker x ticker)
        metodo_linkage: 'ward' (recomendado p/ HERC), 'single' ou 'average'

    Returns:
        Dict {ticker: peso}, pesos somam 1, todos >= 0.
    """
    tickers = matriz_cov.index.tolist()
    if len(tickers) == 1:
        return {tickers[0]: 1.0}

    corr = _matriz_correlacao(matriz_cov)
    dist_condensada = _distancia_correlacao(corr)
    arvore = linkage(dist_condensada, method=metodo_linkage)
    raiz = to_tree(arvore)

    pesos = pd.Series(0.0, index=tickers)

    def _alocar(no, capital: float):
        folhas = [tickers[i] for i in _coletar_folhas(no)]
        if no.is_leaf() or len(folhas) == 1:
            pesos[folhas[0]] += capital
            return
        if len(folhas) == 2:
            # nó terminal com 2 ativos: inverso-variância direto
            cov = matriz_cov.loc[folhas, folhas]
            ivp = 1.0 / np.diag(cov.values)
            ivp /= ivp.sum()
            for t, w in zip(folhas, ivp):
                pesos[t] += capital * w
            return

        esq, dir_ = no.get_left(), no.get_right()
        var_esq = _variancia_cluster(matriz_cov, [tickers[i] for i in _coletar_folhas(esq)])
        var_dir = _variancia_cluster(matriz_cov, [tickers[i] for i in _coletar_folhas(dir_)])

        # equal risk contribution entre os dois clusters:
        # peso proporcional ao inverso do risco
        alpha = (1.0 / var_esq) / (1.0 / var_esq + 1.0 / var_dir)
        _alocar(esq, capital * alpha)
        _alocar(dir_, capital * (1.0 - alpha))

    _alocar(raiz, 1.0)
    pesos /= pesos.sum()
    return pesos.to_dict()
