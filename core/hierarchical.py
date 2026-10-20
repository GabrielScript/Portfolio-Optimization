"""
hierarchical.py - Alocações hierárquicas risk-based (ignoram mu).

HRP (López de Prado, 2016): clusterização por distância de correlação,
quasi-diagonalização e bissecção recursiva com inverso-variância. Não inverte
Sigma nem estima mu, por isso é robusta fora da amostra.

HRP-Árvore: em vez de bissectar a ORDEM quasi-diagonal ao meio, desce a
própria árvore do linkage e divide o capital em cada nó por inverso da
variância dos dois ramos. É a ideia do HERC de Raffinot (2018), mas sem o
corte ótimo do número de clusters (gap statistic) nem a medida de risco
alternativa. Por isso NÃO se chama HERC aqui.

Propriedade (verificada): a cada divisão, alpha = V2/(V1+V2) está em (0,1) e
equivale à divisão inverso-variância (1/V1)/(1/V1+1/V2). Cada peso final é
produto de fatores em (0,1), logo é positivo, e a soma dos pesos se preserva
em 1 sem precisar normalizar.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import leaves_list, linkage, to_tree
from scipy.spatial.distance import squareform

_VARIANCIA_MINIMA = 1e-12


def _correlacao(cov: pd.DataFrame) -> np.ndarray:
    sigma = np.sqrt(np.maximum(np.diag(cov.values), _VARIANCIA_MINIMA))
    return np.clip(cov.values / np.outer(sigma, sigma), -1.0, 1.0)


def _arvore(cov: pd.DataFrame, metodo_linkage: str) -> np.ndarray:
    """Linkage sobre a distância de López de Prado d_ij = sqrt((1 - rho_ij) / 2)."""
    distancia = np.sqrt(np.clip(0.5 * (1.0 - _correlacao(cov)), 0.0, 1.0))
    distancia = (distancia + distancia.T) / 2.0
    np.fill_diagonal(distancia, 0.0)
    return linkage(squareform(distancia, checks=False), method=metodo_linkage)


def _variancia_cluster(cov: pd.DataFrame, membros: list[str]) -> float:
    """Variância do cluster sob pesos inverso-variância internos."""
    sub = cov.loc[membros, membros].values
    ivp = 1.0 / np.maximum(np.diag(sub), _VARIANCIA_MINIMA)
    ivp /= ivp.sum()
    return max(float(ivp @ sub @ ivp), _VARIANCIA_MINIMA)


def _fracao_esquerda(cov: pd.DataFrame, esquerda: list[str], direita: list[str]) -> float:
    var_esq = _variancia_cluster(cov, esquerda)
    var_dir = _variancia_cluster(cov, direita)
    return var_dir / (var_esq + var_dir)


def otimizar_hrp(cov: pd.DataFrame, metodo_linkage: str = "single") -> pd.Series:
    """HRP original: bissecção recursiva da ordem quasi-diagonal."""
    tickers = cov.index.tolist()
    if len(tickers) == 1:
        return pd.Series(1.0, index=tickers)
    ordem = [tickers[i] for i in leaves_list(_arvore(cov, metodo_linkage))]
    pesos = pd.Series(1.0, index=ordem)
    grupos = [ordem]
    while grupos:
        proximos = []
        for grupo in grupos:
            if len(grupo) < 2:
                continue
            meio = len(grupo) // 2
            esquerda, direita = grupo[:meio], grupo[meio:]
            alpha = _fracao_esquerda(cov, esquerda, direita)
            pesos[esquerda] *= alpha
            pesos[direita] *= 1.0 - alpha
            proximos.extend([esquerda, direita])
        grupos = proximos
    return pesos.reindex(tickers)


def otimizar_hrp_arvore(cov: pd.DataFrame, metodo_linkage: str = "ward") -> pd.Series:
    """Variante que desce a árvore de clusterização (inspirada no HERC)."""
    tickers = cov.index.tolist()
    if len(tickers) == 1:
        return pd.Series(1.0, index=tickers)
    raiz = to_tree(_arvore(cov, metodo_linkage))
    pesos = pd.Series(0.0, index=tickers)
    pilha = [(raiz, 1.0)]
    while pilha:
        no, capital = pilha.pop()
        if no.is_leaf():
            pesos.iloc[no.id] += capital
            continue
        esquerda = [tickers[i] for i in no.get_left().pre_order()]
        direita = [tickers[i] for i in no.get_right().pre_order()]
        alpha = _fracao_esquerda(cov, esquerda, direita)
        pilha.append((no.get_left(), capital * alpha))
        pilha.append((no.get_right(), capital * (1.0 - alpha)))
    return pesos
