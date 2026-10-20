"""
hrp_paper.py - HRP com a distância do paper (López de Prado, 2016), fora do caminho do satélite.

O ``core.hierarchical`` clusteriza d_ij = sqrt((1 - rho_ij) / 2). O paper clusteriza d~_ij = ||D_i - D_j|| (Estágio 1,
2º passo), a distância euclidiana entre colunas da matriz D, como no Apêndice A.3 (``linkage`` recebe D quadrada e a
trata como observações). O ``core.hierarchical`` está no caminho vigiado do paper-v1 (DECISOES 68), então a opção vive
aqui e reaproveita as peças dele sem mudá-lo; com ``distancia="d"`` os pesos são os do ``core.hierarchical``.

O app continua com a d nas duas famílias (DECISOES 88): no teste pré-registrado em
``scripts/avaliar_hrp_distancia.py`` a d~ empatou com a d no HRP e perdeu no HRP-Árvore.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import leaves_list, linkage, to_tree
from scipy.spatial.distance import pdist, squareform

from core.backtest import Estrategia
from core.estimation import covariancia_ledoit_wolf
from core.hierarchical import _correlacao, _fracao_esquerda

DISTANCIAS = ("d", "d_til")


def _arvore(cov: pd.DataFrame, metodo_linkage: str, distancia: str) -> np.ndarray:
    """Linkage sobre d_ij = sqrt((1 - rho_ij) / 2) ("d") ou sobre a euclidiana entre colunas de D ("d_til")."""
    if distancia not in DISTANCIAS:
        raise ValueError(f"distancia deve ser uma de {DISTANCIAS}: {distancia!r}")
    d = np.sqrt(np.clip(0.5 * (1.0 - _correlacao(cov)), 0.0, 1.0))
    d = (d + d.T) / 2.0
    np.fill_diagonal(d, 0.0)
    condensada = pdist(d) if distancia == "d_til" else squareform(d, checks=False)
    return linkage(condensada, method=metodo_linkage)


def otimizar_hrp_paper(cov: pd.DataFrame, metodo_linkage: str = "single", distancia: str = "d_til") -> pd.Series:
    """HRP do paper: bissecção recursiva da ordem quasi-diagonal da árvore de ``distancia``."""
    tickers = cov.index.tolist()
    if len(tickers) == 1:
        return pd.Series(1.0, index=tickers)
    ordem = [tickers[i] for i in leaves_list(_arvore(cov, metodo_linkage, distancia))]
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


def otimizar_hrp_arvore_paper(cov: pd.DataFrame, metodo_linkage: str = "ward", distancia: str = "d_til") -> pd.Series:
    """HRP-Árvore (desce a árvore, como o ``core.hierarchical.otimizar_hrp_arvore``) sobre a árvore de ``distancia``."""
    tickers = cov.index.tolist()
    if len(tickers) == 1:
        return pd.Series(1.0, index=tickers)
    raiz = to_tree(_arvore(cov, metodo_linkage, distancia))
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


def estrategia_hrp_paper() -> Estrategia:
    """HRP do paper sobre a Ledoit-Wolf da janela (o ``estrategia_hrp`` do ``core.backtest`` com a d~)."""
    return lambda treino: otimizar_hrp_paper(covariancia_ledoit_wolf(treino))
