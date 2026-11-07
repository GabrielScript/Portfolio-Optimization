"""
periferia.py - The periphery tilt of the Composite (docs/superpowers/specs/2026-11-07-periferia-design.md): at each
decision, the minimum spanning tree of the last year's correlations among the eligible names, Pozzi, Di Matteo and
Aste's (2013) hybrid centrality on it, and a score that averages the Composite's percentile with the periphery's.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd
from scipy.sparse.csgraph import minimum_spanning_tree, shortest_path

from core.backtest import Estrategia
from core.composto import estrategia_por_nota
from core.valor import NotaDoMes

JANELA = 252                    # sessions of daily returns behind the correlations
PESO_DA_PERIFERIA = 0.5
_SEPARA = 1e-12                 # keeps a perfect correlation (distance 0) an edge for scipy, which reads 0 as none


def centralidades_na_arvore(saltos: np.ndarray) -> pd.DataFrame:
    """``saltos``: hop distances on a tree (n x n). Degree, betweenness, closeness, eccentricity, eigenvector."""
    n = len(saltos)
    vizinhos = saltos == 1.0
    grau = vizinhos.sum(axis=1)
    pares = (n - 1) * (n - 2) / 2
    intermediacao = np.empty(n)
    for no in range(n):
        # removing a tree node leaves one component per neighbour: the nodes nearer that neighbour than the node
        tamanhos = [(saltos[:, vizinho] < saltos[:, no]).sum() for vizinho in np.flatnonzero(vizinhos[no])]
        intermediacao[no] = pares - sum(t * (t - 1) / 2 for t in tamanhos)
    proximidade = (n - 1) / saltos.sum(axis=1)
    excentricidade = saltos.max(axis=1)
    _, vetores = np.linalg.eigh(vizinhos.astype(float))
    autovetor = np.abs(vetores[:, -1])
    return pd.DataFrame({"grau": grau, "intermediacao": intermediacao, "proximidade": proximidade,
                         "excentricidade": excentricidade, "autovetor": autovetor})


def periferia_da_janela(retornos: pd.DataFrame) -> pd.Series:
    """Percentile of periphery (1 = most peripheral) of each column, from the last ``JANELA`` sessions only."""
    recorte = retornos.iloc[-JANELA:]
    correlacao = np.clip(recorte.corr().to_numpy(), -1.0, 1.0)
    distancia = np.sqrt(2.0 * (1.0 - correlacao)) + _SEPARA
    np.fill_diagonal(distancia, 0.0)
    arvore = minimum_spanning_tree(distancia)
    saltos = shortest_path(arvore, directed=False, unweighted=True)
    medidas = centralidades_na_arvore(saltos).assign(excentricidade=lambda m: -m["excentricidade"])
    centralidade = medidas.rank(pct=True).mean(axis=1)
    return pd.Series((-centralidade).rank(pct=True).to_numpy(), index=retornos.columns, name="periferia")


def estrategia_periferia(nota: NotaDoMes, setores: Mapping[str, str]) -> Estrategia:
    """The Composite rule on 0.5 x the Composite score's percentile + 0.5 x periphery's, among the names with a score."""
    def _estrategia(treino: pd.DataFrame) -> pd.Series:
        periferia = periferia_da_janela(treino)

        def _inclinada(mes: pd.Timestamp, candidatos: list[str]) -> pd.Series:
            composto = nota(mes, candidatos)
            com_nota = composto.dropna().index
            inclinada = ((1.0 - PESO_DA_PERIFERIA) * composto[com_nota].rank(pct=True)
                         + PESO_DA_PERIFERIA * periferia[com_nota].rank(pct=True))
            return inclinada.reindex(composto.index)
        return estrategia_por_nota(_inclinada, setores)(treino)
    return _estrategia
