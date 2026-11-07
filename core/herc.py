"""
herc.py - Raffinot's (2018) Hierarchical Equal Risk Contribution with CDaR, as skfolio 1.12.2's
``HierarchicalEqualRiskContribution(risk_measure=CDAR)`` computes it (docs/superpowers/specs/2026-11-07-herc-design.md
and its addendum; skfolio is the test-only oracle, DECISOES 168): Pearson distance, Ward, clusters by the two-order
difference to gap statistic, inverse-CDaR weights inside each cluster, the capital across by the clusters' CDaR down
the dendrogram. Then the fixed caps and the Composite's check, for the two arms of the reading.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd
import scipy.cluster.hierarchy as sch
import scipy.spatial.distance as scd

from core.backtest import CarteiraRecusada, Estrategia
from core.estimation import estimar_momentos
from core.risco import LIMITES, checar_carteira
from core.sinais import NOMES, POR_SETOR, carteira_da_nota
from core.valor import NotaDoMes

BETA = 0.95


def _cvar(valores: np.ndarray, beta: float = BETA) -> float:
    """skfolio's ``cvar`` with equal weights: minus the mean of the worst (1 - beta) of ``valores``."""
    ordenados = np.sort(valores)
    probs = np.full(len(ordenados), 1.0 / len(ordenados))
    acumulado = np.cumsum(probs)
    cauda = (1.0 - beta) * acumulado[-1]
    i = int(np.searchsorted(acumulado, cauda))
    if i == 0:
        return float(-ordenados[0])
    return float(-(probs[:i] @ ordenados[:i] + ordenados[i] * (cauda - acumulado[i - 1])) / cauda)


def cdar(retornos: np.ndarray, beta: float = BETA) -> float:
    """CDaR of uncompounded returns: the drawdowns from a start at 0, then their CVaR."""
    acumulado = np.cumsum(retornos)
    pico = np.maximum.accumulate(np.concatenate([[0.0], acumulado]))[1:]
    return _cvar(acumulado - pico, beta)


def _numero_de_grupos(distancia: np.ndarray, ligacoes: np.ndarray) -> int:
    """skfolio's ``compute_optimal_n_clusters`` (two-order difference to gap statistic)."""
    corte = sch.cut_tree(ligacoes)
    n = corte.shape[1]
    maximo = min(n, max(8, round(np.sqrt(n))))
    dispersao = []
    for k in range(maximo):
        nivel = corte[:, n - k - 1]
        densidades = []
        for grupo in range(np.max(nivel) + 1):
            membros = np.argwhere(nivel == grupo).flatten()
            distancias = scd.squareform(distancia[membros, :][:, membros], checks=False)
            if distancias.shape[0] != 0:
                densidades.append(np.nan_to_num(distancias.mean()))
        dispersao.append(np.sum(densidades))
    dispersao = np.array(dispersao)
    gaps = (np.roll(dispersao, -2) + dispersao - 2 * np.roll(dispersao, -1))[:-2]
    return int(np.argmax(gaps) + 2)


def pesos_herc_cdar(retornos: pd.DataFrame) -> pd.Series:
    """HERC-CDaR weights of the columns of ``retornos`` (daily simple returns, no NaN), before any cap."""
    x = retornos.to_numpy(dtype=float)
    correlacao = np.corrcoef(x.T)
    distancia = np.sqrt(np.clip(0.5 * (1.0 - correlacao), 0.0, None))
    np.fill_diagonal(distancia, 0.0)
    ligacoes = sch.linkage(scd.squareform(distancia, checks=False), method="ward", optimal_ordering=False)
    rotulos = sch.fcluster(ligacoes, t=_numero_de_grupos(distancia, ligacoes), criterion="maxclust") - 1
    n = x.shape[1]
    riscos = np.array([cdar(x[:, j]) for j in range(n)])
    pesos = np.ones(n)
    grupos = [np.argwhere(rotulos == g).flatten() for g in range(rotulos.max() + 1)]
    conjuntos = [set(grupo) for grupo in grupos]
    riscos_dos_grupos = []
    for grupo in grupos:
        inverso = np.zeros(n)
        inverso[grupo] = 1.0 / riscos[grupo]
        inverso /= inverso.sum()
        riscos_dos_grupos.append(cdar(x @ inverso))
        pesos[grupo] = inverso[grupo]
    riscos_dos_grupos = np.array(riscos_dos_grupos)
    pesos_dos_grupos = np.ones(len(grupos))
    pilha = [sch.to_tree(ligacoes)]
    while pilha:
        no = pilha.pop()
        if set(no.pre_order()) in conjuntos:
            continue
        esquerda, direita = set(no.get_left().pre_order()), set(no.get_right().pre_order())
        lado_e = np.array([i for i, c in enumerate(conjuntos) if c <= esquerda])
        lado_d = np.array([i for i, c in enumerate(conjuntos) if c <= direita])
        risco_e, risco_d = riscos_dos_grupos[lado_e].sum(), riscos_dos_grupos[lado_d].sum()
        alfa = 1.0 - risco_e / (risco_e + risco_d)
        pesos_dos_grupos[lado_e] *= alfa
        pesos_dos_grupos[lado_d] *= 1.0 - alfa
        pilha += [no.get_left(), no.get_right()]
    for g, grupo in enumerate(grupos):
        pesos[grupo] *= pesos_dos_grupos[g]
    return pd.Series(pesos / pesos.sum(), index=retornos.columns)


def aplicar_tetos(pesos: pd.Series, setores: Mapping[str, str], nome: float = LIMITES.peso_maximo_acao,
                  setor: float = LIMITES.teto_setor, rodadas: int = 200) -> pd.Series:
    """Cut each name at ``nome`` and each sector at ``setor``, spreading each cut over the names with room in proportion
    to their weights, until both hold."""
    w = pesos.astype(float) / pesos.sum()
    grupo = pd.Series({papel: setores.get(papel, papel) for papel in w.index})
    for _ in range(rodadas):
        por_setor = w.groupby(grupo).sum()
        acima_setor = por_setor[por_setor > setor + 1e-12]
        if w.max() <= nome + 1e-12 and acima_setor.empty:
            break
        for s, total in acima_setor.items():
            w[grupo == s] *= setor / total
        w = w.clip(upper=nome)
        sobra = 1.0 - w.sum()
        por_setor = w.groupby(grupo).sum()
        livres = (w < nome - 1e-12) & grupo.map(por_setor < setor - 1e-12)
        if sobra > 1e-12 and livres.any():
            w[livres] += sobra * w[livres] / w[livres].sum()
    return w


def _checado(pesos: pd.Series, setores: Mapping[str, str], treino: pd.DataFrame) -> pd.Series:
    violacoes = checar_carteira(pesos, setores, estimar_momentos(treino[pesos.index], shrink_mu=False).cov, LIMITES)
    if violacoes:
        raise CarteiraRecusada(violacoes)
    return pesos


def estrategia_herc_cdar(setores: Mapping[str, str]) -> Estrategia:
    """``HERC-CDaR``: every eligible name with a sector."""
    def _estrategia(treino: pd.DataFrame) -> pd.Series:
        retornos = treino[[papel for papel in treino.columns if setores.get(papel)]]
        return _checado(aplicar_tetos(pesos_herc_cdar(retornos), setores), setores, retornos)
    return _estrategia


def estrategia_herc_cdar_15(nota: NotaDoMes, setores: Mapping[str, str]) -> Estrategia:
    """``HERC-CDaR-15``: the Composite's picks of the decision, weighted by HERC-CDaR."""
    def _estrategia(treino: pd.DataFrame) -> pd.Series:
        mes = treino.index[-1].to_period("M").to_timestamp()
        escolhas = carteira_da_nota(nota(mes, list(treino.columns)).dropna(), setores, NOMES, POR_SETOR)
        if escolhas.empty:
            raise ValueError(f"{treino.index[-1]:%Y-%m-%d}: no eligible name with a score")
        retornos = treino[list(escolhas.index)]
        return _checado(aplicar_tetos(pesos_herc_cdar(retornos), setores), setores, retornos)
    return _estrategia
