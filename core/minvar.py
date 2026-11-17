"""
minvar.py - Minimum variance inside the house rules (spec ``docs/superpowers/specs/2026-11-16-minima-variancia-design.md``).
Research only: the paper logs and the screen do not read this.

    min w'Sigma w   s.t.  1'w = 1,  0 <= w_i <= u,  each sector <= c,  at most k names,  [p'w >= floor]

``p`` is each name's score percentile (``percentil_da_nota``): the optional floor asks for a weighted mean percentile,
so the whole eligible universe stays in the problem and the score only tilts it.

The count uses ``core.optimization``'s two-step heuristic (solve relaxed, keep the k largest weights, solve again on
them); when the k largest cannot hold 100% inside the caps, the relaxed weights stay (more than k names).
"""

from __future__ import annotations

from collections.abc import Mapping

import cvxpy as cp
import numpy as np
import pandas as pd

from core.backtest import CarteiraRecusada, Estrategia, so_com_setor
from core.config import DIAS_UTEIS_ANO, ConfigOtimizacao, Objetivo, limites_por_setor
from core.estimation import covariancia_ledoit_wolf
from core.optimization import (ErroInviavel, _limpar_pesos, _resolver, _restricoes, ajustar_aos_tetos,
                               preparar_covariancia)
from core.risco import LIMITES, Limites, checar_carteira
from core.valor import NotaDoMes

NOMES = 15                        # the Composite's count
PISO = 0.70                       # MV-nota's floor on the weighted mean score percentile (fixed, not tuned)


def percentil_da_nota(nota: pd.Series, nomes: pd.Index) -> pd.Series:
    """Each name's rank among the scored names, in (0, 1]; 0 for a name without a score."""
    return nota.dropna().rank(pct=True).reindex(nomes).fillna(0.0)


def _resolver_minima(cov: np.ndarray, tickers: list[str], cfg: ConfigOtimizacao, percentil: np.ndarray | None,
                     piso: float) -> np.ndarray:
    w = cp.Variable(len(tickers))
    restricoes = _restricoes(w, tickers, cfg)
    if percentil is not None:
        restricoes.append(percentil @ w >= piso)
    return _limpar_pesos(_resolver(cp.Problem(cp.Minimize(cp.quad_form(w, cp.psd_wrap(cov))), restricoes), w)[0])


def minima_variancia(cov: pd.DataFrame, cfg: ConfigOtimizacao, percentil: pd.Series | None = None,
                     piso: float = PISO) -> pd.Series:
    """The minimum-variance weights of ``cov`` under ``cfg``'s caps (per name, per group) and count; with
    ``percentil`` (indexed as ``cov``), also ``percentil'w >= piso``. ``ErroInviavel`` when no portfolio fits."""
    tickers = list(cov.index)
    matriz = preparar_covariancia(cov)
    p = None if percentil is None else percentil.reindex(cov.index).to_numpy(dtype=float)
    pesos = _resolver_minima(matriz, tickers, cfg, p, piso)
    k = cfg.n_ativos_max
    if k is not None and int((pesos > 0).sum()) > k:
        maiores = np.sort(np.argsort(-pesos, kind="stable")[:k])
        try:
            dos_maiores = _resolver_minima(matriz[np.ix_(maiores, maiores)], [tickers[i] for i in maiores], cfg,
                                           None if p is None else p[maiores], piso)
        except ErroInviavel:
            pass
        else:
            pesos = np.zeros(len(tickers))
            pesos[maiores] = dos_maiores
    return ajustar_aos_tetos(pd.Series(pesos, index=cov.index), cfg)


def estrategia_minima_variancia(setores: Mapping[str, str], nota: NotaDoMes | None = None, piso: float = PISO,
                                nomes: int = NOMES, limites: Limites = LIMITES) -> Estrategia:
    """At each decision: the names with a sector, the training window's annual Ledoit-Wolf covariance, minimum
    variance under the fixed rules' caps; a violation of ``checar_carteira`` is ``CarteiraRecusada``. With ``nota``
    (the month's score), the weighted mean of its percentile among the decision's names is at least ``piso``."""
    def _estrategia(treino: pd.DataFrame) -> pd.Series:
        treino = so_com_setor(treino, setores)
        cov = covariancia_ledoit_wolf(treino) * DIAS_UTEIS_ANO
        cfg = ConfigOtimizacao(objetivo=Objetivo.MIN_VOL, peso_maximo=limites.peso_maximo_acao, n_ativos_max=nomes,
                               vol_maxima=limites.vol_maxima,
                               grupos=limites_por_setor({papel: setores[papel] for papel in treino.columns},
                                                        limites.teto_setor))
        percentil = None
        if nota is not None:
            mes = treino.index[-1].to_period("M").to_timestamp()
            percentil = percentil_da_nota(nota(mes, list(treino.columns)), treino.columns)
        pesos = minima_variancia(cov, cfg, percentil, piso)
        violacoes = checar_carteira(pesos, setores, cov, limites)
        if violacoes:
            raise CarteiraRecusada(violacoes)
        return pesos[pesos > 0]
    return _estrategia
