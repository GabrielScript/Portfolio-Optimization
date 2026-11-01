"""
estrategias_todas.py - The eleven new arms (docs/superpowers/specs/2026-11-01-todas-as-estrategias-design.md, section 3) as strategies
of the engine, on the B3's daily rows and on the US's monthly ones. As ``estrategia_otimizada_nas_regras``: only names with a sector
are eligible, the sector cap goes to the optimizer, and the portfolio goes through ``checar_carteira``; a violation is
``CarteiraRecusada``, so the engine keeps the current portfolio.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace

import numpy as np
import pandas as pd

from core.backtest import CarteiraRecusada, Estrategia
from core.baixo_posto import CovarianciaBaixoPosto, densa, ledoit_wolf_baixo_posto
from core.carteiras_livros import (media_por_periodo, pesos_cdar, pesos_cvar, pesos_herc, pesos_ivar, pesos_ivol, pesos_mdec,
                                   pesos_mdiv, pesos_mvsk, pesos_robusto, pesos_rpp)
from core.config import ConfigOtimizacao, Objetivo, limites_por_setor
from core.hierarchical import otimizar_hrp
from core.optimization import ajustar_aos_tetos, projetar_pesos
from core.risco import LIMITES, Limites, checar_carteira

N_CONCENTRADOS = 10
AVERSAO_ROBUSTO = 1.0       # the spec's (1/2) w'Sigma w


@dataclass(frozen=True)
class Decisao:
    """What an arm sees at one decision. ``sigma`` is annual; ``cfg`` holds the caps of that decision's names."""

    retornos: pd.DataFrame
    sigma: CovarianciaBaixoPosto
    periodos: int
    cfg: ConfigOtimizacao

    @property
    def media(self) -> pd.Series:
        return media_por_periodo(self.retornos, self.sigma, self.periodos)

    @property
    def anos(self) -> float:
        return len(self.retornos) / self.periodos


Braco = Callable[[Decisao], pd.Series]
Construtor = Callable[..., Estrategia]


def _no_universo(regra: Callable[[CovarianciaBaixoPosto], pd.Series]) -> Braco:
    """A rule that holds every name, its weights cut to the caps."""
    return lambda decisao: ajustar_aos_tetos(regra(decisao.sigma), decisao.cfg)


def _hrp_projetado(decisao: Decisao) -> pd.Series:
    cov = densa(decisao.sigma)
    return projetar_pesos(otimizar_hrp(cov).reindex(cov.index), cov, decisao.cfg).pesos


def _cvar(decisao: Decisao) -> pd.Series:
    return pesos_cvar(decisao.retornos, decisao.media, decisao.sigma, decisao.cfg)


def _cdar(decisao: Decisao) -> pd.Series:
    return pesos_cdar(decisao.retornos, decisao.media, decisao.sigma, decisao.cfg)


def _mvsk(decisao: Decisao) -> pd.Series:
    return pesos_mvsk(decisao.retornos, decisao.media, decisao.sigma, decisao.periodos, decisao.cfg)


def _robusto(decisao: Decisao) -> pd.Series:
    return pesos_robusto(decisao.media * decisao.periodos, decisao.sigma, decisao.cfg, decisao.anos)


def _construtor(braco: Braco) -> Construtor:
    def construir(setores: Mapping[str, str], periodos: int, limites: Limites = LIMITES) -> Estrategia:
        def _estrategia(treino: pd.DataFrame) -> pd.Series:
            retornos = treino[[papel for papel in treino.columns if setores.get(papel)]]
            sigma = ledoit_wolf_baixo_posto(retornos, periodos)
            cfg = ConfigOtimizacao(objetivo=Objetivo.KELLY, peso_maximo=limites.peso_maximo_acao, n_ativos_max=N_CONCENTRADOS,
                                   vol_maxima=limites.vol_maxima, aversao_risco=AVERSAO_ROBUSTO,
                                   grupos=limites_por_setor({papel: setores[papel] for papel in retornos.columns}, limites.teto_setor))
            pesos = braco(Decisao(retornos, sigma, periodos, cfg)).reindex(retornos.columns).fillna(0.0)
            violacoes = checar_carteira(pesos, setores, densa(sigma, np.flatnonzero(pesos.to_numpy() > 0.0)), limites)
            if violacoes:
                raise CarteiraRecusada(violacoes)
            return pesos
        return _estrategia
    return construir


BRACOS_NOVOS: dict[str, Construtor] = {
    "HRP-projetado": _construtor(_hrp_projetado),
    "IVarP": _construtor(_no_universo(pesos_ivar)),
    "IVolP": _construtor(_no_universo(pesos_ivol)),
    "MDivP": _construtor(lambda decisao: pesos_mdiv(decisao.sigma, decisao.cfg)),
    "MDecP": _construtor(_no_universo(pesos_mdec)),
    "RPP": _construtor(_no_universo(pesos_rpp)),
    "HERC": _construtor(_no_universo(pesos_herc)),
    "Mean-CVaR": _construtor(_cvar),
    "Mean-CDaR": _construtor(_cdar),
    "MVSK": _construtor(_mvsk),
    "Robust-MV": _construtor(_robusto),
}


def no_grid_mensal_real(estrategia: Estrategia) -> Estrategia:
    """``no_grid_mensal`` without its scale to a day: these arms annualize by 12 and read the mean, which that scale would inflate."""
    def _estrategia(treino: pd.DataFrame) -> pd.Series:
        return estrategia(treino.set_axis(treino.index + pd.offsets.MonthEnd(1)))
    return _estrategia
