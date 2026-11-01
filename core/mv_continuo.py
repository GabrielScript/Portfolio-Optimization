"""
mv_continuo.py - The continuous arm of the hybrid QPO (docs/superpowers/specs/2026-11-01-qpo-hibrido-design.md, Prop. 3 and
section 4): the QUBO's relaxation, max alpha'w - (1/2) w'Sigma w, which is what Rebentrost-Lloyd's and Kerenidis et al.'s
quantum solvers would output, solved exactly on the classical side.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import replace

import pandas as pd

from core.backtest import CarteiraRecusada, Estrategia
from core.baixo_posto import densa, ledoit_wolf_baixo_posto, otimizar_baixo_posto
from core.config import ConfigOtimizacao, Objetivo, limites_por_setor
from core.estimation import estimar_momentos
from core.optimization import otimizar
from core.qpo import RISK_AVERSION, candidates, qubo_alphas
from core.qubo_universo import FULL_UNIVERSE
from core.risco import LIMITES, Limites, checar_carteira
from core.valor import NotaDoMes

CFG = ConfigOtimizacao(objetivo=Objetivo.KELLY, peso_maximo=0.20, n_ativos_max=None, vol_maxima=0.40,
                       aversao_risco=RISK_AVERSION)


def _denso(treino: pd.DataFrame, nota: pd.Series, cfg: ConfigOtimizacao) -> tuple[pd.Series, pd.DataFrame]:
    cov = estimar_momentos(treino, shrink_mu=False).cov
    return otimizar(qubo_alphas(nota, cov), cov, cfg).pesos, cov


def _baixo_posto(treino: pd.DataFrame, nota: pd.Series, cfg: ConfigOtimizacao) -> tuple[pd.Series, pd.DataFrame]:
    sigma = ledoit_wolf_baixo_posto(treino)
    cov = densa(sigma)
    return otimizar_baixo_posto(qubo_alphas(nota, cov), sigma, cfg), cov


def _estrategia_com(nota: NotaDoMes, setores: Mapping[str, str], limites: Limites,
                    resolver: Callable[[pd.DataFrame, pd.Series, ConfigOtimizacao], tuple[pd.Series, pd.DataFrame]]) -> Estrategia:
    def _estrategia(treino: pd.DataFrame) -> pd.Series:
        data = treino.index[-1]
        nota_do_mes = nota(data.to_period("M").to_timestamp(), list(treino.columns)).dropna()
        nomes = candidates(nota_do_mes, setores, FULL_UNIVERSE)
        if not nomes:
            raise ValueError(f"{data:%Y-%m-%d}: no eligible name with a score")
        grupos = limites_por_setor({papel: setores[papel] for papel in nomes}, limites.teto_setor)
        pesos, cov = resolver(treino[nomes], nota_do_mes[nomes], replace(CFG, grupos=grupos))
        violacoes = checar_carteira(pesos, setores, cov, limites)
        if violacoes:
            raise CarteiraRecusada(violacoes)
        return pesos
    return _estrategia


def estrategia_mv_continua(nota: NotaDoMes, setores: Mapping[str, str], limites: Limites = LIMITES) -> Estrategia:
    """At each decision, one class per issuer among the names with a score and a sector, the Composite's alpha and the
    training window's Ledoit-Wolf covariance; checked by the fixed rules, a violation is ``CarteiraRecusada``."""
    return _estrategia_com(nota, setores, limites, _denso)


def estrategia_mv_continua_grande(nota: NotaDoMes, setores: Mapping[str, str], limites: Limites = LIMITES) -> Estrategia:
    """The same arm with the same optimum on the covariance's low-rank form, for a universe of thousands of names on few rows."""
    return _estrategia_com(nota, setores, limites, _baixo_posto)
