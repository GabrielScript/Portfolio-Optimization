"""
mv_continuo.py - The continuous arm of the hybrid QPO (docs/superpowers/specs/2026-11-01-qpo-hibrido-design.md, Prop. 3 and
section 4): the QUBO's relaxation, max alpha'w - (1/2) w'Sigma w, which is what Rebentrost-Lloyd's and Kerenidis et al.'s
quantum solvers would output, solved exactly on the classical side.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace

import pandas as pd

from core.backtest import CarteiraRecusada, Estrategia
from core.config import ConfigOtimizacao, Objetivo, limites_por_setor
from core.estimation import estimar_momentos
from core.optimization import otimizar
from core.qpo import RISK_AVERSION, candidates, qubo_alphas
from core.qubo_universo import FULL_UNIVERSE
from core.risco import LIMITES, Limites, checar_carteira
from core.valor import NotaDoMes

CFG = ConfigOtimizacao(objetivo=Objetivo.KELLY, peso_maximo=0.20, n_ativos_max=None, vol_maxima=0.40,
                       aversao_risco=RISK_AVERSION)


def estrategia_mv_continua(nota: NotaDoMes, setores: Mapping[str, str], limites: Limites = LIMITES) -> Estrategia:
    """At each decision, one class per issuer among the names with a score and a sector, the Composite's alpha and the
    training window's Ledoit-Wolf covariance; checked by the fixed rules, a violation is ``CarteiraRecusada``."""
    def _estrategia(treino: pd.DataFrame) -> pd.Series:
        data = treino.index[-1]
        nota_do_mes = nota(data.to_period("M").to_timestamp(), list(treino.columns)).dropna()
        nomes = candidates(nota_do_mes, setores, FULL_UNIVERSE)
        if not nomes:
            raise ValueError(f"{data:%Y-%m-%d}: no eligible name with a score")
        cov = estimar_momentos(treino[nomes], shrink_mu=False).cov
        grupos = limites_por_setor({papel: setores[papel] for papel in nomes}, limites.teto_setor)
        pesos = otimizar(qubo_alphas(nota_do_mes[nomes], cov), cov, replace(CFG, grupos=grupos)).pesos
        violacoes = checar_carteira(pesos, setores, cov, limites)
        if violacoes:
            raise CarteiraRecusada(violacoes)
        return pesos
    return _estrategia
