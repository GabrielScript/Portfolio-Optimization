"""
estrategias_lote2.py - The second batch's fourteen arms (docs/superpowers/specs/2026-11-01-lote2-livro-design.md, section 1) as strategies
of the engine, under the same rule check as ``estrategias_todas``: a violation is ``CarteiraRecusada``, the engine keeps the current portfolio.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd

from core.backtest import Estrategia
from core.carteiras_lote2 import (LOG, RAIZ, ConfigSubconjunto, Utilidade, pesos_box, pesos_downside, pesos_quintil, pesos_retorno_minimo,
                                  pesos_subconjunto, pesos_tilting, pesos_utilidade)
from core.composto import estrategia_por_nota
from core.estrategias_todas import Braco, Construtor, Decisao, _construtor
from core.risco import LIMITES, Limites
from core.valor import NotaDoMes

DIAS_DO_MOTOR = 252
MESES_POR_ANO = 12
FOLGA_DO_TETO = 1e-4       # solver noise put a portfolio 3.8e-6 above the 40%, past the rule's 1e-6: the target is 39.996%


def bloco_de_um_mes(periodos: int) -> int:
    return periodos // MESES_POR_ANO


def _nas_regras(braco: Braco) -> Construtor:
    """An arm under the rules, its optimizer's volatility target just inside the rule's cap."""
    def com_folga(decisao: Decisao) -> pd.Series:
        return braco(replace(decisao, cfg=replace(decisao.cfg, vol_maxima=decisao.cfg.vol_maxima * (1.0 - FOLGA_DO_TETO))))
    return _construtor(com_folga)


def _utilidade(utilidade: Utilidade) -> Braco:
    return lambda decisao: pesos_utilidade(decisao.retornos, decisao.sigma, decisao.cfg, utilidade)


def _kelly_sem_teto(setores, periodos: int, limites: Limites = LIMITES) -> Estrategia:
    """The one arm that relaxes the mandatory volatility rule, on purpose (spec, section 1): neither the optimizer nor the check sees it."""
    def braco(decisao: Decisao) -> pd.Series:
        return pesos_utilidade(decisao.retornos, decisao.sigma, replace(decisao.cfg, vol_maxima=None), LOG)
    return _construtor(braco)(setores, periodos, replace(limites, vol_maxima=np.inf))


def _quintil(ranking: str) -> Braco:
    """Normalized as the engine will: a quintile that the caps leave with cash is then seen over them, and refused."""
    def braco(decisao: Decisao) -> pd.Series:
        pesos = pesos_quintil(decisao.media, decisao.sigma, decisao.cfg, ranking)
        return pesos / pesos.sum()
    return braco


def _box(decisao: Decisao) -> pd.Series:
    return pesos_box(decisao.media * decisao.periodos, decisao.sigma, decisao.cfg, decisao.anos)


def _subconjunto(com_linhas: bool) -> Braco:
    def braco(decisao: Decisao) -> pd.Series:
        bloco = bloco_de_um_mes(decisao.periodos) if com_linhas else None
        return pesos_subconjunto(decisao.retornos, decisao.periodos, decisao.cfg, ConfigSubconjunto(bloco=bloco))
    return braco


def _downside(alfa: int) -> Braco:
    return lambda decisao: pesos_downside(decisao.retornos, decisao.sigma, decisao.cfg, alfa)


def _retorno_minimo(decisao: Decisao) -> pd.Series:
    return pesos_retorno_minimo(decisao.media * decisao.periodos, decisao.sigma, decisao.cfg)


def _tilting(nota: NotaDoMes) -> Construtor:
    """From the Composite's portfolio at the same decision. Its rule check annualizes by 252, so it gets rows on the engine's scale."""
    def construir(setores, periodos: int, limites: Limites = LIMITES) -> Estrategia:
        composto = estrategia_por_nota(nota, setores, limites=limites)
        escala = np.sqrt(periodos / DIAS_DO_MOTOR)

        def _estrategia(treino: pd.DataFrame) -> pd.Series:
            referencia = composto(treino * escala)

            def braco(decisao: Decisao) -> pd.Series:
                return pesos_tilting(decisao.retornos, referencia, decisao.media, decisao.sigma, decisao.periodos, decisao.cfg)
            return _nas_regras(braco)(setores, periodos, limites)(treino)
        return _estrategia
    return construir


def bracos_lote2(nota: NotaDoMes) -> dict[str, Construtor]:
    """In the spec's row order. ``nota`` is the Composite's score, which only the tilting arm reads."""
    return {
        "Kelly-exato": _nas_regras(_utilidade(LOG)),
        "Kelly-exato-sem-teto": _kelly_sem_teto,
        "Quintil-mu": _nas_regras(_quintil("mu")),
        "Quintil-mu-sigma": _nas_regras(_quintil("mu_sigma")),
        "Quintil-mu-sigma2": _nas_regras(_quintil("mu_sigma2")),
        "Box-robusto": _nas_regras(_box),
        "Subset-resampling": _nas_regras(_subconjunto(com_linhas=False)),
        "Subset-bagging": _nas_regras(_subconjunto(com_linhas=True)),
        "Downside-1": _nas_regras(_downside(1)),
        "Semi-variancia": _nas_regras(_downside(2)),
        "Downside-3": _nas_regras(_downside(3)),
        "MVSK-tilting": _tilting(nota),
        "Utilidade-raiz": _nas_regras(_utilidade(RAIZ)),
        "Retorno-minimo": _nas_regras(_retorno_minimo),
    }
