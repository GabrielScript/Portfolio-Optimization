"""
estrategias_lote3.py - Three optimizers fed the Composite's alpha (docs/superpowers/specs/2026-11-05-lote3-composto-design.md, section 1)
as strategies of the engine, under the book batches' rule check: a violation is ``CarteiraRecusada``, the engine keeps the current portfolio.

The forecast is MV-continuo's, Grinold-Kahn's alpha = IC0 x vol x z of the Composite's score, on the names ``core.qpo.candidates``
keeps (a score, a sector, one class per issuer). Rows that read scenarios get them recentred on that alpha: the risk is the history's,
the mean is the signal's.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace

import pandas as pd

from core.backtest import Estrategia
from core.baixo_posto import densa, ledoit_wolf_baixo_posto
from core.carteiras_livros import pesos_cvar, pesos_robusto
from core.carteiras_lote2 import Utilidade, pesos_utilidade
from core.estrategias_lote2 import FOLGA_DO_TETO
from core.estrategias_todas import Braco, Construtor, Decisao, config_das_regras, pesos_checados
from core.qpo import IC0, candidates, qubo_alphas
from core.qubo_universo import FULL_UNIVERSE
from core.risco import LIMITES, Limites
from core.valor import NotaDoMes

# CRRA with gamma 2, the half Kelly (MacLean, Thorp and Ziemba): U(y) = -1/y
MEIO_KELLY = Utilidade(lambda y: -1.0 / y, lambda y: 1.0 / y ** 2, lambda y: -2.0 / y ** 3)


@dataclass(frozen=True)
class DecisaoComAlfa(Decisao):
    """A decision whose mean is the Composite's alpha (``alfa`` annual) and whose rows are recentred on it."""

    alfa: pd.Series = None

    @property
    def media(self) -> pd.Series:
        return self.alfa / self.periodos


def decisao_com_alfa(treino: pd.DataFrame, nota_do_mes: pd.Series, setores: Mapping[str, str], periodos: int,
                     limites: Limites) -> DecisaoComAlfa:
    nomes = candidates(nota_do_mes.dropna(), setores, FULL_UNIVERSE)
    if not nomes:
        raise ValueError(f"{treino.index[-1]:%Y-%m-%d}: no eligible name with a score")
    retornos = treino[nomes]
    sigma = ledoit_wolf_baixo_posto(retornos, periodos)
    alfa = qubo_alphas(nota_do_mes[nomes], densa(sigma), IC0)
    recentrados = retornos - retornos.mean() + alfa / periodos
    return DecisaoComAlfa(recentrados, sigma, periodos, config_das_regras(nomes, setores, limites), alfa)


def _com_alfa(nota: NotaDoMes, braco: Braco) -> Construtor:
    def construir(setores: Mapping[str, str], periodos: int, limites: Limites = LIMITES) -> Estrategia:
        def _estrategia(treino: pd.DataFrame) -> pd.Series:
            nota_do_mes = nota(treino.index[-1].to_period("M").to_timestamp(), list(treino.columns))
            return pesos_checados(braco, decisao_com_alfa(treino, nota_do_mes, setores, periodos, limites), setores, limites)
        return _estrategia
    return construir


def _meio_kelly(decisao: Decisao) -> pd.Series:
    return pesos_utilidade(decisao.retornos, decisao.sigma, decisao.cfg, MEIO_KELLY)


def _cvar(decisao: Decisao) -> pd.Series:
    return pesos_cvar(decisao.retornos, decisao.media, decisao.sigma, decisao.cfg)


def _robusto(decisao: Decisao) -> pd.Series:
    return pesos_robusto(decisao.media * decisao.periodos, decisao.sigma, decisao.cfg, decisao.anos)


def _com_folga(braco: Braco) -> Braco:
    """The optimizer's volatility target just inside the rule's cap, as the second batch: solver noise must not cross it."""
    def com_folga(decisao: Decisao) -> pd.Series:
        return braco(replace(decisao, cfg=replace(decisao.cfg, vol_maxima=decisao.cfg.vol_maxima * (1.0 - FOLGA_DO_TETO))))
    return com_folga


def bracos_lote3(nota: NotaDoMes) -> dict[str, Construtor]:
    """In the spec's row order."""
    return {
        "Meio-Kelly-alfa": _com_alfa(nota, _com_folga(_meio_kelly)),
        "Mean-CVaR-alfa": _com_alfa(nota, _com_folga(_cvar)),
        "Robust-MV-alfa": _com_alfa(nota, _com_folga(_robusto)),
    }

