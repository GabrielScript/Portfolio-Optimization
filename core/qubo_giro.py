"""
qubo_giro.py - The turnover arm of the hybrid QPO (docs/superpowers/specs/2026-11-01-qpo-hibrido-design.md, Prop. 4 and
section 4): the Composite's QUBO over the full universe, with Mugel's turnover term on the portfolio held. New module, so
the paper guards' paths stay as they are.
"""

from __future__ import annotations

from collections.abc import Callable, Collection, Mapping
from dataclasses import replace

import numpy as np
import pandas as pd

from core.backtest import CarteiraRecusada, Estrategia
from core.estimation import estimar_momentos
from core.qpo import K, QuboInstance, candidates, qubo_instance
from core.qubo_universo import FULL_UNIVERSE, SECTOR_CAP, anneal_solver
from core.risco import LIMITES, Limites, checar_carteira
from core.valor import NotaDoMes

REBALANCES_PER_YEAR = 4
COST_PER_SIDE = 0.001
TAU = REBALANCES_PER_YEAR * COST_PER_SIDE / K       # derived in the spec, not tuned: yearly units, as the QUBO's alpha


def com_giro(instancia: QuboInstance, carteira: Collection[str], tau: float = TAU) -> QuboInstance:
    """Q + tau diag(1 - 2h), h = the names held: x'Qx + tau sum (x - h)^2 - tau |h| (Prop. 4). A held name outside the
    instance has no bit, so it adds nothing here."""
    h = np.isin(instancia.tickers, list(carteira)).astype(float)
    return replace(instancia, q=instancia.q + tau * np.diag(1.0 - 2.0 * h))


def estrategia_qubo_giro(nota: NotaDoMes, setores: Mapping[str, str], log: dict | None = None, limites: Limites = LIMITES,
                         resolver: Callable[[QuboInstance], tuple[int, ...]] | None = None) -> Estrategia:
    """At each decision, the capped QUBO of ``nota`` over every eligible name with the turnover term on the portfolio
    held, equal weights. One instance per backtest: it remembers the portfolio, and a refused one was never held."""
    resolver = resolver if resolver is not None else anneal_solver(setores, SECTOR_CAP, log)
    carteira: set[str] = set()

    def _estrategia(treino: pd.DataFrame) -> pd.Series:
        data = treino.index[-1]
        nota_do_mes = nota(data.to_period("M").to_timestamp(), list(treino.columns)).dropna()
        instancia = qubo_instance(data, treino, nota_do_mes, setores, FULL_UNIVERSE)
        if instancia is None:
            nomes = candidates(nota_do_mes, setores, FULL_UNIVERSE)
        else:
            nomes = [instancia.tickers[i] for i in resolver(com_giro(instancia, carteira))]
        if not nomes:
            raise ValueError(f"{data:%Y-%m-%d}: no eligible name with a score")
        pesos = pd.Series(1.0 / len(nomes), index=nomes)
        violacoes = checar_carteira(pesos, setores, estimar_momentos(treino[nomes], shrink_mu=False).cov, limites)
        if violacoes:
            raise CarteiraRecusada(violacoes)
        carteira.clear()
        carteira.update(nomes)
        return pesos
    return _estrategia
