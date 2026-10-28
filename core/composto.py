"""
composto.py - The Conservadora's portfolio rule ordered by another score of the month, attempt 51 (DECISOES 164,
pre-registered in ``docs/superpowers/specs/2026-10-28-composto-crescimento-design.md``): the Conservadora + value +
profit growth composite. Research only: the Alepha plan, the paper logs and the screen do not read this.

Kept out of ``core/sinais.py``, which is in the paper guards' paths; the rule is ``estrategia_conservadora``'s with the
score passed in.
"""

from __future__ import annotations

from collections.abc import Mapping

import pandas as pd

from core.backtest import CarteiraRecusada, Estrategia
from core.estimation import estimar_momentos
from core.risco import LIMITES, Limites, checar_carteira
from core.sinais import NOMES, POR_SETOR, carteira_da_nota
from core.valor import NotaDoMes


def estrategia_por_nota(nota: NotaDoMes, setores: Mapping[str, str], nomes: int = NOMES, por_setor: int = POR_SETOR,
                        limites: Limites = LIMITES) -> Estrategia:
    """At the decision (the training window's last session), ``carteira_da_nota`` of ``nota`` over the engine's
    eligible names; checked by the fixed rules with the training window's Ledoit-Wolf covariance, a violation is
    ``CarteiraRecusada`` (the engine keeps the current portfolio)."""
    def _estrategia(treino: pd.DataFrame) -> pd.Series:
        mes = treino.index[-1].to_period("M").to_timestamp()
        pesos = carteira_da_nota(nota(mes, list(treino.columns)).dropna(), setores, nomes, por_setor)
        if pesos.empty:
            raise ValueError(f"{treino.index[-1]:%Y-%m-%d}: no eligible name with a score")
        violacoes = checar_carteira(pesos, setores, estimar_momentos(treino[pesos.index], shrink_mu=False).cov,
                                    limites)
        if violacoes:
            raise CarteiraRecusada(violacoes)
        return pesos
    return _estrategia
