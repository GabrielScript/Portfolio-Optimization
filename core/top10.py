"""
top10.py - Attempt 55 (docs/superpowers/specs/2026-11-04-top10-design.md): the Composite's portfolio with up to 4 of
its 15 places taken first by the 10 best 252-session returns, leaving out the names already 15% or more below their
63-session high. Research only: the plan, the paper logs and the screen do not read this.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping

import pandas as pd

from core.backtest import CarteiraRecusada, Estrategia
from core.estimation import estimar_momentos
from core.risco import LIMITES, Limites, checar_carteira, emissor
from core.sinais import NOMES, POR_SETOR
from core.valor import NotaDoMes

SESSOES_DO_RETORNO = 252
TOPO = 10
SESSOES_DO_PICO = 63
QUEDA_DO_PICO = 0.15
NOMES_DO_TOPO = 4


def nomes_do_topo(treino: pd.DataFrame, nota: pd.Series, topo: int = TOPO) -> list[str]:
    """The ``topo`` best total returns of ``treino``'s (daily returns) last 252 sessions among the names with a score,
    minus those whose index ends 15% or more below its high of the last 63 sessions (the starting level included)."""
    papeis = [papel for papel in treino.columns if papel in nota.dropna().index]
    retorno = (1.0 + treino[papeis].tail(SESSOES_DO_RETORNO)).prod() - 1.0
    melhores = retorno.sort_values(ascending=False, kind="stable").index[:topo]
    indice = (1.0 + treino[list(melhores)].tail(SESSOES_DO_PICO)).cumprod()
    pico = indice.max().clip(lower=1.0)
    queda = indice.iloc[-1] / pico - 1.0
    return [papel for papel in melhores if queda[papel] > -QUEDA_DO_PICO]


def carteira_com_topo(nota: pd.Series, topo: Iterable[str], setores: Mapping[str, str], nomes: int = NOMES,
                      por_setor: int = POR_SETOR, do_topo: int = NOMES_DO_TOPO) -> pd.Series:
    """Equal weights: up to ``do_topo`` names of ``topo`` in the order of ``nota``, then the rest of ``nota``'s order,
    with ``carteira_da_nota``'s skips (no sector, a second class of a chosen issuer, a full sector)."""
    nota = nota.dropna()
    ordem = nota.sort_values(ascending=False, kind="stable").index
    escolhidos: list[str] = []
    emissores: set[str] = set()
    contagem: dict[str, int] = {}

    def tentar(papel: str) -> bool:
        setor = setores.get(papel)
        if papel in escolhidos or not setor or emissor(papel) in emissores or contagem.get(setor, 0) >= por_setor:
            return False
        escolhidos.append(papel)
        emissores.add(emissor(papel))
        contagem[setor] = contagem.get(setor, 0) + 1
        return True

    do_topo_dentro = 0
    for papel in (papel for papel in ordem if papel in set(topo)):
        if do_topo_dentro == do_topo or len(escolhidos) == nomes:
            break
        do_topo_dentro += tentar(papel)
    for papel in ordem:
        if len(escolhidos) == nomes:
            break
        tentar(papel)
    return pd.Series(1.0 / len(escolhidos), index=escolhidos) if escolhidos else pd.Series(dtype=float)


def estrategia_top10(nota_do_mes: NotaDoMes, setores: Mapping[str, str], limites: Limites = LIMITES) -> Estrategia:
    """``carteira_com_topo`` in the walk-forward engine, checked by the fixed rules like ``estrategia_por_nota``."""
    def _estrategia(treino: pd.DataFrame) -> pd.Series:
        mes = treino.index[-1].to_period("M").to_timestamp()
        nota = nota_do_mes(mes, list(treino.columns)).dropna()
        pesos = carteira_com_topo(nota, nomes_do_topo(treino, nota), setores)
        if pesos.empty:
            raise ValueError(f"{treino.index[-1]:%Y-%m-%d}: no eligible name with a score")
        violacoes = checar_carteira(pesos, setores, estimar_momentos(treino[pesos.index], shrink_mu=False).cov,
                                    limites)
        if violacoes:
            raise CarteiraRecusada(violacoes)
        return pesos
    return _estrategia
