"""
composto_faixa.py - The Composite with a holding band (spec docs/superpowers/specs/2026-11-15-composto-faixa-design.md).

A name held from the previous accepted book stays while it ranks among the first ``faixa`` names the book could take
(score order, skipping a name without a sector and an issuer's second class); the other slots are filled as
``carteira_da_nota`` fills them, the kept names counting in the sector and issuer counts. Equal weights.
"""

from __future__ import annotations

from collections.abc import Collection, Mapping

import pandas as pd

from core.backtest import CarteiraRecusada, Estrategia
from core.estimation import estimar_momentos
from core.risco import LIMITES, Limites, checar_carteira, emissor
from core.sinais import NOMES, POR_SETOR
from core.valor import NotaDoMes

FAIXA = 20


def _ordem(nota: pd.Series, setores: Mapping[str, str]) -> list[str]:
    """The names the book could take, best score first: a sector, one class per issuer."""
    vistos: set[str] = set()
    ordem = []
    for papel in nota.sort_values(ascending=False, kind="stable").index:
        if setores.get(papel) and emissor(papel) not in vistos:
            ordem.append(papel)
            vistos.add(emissor(papel))
    return ordem


def carteira_com_faixa(nota: pd.Series, setores: Mapping[str, str], mantidos: Collection[str], nomes: int = NOMES,
                       por_setor: int = POR_SETOR, faixa: int = FAIXA) -> pd.Series:
    ordem = _ordem(nota, setores)
    escolhidos = [p for p in ordem[:faixa] if p in set(mantidos)][:nomes]
    contagem: dict[str, int] = {}
    for papel in escolhidos:
        contagem[setores[papel]] = contagem.get(setores[papel], 0) + 1
    for papel in ordem:
        if len(escolhidos) == nomes:
            break
        if papel in escolhidos or contagem.get(setores[papel], 0) >= por_setor:
            continue
        escolhidos.append(papel)
        contagem[setores[papel]] = contagem.get(setores[papel], 0) + 1
    escolhidos.sort(key=ordem.index)
    return pd.Series(1.0 / len(escolhidos), index=escolhidos) if escolhidos else pd.Series(dtype=float)


def estrategia_com_faixa(nota: NotaDoMes, setores: Mapping[str, str], faixa: int = FAIXA,
                         limites: Limites = LIMITES) -> Estrategia:
    """``core.composto.estrategia_por_nota`` with the band; the held book is the last one accepted (a refusal keeps it)."""
    mantidos: set[str] = set()

    def _estrategia(treino: pd.DataFrame) -> pd.Series:
        mes = treino.index[-1].to_period("M").to_timestamp()
        pesos = carteira_com_faixa(nota(mes, list(treino.columns)).dropna(), setores, mantidos, faixa=faixa)
        if pesos.empty:
            raise ValueError(f"{treino.index[-1]:%Y-%m-%d}: no eligible name with a score")
        violacoes = checar_carteira(pesos, setores, estimar_momentos(treino[pesos.index], shrink_mu=False).cov, limites)
        if violacoes:
            raise CarteiraRecusada(violacoes)
        mantidos.clear()
        mantidos.update(pesos.index)
        return pesos
    return _estrategia
