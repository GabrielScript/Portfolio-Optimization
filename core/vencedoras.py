"""
vencedoras.py - Extreme winners confirmed by profit growth, attempt 50 (DECISOES 163, pre-registered in
``docs/superpowers/specs/2026-10-28-vencedoras-crescimento-design.md``): stocks whose total return over the last 12
months is above 100% and whose profit-growth score is above the month's median, equal weights capped by the fund's
fixed rules, the rest in the CDI. Research only: the Alepha plan, the paper logs and the screen do not read this.

Kept out of ``core/sinais.py``, ``core/backtest.py`` and ``core/risco.py``, which are in the paper guards' paths.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd

from core.backtest import Estrategia
from core.estimation import estimar_momentos
from core.risco import LIMITES, Limites, checar_carteira
from core.sinais import _meses_antes, carteira_da_nota
from core.valor import NotaDoMes

CAIXA = "CDI"           # the cash column: a CDI total-return index added to the arm's prices
MESES = 12
LIMIAR = 1.0            # 12-month total return above 100%
NOMES = 20
POR_SETOR = 4


class ErroDeRegra(RuntimeError):
    """The weights broke a fixed rule they were built to respect: stops the run (the engine does not catch it)."""


def retorno_12m(retornos_mensais: pd.DataFrame, mes: pd.Timestamp) -> pd.Series:
    """Total return compounded over the calendar months ``mes`` - 12 .. ``mes`` - 1 (no skipped month); a name with
    any of them empty is out."""
    janela = _meses_antes(retornos_mensais, mes, MESES)
    return (1.0 + janela).prod().sub(1.0)[janela.notna().all()]


def escolhas(r12: pd.Series, crescimento: pd.Series, setores: Mapping[str, str]) -> list[str]:
    """The picks: names with ``r12`` above ``LIMIAR`` and ``crescimento`` strictly above its median over every scored
    name, in profit-growth order through ``carteira_da_nota`` (``NOMES`` names, ``POR_SETOR`` per sector, one class
    per issuer, none without a sector)."""
    pontuados = crescimento.dropna()
    vencedoras = r12[r12 > LIMIAR].index
    confirmadas = pontuados[pontuados.index.isin(vencedoras) & (pontuados > pontuados.median())]
    return list(carteira_da_nota(confirmadas, setores, NOMES, POR_SETOR).index)


def pesos_das_escolhas(picks: Sequence[str], setores: Mapping[str, str], cov: pd.DataFrame,
                       limites: Limites = LIMITES) -> pd.Series:
    """Stock weights (the rest is cash): ``min(1/n, peso_maximo_acao)`` each, a sector above ``teto_setor`` scaled
    down to it, then every weight scaled down if the ex-ante vol (annual ``cov``) is above ``vol_maxima``."""
    if not picks:
        return pd.Series(dtype=float)
    pesos = pd.Series(min(1.0 / len(picks), limites.peso_maximo_acao), index=list(picks))
    por_setor = pesos.groupby(pesos.index.map(setores)).transform("sum")
    pesos = pesos * (limites.teto_setor / por_setor).clip(upper=1.0)
    vol = float(np.sqrt(pesos @ cov.loc[pesos.index, pesos.index] @ pesos))
    return pesos * min(1.0, limites.vol_maxima / vol)


def indice_do_caixa(cdi_diario: pd.Series, sessoes: pd.DatetimeIndex) -> pd.Series:
    """The CDI compounded day by day, read on ``sessoes`` (the last level on or before each one; 1.0 before the
    first rate): the cash column's price, so the engine's buy-and-hold earns the CDI between rebalances."""
    nivel = (1.0 + cdi_diario.sort_index()).cumprod()
    return nivel.reindex(nivel.index.union(sessoes)).ffill().reindex(sessoes).fillna(1.0)


def estrategia_vencedoras(retornos_mensais: pd.DataFrame, crescimento: NotaDoMes, setores: Mapping[str, str],
                          limites: Limites = LIMITES) -> Estrategia:
    """The rule in the walk-forward engine: at the decision (the training window's last session), the picks among the
    engine's eligible names, their weights, and ``CAIXA`` = 1 - the stock weights (all of it with no pick). The fund's
    fixed rules are checked on the stock weights; a violation is a wiring bug (``ErroDeRegra``), never a refusal."""
    def _estrategia(treino: pd.DataFrame) -> pd.Series:
        mes = treino.index[-1].to_period("M").to_timestamp()
        candidatos = list(treino.columns)
        picks = escolhas(retorno_12m(retornos_mensais[candidatos], mes), crescimento(mes, candidatos), setores)
        cov = estimar_momentos(treino[picks], shrink_mu=False).cov if picks else pd.DataFrame()
        pesos = pesos_das_escolhas(picks, setores, cov, limites)
        violacoes = checar_carteira(pesos, setores, cov if picks else None, limites)
        if violacoes:
            raise ErroDeRegra(f"{treino.index[-1]:%Y-%m-%d}: {violacoes}")
        return pd.concat([pesos, pd.Series({CAIXA: 1.0 - float(pesos.sum())})])
    return _estrategia
