"""
indicadores_no_tempo.py - The columns of the app's per-stock table (``core.indicadores``) as they would have been at
a past month, with only what was known then, to read them as signals
(``docs/superpowers/specs/2026-11-13-indicadores-risco-design.md``, the user's question of 2026-11-13, DECISOES 357).
Every signal is oriented so that higher is better. Research only: the app, the paper logs and the accounts do not
import this.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
import pandas as pd

from core.indicadores import desempenho_por_acao, multiplos, risco_por_acao
from core.risco import emissor
from core.sinais import _meses_antes
from core.valor import fundamentos_no_mes
from core.views import escore_normal

FECHAMENTOS_DE_3_ANOS = 756     # the table's window for Sharpe, Sortino, CVaR and the CAPM
FECHAMENTOS_DO_BETA = 253       # 252 returns, the table's window for the beta
COBERTURA_MINIMA = 0.9          # share of the window's sessions a name must have a close on
DIARIAS = ("SHARPE", "SORTINO", "CVAR", "ALPHA", "T-ALPHA", "BETA")
SINAIS = (*DIARIAS, "RET-3y", "RET-6y", "ROE")
CONJUNTO = "RISK-ALL"
MINIMO_DO_CONJUNTO = 5          # signals a name needs for the mean of the nine normal scores


@dataclass(frozen=True)
class Fontes:
    """What the table is computed from; each function reads only what is dated before its month."""

    precos: pd.DataFrame            # session x ticker: daily total-return closes, ``referencia`` among them
    juro: pd.Series                 # daily CDI, decimal per day
    retornos_mensais: pd.DataFrame  # month (first day) x ticker
    fundamentos: pd.DataFrame       # CVM statements with ``available_from`` (``core.valor``)
    referencia: str = "BOVA11"


def retorno_de_meses(retornos_mensais: pd.DataFrame, mes: pd.Timestamp, meses: int) -> pd.Series:
    """Total return of the ``meses`` calendar months before ``mes``; a name with any of them empty has none (a
    month without a return is not a zero return, as in ``core.sinais.momentum_12_1``)."""
    janela = _meses_antes(retornos_mensais, mes, meses)
    return (1.0 + janela).prod(min_count=meses).sub(1.0).where(janela.notna().all())


def _janela(precos: pd.DataFrame, mes: pd.Timestamp, fechamentos: int, papeis: list[str],
            referencia: str) -> pd.DataFrame:
    """The last ``fechamentos`` sessions before ``mes``: ``referencia`` and the ``papeis`` with a close on at least
    ``COBERTURA_MINIMA`` of them."""
    antes = precos.loc[precos.index < mes, [*papeis, referencia]].iloc[-fechamentos:]
    cobertos = antes[papeis].notna().sum() >= math.ceil(COBERTURA_MINIMA * fechamentos)
    return antes[[*cobertos.index[cobertos], referencia]]


def medidas_diarias(precos: pd.DataFrame, juro: pd.Series, referencia: str, mes: pd.Timestamp,
                    candidatos: Iterable[str]) -> pd.DataFrame:
    """Per candidate with enough closes before ``mes``, the table's own formulas on the window instead of on
    today's history: SHARPE, SORTINO, minus the daily 95% expected shortfall (CVAR), the CAPM alpha against
    ``referencia`` and its Newey-West t over ``FECHAMENTOS_DE_3_ANOS``; minus the beta over the last 252 returns
    (BETA). ``precos``: session x ticker, daily total-return closes; ``juro``: decimal per day."""
    papeis = [papel for papel in candidatos if papel in precos.columns and papel != referencia]
    desempenho = desempenho_por_acao(_janela(precos, mes, FECHAMENTOS_DE_3_ANOS, papeis, referencia), referencia, juro)
    beta = risco_por_acao(_janela(precos, mes, FECHAMENTOS_DO_BETA, papeis, referencia), referencia)["Beta"]
    medidas = pd.DataFrame({"SHARPE": desempenho["Sharpe"], "SORTINO": desempenho["Sortino"],
                            "CVAR": -desempenho["CVaR 95% diário"], "ALPHA": desempenho["Alfa a.a."],
                            "T-ALPHA": desempenho["t do alfa"], "BETA": -beta.drop(index=referencia)},
                           columns=list(DIARIAS))
    return medidas.replace([np.inf, -np.inf], np.nan).reindex(papeis).dropna(how="all").astype(float)


def roe_no_mes(fundamentos: pd.DataFrame, mes: pd.Timestamp, candidatos: Iterable[str]) -> pd.Series:
    """The table's ROE (12-month net income of the parent over equity less minorities, equity > 0) from the latest
    statement received before ``mes`` (``core.valor.fundamentos_no_mes``), on each candidate of the issuer."""
    roe = multiplos(fundamentos_no_mes(fundamentos, mes), pd.Series(dtype=float))["ROE"].dropna()
    papeis = [papel for papel in candidatos if emissor(papel) in roe.index]
    return pd.Series([roe[emissor(papel)] for papel in papeis], index=papeis, dtype=float)


def tabela_no_mes(fontes: Fontes, mes: pd.Timestamp, candidatos: Iterable[str]) -> pd.DataFrame:
    """Candidate x signal for ``mes``: ``SINAIS`` (each empty where the name lacks the history or the statement) and
    ``CONJUNTO``, the mean of their normal scores among the candidates for a name with at least
    ``MINIMO_DO_CONJUNTO`` of them. Candidates without any signal are left out."""
    candidatos = list(candidatos)
    mensais = fontes.retornos_mensais.reindex(columns=candidatos)
    tabela = medidas_diarias(fontes.precos, fontes.juro, fontes.referencia, mes, candidatos).reindex(candidatos)
    tabela["RET-3y"] = retorno_de_meses(mensais, mes, 36)
    tabela["RET-6y"] = retorno_de_meses(mensais, mes, 72)
    tabela["ROE"] = roe_no_mes(fontes.fundamentos, mes, candidatos)
    tabela = tabela.dropna(how="all")
    escores = tabela.apply(escore_normal)
    tabela[CONJUNTO] = escores.mean(axis=1).where(escores.notna().sum(axis=1) >= MINIMO_DO_CONJUNTO)
    return tabela


def residuo(sinal: pd.Series, componentes: pd.DataFrame) -> pd.Series:
    """What the normal score of ``sinal`` keeps after a least-squares fit, across the names with the signal, on an
    intercept and the normal scores of ``componentes`` (name x component, each scored among those names; a missing
    component counts as 0, the mean). Empty with fewer names than the fit needs."""
    alvo = escore_normal(sinal.dropna())
    if len(alvo) < componentes.shape[1] + 2:
        return pd.Series(dtype=float)
    escores = componentes.reindex(alvo.index).apply(escore_normal).fillna(0.0)
    regressores = np.column_stack([np.ones(len(alvo)), escores.to_numpy()])
    coeficientes, *_ = np.linalg.lstsq(regressores, alvo.to_numpy(), rcond=None)
    return alvo - regressores @ coeficientes
