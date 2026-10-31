"""
The monthly grid of the US test (spec: ``docs/superpowers/specs/2026-10-31-us-estrategias-design.md``). The strategies and
the walk-forward engine are the B3's, unchanged; this module adapts monthly data to them and measures the result. The
free data has no daily history for the dead names, so a "session" here is a month.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

MESES_POR_ANO = 12
DIAS_DO_ANO_DO_MOTOR = 252
FATORES = ("mkt_rf", "smb", "hml", "rmw", "cma", "mom")
P_MINIMO = 0.95
LAGS_NEWEY_WEST = 3
_FIM_DE_MES = pd.offsets.MonthEnd(0)


@dataclass(frozen=True)
class ConfigMensal:
    """What the engine reads of ``ConfigBacktest``, whose own check requires 60 rows of training: it counts sessions."""

    janela_treino: int = 36
    janela_teste: int = 3
    capital_inicial: float = 100_000.0
    custo_bps: float = 10.0


CONFIG = ConfigMensal()


def indice_total(retornos: pd.DataFrame, terminais: pd.Series | None = None) -> pd.DataFrame:
    """Total-return index at month ends from ``retornos`` (first day of the month x security). A name starts at 1 the month
    before its first return and stays empty after its last price. ``terminais``: the loss of a name's last month."""
    fatores = 1.0 + retornos
    if terminais is not None:
        for papel, perda in terminais.items():
            ultimo = retornos[papel].last_valid_index()
            if ultimo is not None:
                fatores.loc[ultimo, papel] *= 1.0 + perda
    indice = fatores.cumprod(skipna=True)
    valores = indice.to_numpy(copy=True)
    com_dado = indice.notna().to_numpy()
    primeira = com_dado.argmax(axis=0)
    colunas = np.flatnonzero(com_dado.any(axis=0) & (primeira > 0))
    valores[primeira[colunas] - 1, colunas] = 1.0
    return pd.DataFrame(valores, index=indice.index + _FIM_DE_MES, columns=indice.columns)


def no_grid_mensal(estrategia: Callable[[pd.DataFrame], pd.Series]) -> Callable[[pd.DataFrame], pd.Series]:
    """A strategy of the engine on the monthly grid. It takes the month of the training's last row as the decision's, where
    daily data has both in the same month, so the rows move one month ahead. Returns go to the scale of a day, because
    it annualizes the risk rule's covariance by 252."""
    escala = np.sqrt(MESES_POR_ANO / DIAS_DO_ANO_DO_MOTOR)

    def _estrategia(treino: pd.DataFrame) -> pd.Series:
        return estrategia(pd.DataFrame(treino.to_numpy() * escala, columns=treino.columns,
                                       index=treino.index + pd.offsets.MonthEnd(1)))
    return _estrategia


def medidas(valores: pd.Series, rf: pd.Series) -> dict[str, float]:
    """CAGR, volatility, Sharpe over ``rf`` (per month) and maximum drawdown of a monthly curve."""
    retornos = valores.pct_change().dropna()
    excesso = retornos - rf.reindex(retornos.index).fillna(0.0)
    desvio = excesso.std()
    return {"CAGR": float((valores.iloc[-1] / valores.iloc[0]) ** (MESES_POR_ANO / len(retornos)) - 1.0),
            "Vol": float(retornos.std() * np.sqrt(MESES_POR_ANO)),
            "Sharpe": float(excesso.mean() / desvio * np.sqrt(MESES_POR_ANO)) if desvio > 1e-12 else float("nan"),
            "Queda máx.": float((valores / valores.cummax() - 1.0).min())}


def alfa_fatorial(excesso: pd.Series, fatores: pd.DataFrame) -> tuple[float, float, dict[str, float]]:
    """Annual alpha of ``excesso`` on the FF5 + momentum factors, its Newey-West t, and the betas."""
    juntos = pd.concat([excesso.rename("y"), fatores.loc[:, list(FATORES)]], axis=1, join="inner").dropna()
    y = juntos["y"].to_numpy()
    x = np.column_stack([np.ones(len(juntos)), juntos[list(FATORES)].to_numpy()])
    inversa = np.linalg.inv(x.T @ x)
    beta = inversa @ x.T @ y
    residuos = x * (y - x @ beta)[:, None]
    meio = residuos.T @ residuos
    for lag in range(1, LAGS_NEWEY_WEST + 1):
        cruzado = residuos[lag:].T @ residuos[:-lag]
        meio += (1.0 - lag / (LAGS_NEWEY_WEST + 1)) * (cruzado + cruzado.T)
    erro = np.sqrt((inversa @ meio @ inversa)[0, 0])
    return float(beta[0] * MESES_POR_ANO), float(beta[0] / erro), dict(zip(FATORES, map(float, beta[1:]), strict=True))


def passa_o_portao(p_contra_referencia: Sequence[float], dsr: Sequence[float]) -> bool:
    return all(valor >= P_MINIMO for valor in (*p_contra_referencia, *dsr))


def em_reais(valores: pd.Series, ptax: pd.Series) -> pd.Series:
    """A dollar curve as a Brazilian holds it: each date's value times the PTAX of that date over the first one's."""
    cambio = ptax.reindex(valores.index).ffill()
    return valores * cambio / cambio.iloc[0]


def melhores_em_12_meses(retornos: pd.DataFrame, mes: pd.Timestamp, quantos: int) -> pd.Series:
    """Equal weights on the ``quantos`` names with the best return in the 12 months before ``mes``, all 12 present."""
    janela = retornos.loc[retornos.index < mes].tail(MESES_POR_ANO).dropna(axis=1)
    melhores = (1.0 + janela).prod().sort_values(ascending=False, kind="stable").head(quantos)
    return pd.Series(1.0 / len(melhores), index=melhores.index) if len(melhores) else pd.Series(dtype=float)


def rotacao_trimestral(retornos: pd.DataFrame, inicio: pd.Timestamp, quantos: int, custo_bps: float,
                       passo: int = 3) -> pd.Series:
    """The best of the last 12 months, bought every ``passo`` months from ``inicio`` and held; costs on what changes.
    A name with no return in a month keeps its value."""
    meses = retornos.index[retornos.index >= inicio]
    posicao = pd.Series(dtype=float)
    valor = 1.0
    serie = {inicio - pd.Timedelta(days=1): valor}
    for numero, mes in enumerate(meses):
        if numero % passo == 0:
            pesos = melhores_em_12_meses(retornos, mes, quantos)
            valor -= float(pesos.mul(valor).sub(posicao, fill_value=0.0).abs().sum()) * custo_bps / 10_000.0
            posicao = pesos * valor
        posicao = posicao * (1.0 + retornos.loc[mes, posicao.index].fillna(0.0))
        valor = float(posicao.sum())
        serie[mes + _FIM_DE_MES] = valor
    return pd.Series(serie)


def melhores_em_retrospecto(retornos: pd.DataFrame, inicio: pd.Timestamp, quantos: int) -> pd.Series:
    """Equal weights, held, in the ``quantos`` names with the best return over the whole period, present in all of it.
    Nobody could have chosen them at the start: this is the ceiling the labels call hindsight."""
    janela = retornos.loc[retornos.index >= inicio].dropna(axis=1)
    acumulado = (1.0 + janela).cumprod()
    melhores = acumulado.iloc[-1].sort_values(ascending=False, kind="stable").head(quantos).index
    curva = acumulado[melhores].mean(axis=1)
    curva.index = curva.index + _FIM_DE_MES
    return pd.concat([pd.Series({inicio - pd.Timedelta(days=1): 1.0}), curva])


def referencia_nas_datas(retornos: pd.Series, datas: pd.DatetimeIndex) -> pd.Series:
    """An ETF's value on an arm's dates, 1 on the first, with no costs. ``retornos``: first day of the month x return."""
    mensais = retornos.copy()
    mensais.index = mensais.index + _FIM_DE_MES
    return pd.concat([pd.Series([1.0], index=datas[:1]), (1.0 + mensais.reindex(datas[1:]).fillna(0.0)).cumprod()])


def so_depois_de_qualificar(retornos: pd.DataFrame, primeiras: pd.Series, defasagem: int) -> pd.DataFrame:
    """The funds of ``primeiras`` (CNPJ -> first month its book qualified as an equity fund), each with its returns only
    after that book became public, ``defasagem`` months later: nobody could tell it was an equity fund before."""
    comuns = retornos.columns.intersection(primeiras.index)
    publicos = primeiras[comuns] + pd.DateOffset(months=defasagem)
    return pd.DataFrame({fundo: retornos[fundo].where(retornos.index > publicos[fundo]) for fundo in comuns},
                        index=retornos.index)
