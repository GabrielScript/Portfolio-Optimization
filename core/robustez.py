"""
robustez.py - Métricas de robustez fora da amostra (PLAN §11, passo 3).

Módulo à parte de ``core.metrics`` de propósito: o ``metrics.py`` está no caminho do satélite do paper trading
(``scripts/paper_trading.CAMINHO_DO_SATELITE``) e qualquer mudança nele bloqueia a rotina.

Convenções do ``core.metrics``: momentos POR PERÍODO (diários), curtose BRUTA, 252 pregões por ano.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy import stats

from core.config import DIAS_UTEIS_ANO
from core.metrics import (MomentosSharpe, cagr, drawdown, momentos_sharpe, retornos_de_valores,
                          sharpe_probabilistico)


def anos_minimos_para_psr(m: MomentosSharpe, sharpe_referencia: float = 0.0, confianca: float = 0.95) -> float:
    """
    MinTRL (Bailey e López de Prado, 2012), em anos: o histórico mínimo para o PSR chegar a ``confianca`` se o
    Sharpe e os momentos observados forem os verdadeiros.

        T* = 1 + (1 - g3 SR + (g4 - 1)/4 SR^2) (z / (SR - SR*))^2

    É a conta do ``sharpe_probabilistico`` resolvida para T. Sharpe que não passa da referência nunca chega: ∞.
    """
    if m.sharpe <= sharpe_referencia:
        return math.inf
    variancia = 1.0 - m.assimetria * m.sharpe + (m.curtose - 1.0) / 4.0 * m.sharpe**2
    z = stats.norm.ppf(confianca)
    return (1.0 + variancia * (z / (m.sharpe - sharpe_referencia)) ** 2) / DIAS_UTEIS_ANO


def psr_da_diferenca(valores: pd.Series, valores_referencia: pd.Series, log: bool = False) -> float:
    """
    P[curva > referência]: PSR da diferença dos retornos diários contra 0. Com ``log``, a diferença é a dos
    log-retornos, cuja média é a diferença de crescimento da riqueza; a aritmética favorece a curva mais volátil.
    """
    retornos, referencia = retornos_de_valores(valores), retornos_de_valores(valores_referencia)
    diferenca = np.log1p(retornos) - np.log1p(referencia) if log else retornos - referencia
    return sharpe_probabilistico(momentos_sharpe(diferenca))


COLUNAS_MEDIDAS = ["cagr", "excesso_cagr", "p_aritmetica", "p_log", "mdd"]
COLUNAS_JANELAS = ["inicio", "fim", "curva", *COLUNAS_MEDIDAS]


def pares_de_janelas(datas: pd.DatetimeIndex, meses: int) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """(início, fim) das janelas de ``meses`` meses: do último pregão de cada mês ao último pregão ``meses`` depois."""
    fins_de_mes = datas.to_series().groupby(datas.to_period("M")).max()
    return list(zip(fins_de_mes.iloc[:-meses], fins_de_mes.iloc[meses:]))


def medir(valores: pd.Series, valores_referencia: pd.Series) -> list[float]:
    """Uma curva contra a referência no mesmo trecho, na ordem de ``COLUNAS_MEDIDAS``."""
    return [cagr(valores), cagr(valores) - cagr(valores_referencia), psr_da_diferenca(valores, valores_referencia),
            psr_da_diferenca(valores, valores_referencia, log=True), drawdown(valores)[1]]


def janelas_moveis(curvas: pd.DataFrame, referencia: str, meses: int) -> pd.DataFrame:
    """
    Janelas de ``meses`` meses começando em cada fim de mês (o último pregão do mês no índice), uma linha por janela e
    curva, a ``referencia`` fora: CAGR, excesso de CAGR sobre a referência, P[> referência] aritmético e em log
    (``psr_da_diferenca``) e drawdown máximo. Janelas sobrepostas não são amostras independentes: descrevem, não
    testam. Curva (ou referência) sem valor em algum pregão da janela fica fora dela.
    """
    linhas = []
    for inicio, fim in pares_de_janelas(curvas.index, meses):
        recorte = curvas.loc[inicio:fim]
        bova = recorte[referencia]
        for nome in curvas.columns.drop(referencia):
            valores = recorte[nome]
            if valores.isna().any() or bova.isna().any():
                continue
            linhas.append([inicio, fim, nome, *medir(valores, bova)])
    return pd.DataFrame(linhas, columns=COLUNAS_JANELAS)


CONFIANCA = 0.95


def resumir_janelas(janelas: pd.DataFrame) -> pd.DataFrame:
    """
    Por curva: nº de janelas, fração com CAGR acima da referência (riqueza), mediana e percentil 10 do excesso de
    CAGR, fração com P[> referência] ≥ 0,95 (aritmético e em log) e o pior drawdown.
    """
    por_curva = janelas.groupby("curva", sort=False)
    return pd.DataFrame({
        "janelas": por_curva.size(),
        "acima_do_bova": por_curva["excesso_cagr"].apply(lambda x: float((x > 0).mean())),
        "excesso_mediano": por_curva["excesso_cagr"].median(),
        "excesso_p10": por_curva["excesso_cagr"].quantile(0.10),
        "p_aritmetica_095": por_curva["p_aritmetica"].apply(lambda p: float((p >= CONFIANCA).mean())),
        "p_log_095": por_curva["p_log"].apply(lambda p: float((p >= CONFIANCA).mean())),
        "pior_mdd": por_curva["mdd"].min(),
    })
