"""
sinais.py - Sinais de fatores do S2 (PLAN §5): uma nota por papel no começo de um mês, só com dado público até o fim
do mês anterior.

Dados do lago, mensais: ``retornos_mensais`` = ``monthly_total_returns`` (mês x papel, retorno total do mês, índice no
1º dia do mês; vazio = o papel não negociou ou o retorno foi anulado pelo lago); ``proventos`` = ``dividends``
(``security``, ``ex_date``, ``yield`` = dinheiro por ação / fechamento cru do dia ex). O sinal do mês ``mes`` usa os
meses estritamente antes dele e os proventos com data ex antes do 1º dia dele, então serve igual ao IC (retorno do
próprio ``mes``) e ao walk-forward (decisão dentro do ``mes``, com o mês corrente ainda aberto).
"""

from __future__ import annotations

from collections.abc import Iterable

import pandas as pd

MESES_DO_MOMENTUM = 11      # 12-1: do mês m-12 ao m-2; o m-1, o último fechado, fica de fora (reversão de curto prazo)
MESES_DA_VOL = 36           # vol de 3 anos de retornos mensais (Blitz e van Vliet, 2018)


def _meses_antes(retornos_mensais: pd.DataFrame, mes: pd.Timestamp, meses: int, pulando: int = 0) -> pd.DataFrame:
    """Os ``meses`` meses que terminam ``pulando`` meses antes do mês anterior a ``mes``; mês que falta no painel
    entra vazio (a janela é de calendário, não de linhas)."""
    fim = mes - pd.DateOffset(months=1 + pulando)
    janela = pd.date_range(end=fim, periods=meses, freq="MS")
    return retornos_mensais.reindex(janela)


def momentum_12_1(retornos_mensais: pd.DataFrame, mes: pd.Timestamp) -> pd.Series:
    """Retorno total acumulado do mês m-12 ao m-2 para o sinal de ``mes`` (= m); papel com algum desses meses vazio
    fica sem momentum (NaN): um mês sem retorno não vira 0."""
    janela = _meses_antes(retornos_mensais, mes, MESES_DO_MOMENTUM, pulando=1)
    return (1.0 + janela).prod(min_count=MESES_DO_MOMENTUM).sub(1.0).where(janela.notna().all())


def volatilidade_36m(retornos_mensais: pd.DataFrame, mes: pd.Timestamp) -> pd.Series:
    """Desvio-padrão amostral dos ``MESES_DA_VOL`` retornos mensais antes de ``mes``; NaN sem os 36 (Blitz e van
    Vliet exigem 3 anos de histórico)."""
    janela = _meses_antes(retornos_mensais, mes, MESES_DA_VOL)
    return janela.std(ddof=1).where(janela.notna().all())


def dividend_yield_12m(proventos: pd.DataFrame, mes: pd.Timestamp, papeis: Iterable[str]) -> pd.Series:
    """Soma dos ``yield`` dos proventos com data ex do 1º dia de m-12 ao último de m-1, para o sinal de ``mes``; papel
    sem provento = 0. Cada yield é relativo ao fechamento do próprio dia ex: um desdobramento no meio dos 12 meses não
    mistura valores por ação de escalas diferentes. Sem recompras (precisam da contagem de ações, CVM FRE): o payout
    líquido de Blitz e van Vliet fica para um pré-registro próprio."""
    datas = pd.to_datetime(proventos["ex_date"])
    dentro = (datas >= mes - pd.DateOffset(months=12)) & (datas < mes)
    soma = proventos.loc[dentro].groupby("security")["yield"].sum()
    return soma.reindex(list(papeis)).fillna(0.0)


def conservadora(retornos_mensais: pd.DataFrame, proventos: pd.DataFrame, mes: pd.Timestamp,
                 candidatos: Iterable[str]) -> pd.Series:
    """
    Nota da Conservadora de Blitz e van Vliet (2018) para ``mes``, nos ``candidatos`` com vol de 36 meses e momentum
    12-1 completos (os outros ficam fora): 1 para a metade de menor vol (os ``n // 2`` de menor vol) e 0 para a outra,
    mais a média dos postos percentuais do dividend yield de 12 meses e do momentum dentro da própria metade (empate
    com o posto médio). As ``k`` maiores notas são as ``k`` escolhas da fórmula, e a nota inteira ordena o universo
    para o IC: a metade de menor vol, que é a maior parte do efeito, entra na ordem.
    """
    papeis = [papel for papel in candidatos if papel in retornos_mensais.columns]
    medidas = pd.DataFrame({"vol": volatilidade_36m(retornos_mensais[papeis], mes),
                            "momentum": momentum_12_1(retornos_mensais[papeis], mes)}).dropna()
    medidas["dy"] = dividend_yield_12m(proventos, mes, medidas.index)
    menor_vol = medidas["vol"].rank(method="first") <= len(medidas) // 2
    postos = pd.Series(index=medidas.index, dtype=float)
    for metade in (menor_vol, ~menor_vol):
        grupo = medidas[metade]
        postos[grupo.index] = (grupo["dy"].rank(pct=True) + grupo["momentum"].rank(pct=True)) / 2.0
    return postos + menor_vol.astype(float)
