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

from collections.abc import Iterable, Mapping

import pandas as pd

from core.backtest import CarteiraRecusada, Estrategia
from core.estimation import estimar_momentos
from core.risco import LIMITES, Limites, checar_carteira, emissor

MESES_DO_MOMENTUM = 11      # 12-1: do mês m-12 ao m-2; o m-1, o último fechado, fica de fora (reversão de curto prazo)
MESES_DA_VOL = 36           # vol de 3 anos de retornos mensais (Blitz e van Vliet, 2018)
NOMES = 15                  # papéis da carteira, pesos iguais (PLAN §5, S2: fixo no pré-registro)
POR_SETOR = 4               # 4/15 = 26,7%, abaixo do teto de 30% por setor


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


def carteira_da_nota(nota: pd.Series, setores: Mapping[str, str], nomes: int, por_setor: int) -> pd.Series:
    """
    Pesos iguais nos ``nomes`` papéis de maior nota, pulando: papel sem setor (a regra fixa SEM_SETOR), a 2ª classe
    de um emissor já escolhido (fica a de nota maior) e o papel de um setor que já tem ``por_setor`` (com 15 nomes e
    4 por setor, 26,7% < 30%, o teto de setor). Com menos candidatos que ``nomes``, pesos iguais nos que houver.
    """
    escolhidos: list[str] = []
    emissores: set[str] = set()
    contagem: dict[str, int] = {}
    for papel in nota.sort_values(ascending=False, kind="stable").index:
        setor = setores.get(papel)
        if not setor or emissor(papel) in emissores or contagem.get(setor, 0) >= por_setor:
            continue
        escolhidos.append(papel)
        emissores.add(emissor(papel))
        contagem[setor] = contagem.get(setor, 0) + 1
        if len(escolhidos) == nomes:
            break
    return pd.Series(1.0 / len(escolhidos), index=escolhidos) if escolhidos else pd.Series(dtype=float)


def estrategia_conservadora(retornos_mensais: pd.DataFrame, proventos: pd.DataFrame, setores: Mapping[str, str],
                            nomes: int = NOMES, por_setor: int = POR_SETOR, limites: Limites = LIMITES) -> Estrategia:
    """
    A Conservadora no motor de walk-forward: na decisão (último pregão do treino), a carteira da nota do mês dela,
    que usa só os meses e proventos antes desse mês (o sinal do IC daquele mês), nos papéis que o motor deixa
    elegíveis. A carteira passa por ``checar_carteira`` com a cov de Ledoit-Wolf do treino; violação =
    ``CarteiraRecusada`` (o motor mantém a atual).
    """
    def _estrategia(treino: pd.DataFrame) -> pd.Series:
        mes = treino.index[-1].to_period("M").to_timestamp()
        pesos = carteira_da_nota(conservadora(retornos_mensais, proventos, mes, treino.columns), setores, nomes,
                                 por_setor)
        if pesos.empty:
            raise ValueError(f"{treino.index[-1]:%Y-%m-%d}: nenhum papel elegível com a nota da Conservadora")
        violacoes = checar_carteira(pesos, setores, estimar_momentos(treino[pesos.index], shrink_mu=False).cov,
                                    limites)
        if violacoes:
            raise CarteiraRecusada(violacoes)
        return pesos
    return _estrategia
