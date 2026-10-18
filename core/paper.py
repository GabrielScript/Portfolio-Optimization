"""
paper.py - Paper trading (S5): o que o registro diz e o que fazer no pregão.

O ``paper/carteiras.csv`` (``core.registro``) guarda só o satélite: pesos que somam até 100% do dinheiro do satélite,
decididos no fechamento da data registrada e valendo a partir do pregão seguinte. O núcleo é o BOVA11, fixo pela regra
do dinheiro real até o critério de pronto (PLAN, "Regras de risco fixas"), então não é registrado. Daqui saem a curva
do satélite e a decisão do dia; ler o lago, otimizar e gravar ficam no ``scripts/paper_trading.py``.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import pandas as pd

from core.metrics import (ComparacaoBenchmark, comparar_benchmark, momentos_sharpe, retornos_de_valores,
                          sharpe_probabilistico)
from core.risco import LIMITES, TOLERANCIA, pode_negociar, stop_relativo

NUCLEO = "BOVA11"               # núcleo do dinheiro real até o critério de pronto; o satélite é medido contra ele
CUSTO_BPS = 10.0                # por lado, sobre o giro de cada registro (o do walk-forward)
MINIMO_PARA_REGRESSAO = 60      # pregões em comum que o ``comparar_benchmark`` exige
LEITURA_INTERMEDIARIA, LEITURA_FINAL = 126, 252     # pregões depois do 1º registro (~6 e ~12 meses), pré-registrados

PASSO = 63         # pregões entre rebalanceios, o do walk-forward (trimestral)


class Acao(str, Enum):
    AGUARDAR = "aguardar"           # nada devido: segura a carteira registrada
    CONGELADO = "congelado"         # perda diária de 5% ou mais no último pregão: nenhuma ordem hoje
    REBALANCEAR = "rebalancear"     # registra a carteira nova
    STOP = "stop"                   # razão satélite/BOVA11 caiu 25% desde o pico: satélite inteiro no BOVA11
    PARADO = "parado"               # stop já disparou: nunca mais rebalanceia (volta só com pré-registro novo)


def curva_do_registro(carteiras: pd.DataFrame, precos: pd.DataFrame) -> pd.Series:
    """
    Valor do satélite (1,0 no fechamento do 1º registro), segurando os pesos de cada registro até o fechamento do
    seguinte, quando a carteira vira a nova. ``carteiras``: data, ticker, peso (o ``preco`` do registro não entra:
    a curva usa as razões do arquivo de preços atual, que não mudam quando o lago reancora o índice). ``precos``:
    índice de retorno total do lago (data x papel); pregão sem negócio repete o último preço (rende 0). A sobra
    abaixo de 100% fica em caixa, rendendo 0.
    """
    fora = sorted(set(carteiras["ticker"]) - set(precos.columns))
    if fora:
        raise ValueError(f"Sem preço no arquivo do lago para {', '.join(fora)}: conferir no lago (deslistada?).")
    datas = sorted(carteiras["data"].unique())
    precos = precos.ffill()
    valores = []
    valor = 1.0
    for inicio, fim in zip(datas, [*datas[1:], precos.index[-1]]):
        pesos = carteiras[carteiras["data"] == inicio].set_index("ticker")["peso"]
        trecho = precos.loc[inicio:fim, list(pesos.index)]
        crescimento = trecho / trecho.iloc[0]
        bloco = valor * (crescimento @ pesos + (1.0 - pesos.sum()))
        valores.append(bloco if not valores else bloco.iloc[1:])
        valor = float(bloco.iloc[-1])
    return pd.concat(valores)


def curva_liquida(carteiras: pd.DataFrame, precos: pd.DataFrame, custo_bps: float) -> pd.Series:
    """
    ``curva_do_registro`` menos ``custo_bps`` por lado sobre o giro de cada registro, cobrado no fechamento da data
    registrada. Giro = soma de |peso novo - peso da carteira anterior já derivado pelos preços|: o 1º registro compra
    tudo, trocar a carteira inteira paga os dois lados. A sobra em caixa não paga.
    """
    curva = curva_do_registro(carteiras, precos)
    precos = precos.ffill()
    fator = pd.Series(1.0, index=curva.index)
    datas = sorted(carteiras["data"].unique())
    derivados = pd.Series(dtype=float)              # nada antes do 1º registro
    for anterior, data in zip([None, *datas[:-1]], datas):
        if anterior is not None:
            pesos = carteiras[carteiras["data"] == anterior].set_index("ticker")["peso"]
            crescimento = precos.loc[data, pesos.index] / precos.loc[anterior, pesos.index]
            derivados = pesos * crescimento / (float(pesos @ crescimento) + 1.0 - pesos.sum())
        novos = carteiras[carteiras["data"] == data].set_index("ticker")["peso"]
        giro = float(novos.sub(derivados, fill_value=0.0).abs().sum())
        fator.loc[data:] *= 1.0 - giro * custo_bps / 1e4
    return curva * fator


def rebalanceio_devido(ultima: pd.Timestamp | None, calendario: pd.DatetimeIndex, passo: int = PASSO) -> bool:
    """True sem registro ou com ``passo`` pregões do lago ou mais depois do último registro."""
    return ultima is None or int((calendario > ultima).sum()) >= passo


def decidir_do_dia(curva_satelite: pd.Series, curva_bova: pd.Series, devido: bool, parado: bool) -> Acao:
    """
    O que fazer depois do último fechamento, nesta ordem: stop já disparado (``parado``) vence tudo; perda diária de
    5% ou mais no último pregão congela as ordens, inclusive o stop (``pode_negociar``; a rotina tenta de novo no
    pregão seguinte); fora do dia devido, segura; no dia devido, stop se a razão satélite/BOVA11 caiu 25% desde o
    pico (``stop_relativo``), senão rebalanceia. ``curva_satelite`` e ``curva_bova`` nas mesmas datas, desde o 1º
    registro; vazias antes dele.
    """
    if parado:
        return Acao.PARADO
    if not pode_negociar(curva_satelite.pct_change().dropna()):
        return Acao.CONGELADO
    if not devido:
        return Acao.AGUARDAR
    return Acao.STOP if stop_relativo(curva_satelite, curva_bova) else Acao.REBALANCEAR


@dataclass(frozen=True)
class Leitura:
    """O paper do 1º registro até ``fim``: retornos acumulados (o satélite líquido de custos, sobre o capital antes da
    montagem) e a regressão contra o BOVA11, que só sai com ``MINIMO_PARA_REGRESSAO`` pregões."""

    inicio: pd.Timestamp
    fim: pd.Timestamp
    pregoes: int
    satelite: float
    bova: float
    cdi: float | None                       # None sem o CDI de todos os pregões
    comparacao: ComparacaoBenchmark | None
    stop: pd.Timestamp | None               # registro de 100% BOVA11 dentro da janela
    registros: int
    congelados: int                         # pregões de perda de 5% ou mais do satélite (curva bruta, a da rotina)
    p_supera: float                         # P[satélite > BOVA11] = PSR da diferença diária; NaN com < 2 pregões


def ler_o_paper(carteiras: pd.DataFrame, precos: pd.DataFrame, cdi_diario: pd.Series | None,
                ate: int | None = None) -> Leitura:
    """
    ``cdi_diario``: decimal ao dia por data (SGS 12); sem ele, a regressão usa taxa livre de risco zero. ``ate``: corta
    a janela nesse número de pregões depois do 1º registro (as leituras de 6 e 12 meses ficam fixas); None = até hoje.
    """
    satelite = curva_liquida(carteiras, precos, CUSTO_BPS)
    if ate is not None:
        satelite = satelite.iloc[:ate + 1]
    fim = satelite.index[-1]
    bova = precos[NUCLEO].ffill().reindex(satelite.index)
    rf = (pd.Series(dtype=float) if cdi_diario is None else cdi_diario).reindex(satelite.index[1:])
    comparacao = (comparar_benchmark(satelite, bova, rf.fillna(0.0)) if len(satelite) > MINIMO_PARA_REGRESSAO
                  else None)
    diferenca = retornos_de_valores(satelite) - retornos_de_valores(bova)
    return Leitura(inicio=satelite.index[0], fim=fim, pregoes=len(satelite) - 1,
                   satelite=float(satelite.iloc[-1] - 1.0), bova=float(bova.iloc[-1] / bova.iloc[0] - 1.0),
                   cdi=None if rf.isna().any() else float((1.0 + rf).prod() - 1.0), comparacao=comparacao,
                   stop=_data_do_stop(carteiras, fim),
                   registros=int(carteiras.loc[carteiras["data"] <= fim, "data"].nunique()),
                   congelados=_pregoes_congelados(curva_do_registro(carteiras, precos).loc[:fim]),
                   p_supera=sharpe_probabilistico(momentos_sharpe(diferenca)))


def _pregoes_congelados(curva: pd.Series) -> int:
    return int((curva.pct_change().dropna() <= -LIMITES.perda_diaria_maxima + TOLERANCIA).sum())


def _data_do_stop(carteiras: pd.DataFrame, fim: pd.Timestamp) -> pd.Timestamp | None:
    for data, carteira in carteiras[carteiras["data"] <= fim].groupby("data"):
        if carteira.set_index("ticker")["peso"].to_dict() == {NUCLEO: 1.0}:
            return pd.Timestamp(data)
    return None


def item_3_cumprido(leitura: Leitura) -> bool | None:
    """Regra pré-registrada (``scripts/paper_trading.py``): com ``LEITURA_FINAL`` pregões, cumprido se o stop não
    disparou e o satélite líquido rendeu pelo menos o BOVA11; antes disso, None."""
    if leitura.pregoes < LEITURA_FINAL:
        return None
    return leitura.stop is None and leitura.satelite >= leitura.bova
