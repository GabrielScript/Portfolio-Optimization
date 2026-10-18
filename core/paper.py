"""
paper.py - Paper trading (S5): o que o registro diz e o que fazer no pregão.

O ``paper/carteiras.csv`` (``core.registro``) guarda só o satélite: pesos que somam até 100% do dinheiro do satélite,
decididos no fechamento da data registrada e valendo a partir do pregão seguinte. O núcleo é o BOVA11, fixo pela regra
do dinheiro real até o critério de pronto (PLAN, "Regras de risco fixas"), então não é registrado. Daqui saem a curva
do satélite e a decisão do dia; ler o lago, otimizar e gravar ficam no ``scripts/paper_trading.py``.
"""

from __future__ import annotations

import pandas as pd

PASSO = 63          # pregões entre rebalanceios, o do walk-forward (trimestral)


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


def rebalanceio_devido(ultima: pd.Timestamp | None, calendario: pd.DatetimeIndex, passo: int = PASSO) -> bool:
    """True sem registro ou com ``passo`` pregões do lago ou mais depois do último registro."""
    return ultima is None or int((calendario > ultima).sum()) >= passo
