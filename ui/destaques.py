"""
destaques.py - The headline numbers beside "O que é o Aleph Quant" (the structural redesign of 2026-11-06, from the
Claude Design comp): the Composite's return a year and worst fall in the survivorship-free comparison, the Composite
account's balance and the number of stocks in the registered Composite. Each number comes from the same file a section
below shows and carries its source; a missing file drops its tile. UI strings are Portuguese.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from core.metrics import cagr, drawdown

MENOS = "−"


@dataclass(frozen=True)
class Numero:
    rotulo: str
    valor: float
    texto: str
    fonte: str


def _br(valor: float, casas: int) -> str:
    texto = f"{abs(valor):,.{casas}f}".replace(",", "X").replace(".", ",").replace("X", ".")
    return (MENOS if valor < 0 else "") + texto


def numeros(comparacao: pd.DataFrame | None, conta: float | None, acoes: int | None) -> list[Numero]:
    """``comparacao``: dates x curves of ``scripts/comparacao_estrategias.py`` (needs ``Composite``); ``conta``: the
    Composite account's balance in R$; ``acoes``: stocks in the registered Composite."""
    saida: list[Numero] = []
    if comparacao is not None and "Composite" in comparacao.columns:
        curva = comparacao["Composite"].dropna()
        ao_ano, queda = cagr(curva), drawdown(curva)[1]          # the tables' convention (core.metrics)
        periodo = f"{curva.index[0]:%Y} a {curva.index[-1]:%Y}"
        fonte = f"Backtest sem viés de sobrevivência, depois de custos, {periodo}"
        saida += [Numero("Composite ao ano", ao_ano, _br(ao_ano * 100, 1) + "%", fonte),
                  Numero("Queda máxima", queda, _br(queda * 100, 1) + "%", fonte)]
    if conta is not None:
        saida.append(Numero("Conta Composite (R$)", conta, _br(conta, 2),       # unit in the label: fits the tile
                            "Conta fictícia de R$ 10.000 desde 03/11/2026, com custos"))
    if acoes is not None:
        saida.append(Numero("Ações na carteira", float(acoes), str(acoes), "A Composite registrada, pesos iguais"))
    return saida


def legenda(comparacao: pd.DataFrame | None) -> str:
    """The tiles' period and terms, on screen (a real-money tool's first viewport keeps them visible)."""
    conta = "Conta fictícia de R$ 10.000 desde 03/11/2026."
    if comparacao is None or "Composite" not in comparacao.columns:
        return conta
    curva = comparacao["Composite"].dropna()
    return (f"Backtest sem viés de sobrevivência de {curva.index[0]:%d/%m/%Y} a {curva.index[-1]:%d/%m/%Y}, "
            f"depois de custos e antes do Imposto de Renda. {conta}")


def mostrar(itens: list[Numero], nota: str = "") -> None:
    """Two by two tiles, the source in each tile's help, the period and terms under them."""
    import streamlit as st

    for linha in range(0, len(itens), 2):
        colunas = st.columns(2)
        for coluna, numero in zip(colunas, itens[linha:linha + 2]):
            coluna.metric(numero.rotulo, numero.texto, help=numero.fonte, border=True)
    if nota:
        st.caption(nota)
