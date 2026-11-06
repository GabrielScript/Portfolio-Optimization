"""The per-stock pack the model reads (spec, "Stage 1"): the ticker, its sector and weight, at most 8 new items
(company first, newest first; the rest wait for the next call) and at most 5 context lines. Context lines are
citable (ids ``ctx-<TICKER>-<n>``) but never make a ``grave`` by themselves (``core.sentinela.checagem``)."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from core.assessor import Evidencia
from core.sentinela.itens import MACRO, Item, _br

MAX_ITENS = 8
MAX_CONTEXTO = 5
PREFIXO_DO_CONTEXTO = "ctx-"


@dataclass(frozen=True)
class Pacote:
    ticker: str
    cabecalho: str
    itens: list[Item]
    contexto: list[str]
    pendentes: list[Item]

    def evidencias(self) -> dict[str, Evidencia]:
        """Every citable id -> its verbatim text: the items, then the context lines (class ``contexto``)."""
        evidencias = {item.id: Evidencia(item.id, item.fonte, item.data, item.texto) for item in self.itens}
        for numero, linha in enumerate(self.contexto, start=1):
            chave = f"{PREFIXO_DO_CONTEXTO}{self.ticker}-{numero}"
            evidencias[chave] = Evidencia(chave, "contexto", "", linha)
        return evidencias

    def texto(self) -> str:
        """The user message: the same bytes for the same inputs."""
        linhas = [self.cabecalho, "", "Contexto:"]
        linhas += [f"{chave} | {evidencia.texto}" for chave, evidencia in self.evidencias().items()
                   if evidencia.classe == "contexto"]
        linhas += ["", "Itens novos (id | fonte | data | texto):"] + [item.linha() for item in self.itens]
        return "\n".join(linhas)


def montar_pacote(ticker: str, setor: str, peso: float, novos: list[Item], contexto: list[str]) -> Pacote:
    ordenados = sorted(novos, key=lambda item: (item.ticker == MACRO, _negativo(item.data)))
    cabecalho = (f"Ação: {ticker}. Setor: {setor or 'sem setor'}. Peso alvo na Composite: {_br(100 * peso, 2)}%.")
    return Pacote(ticker, cabecalho, ordenados[:MAX_ITENS], list(contexto)[:MAX_CONTEXTO], ordenados[MAX_ITENS:])


def _negativo(data: str) -> tuple[int, ...]:
    """Sort key for newest first on YYYY-MM-DD strings."""
    return tuple(-int(parte) for parte in data.split("-") if parte.isdigit())


def contexto_da_acao(ticker: str, fechamentos: pd.DataFrame, registrada: pd.Timestamp, peso: float) -> list[str]:
    """Last close, 21-session return and the Composite record the name is in."""
    serie = fechamentos[ticker].dropna() if ticker in fechamentos.columns else pd.Series(dtype=float)
    linhas = []
    if not serie.empty:
        linhas.append(f"{ticker}: último fechamento R$ {_br(float(serie.iloc[-1]), 2)} em {serie.index[-1]:%d/%m/%Y}")
        if len(serie) > 21:
            retorno = float(serie.iloc[-1] / serie.iloc[-22] - 1.0)
            linhas.append(f"{ticker}: retorno de {_br(100 * retorno, 2)}% nos últimos 21 pregões")
    linhas.append(f"{ticker}: na Composite desde o registro de {registrada:%d/%m/%Y}, peso alvo {_br(100 * peso, 2)}%")
    return linhas
