"""
focus.py - Expectativas de mercado (Focus/BCB) via OData do Olinda, sem Streamlit.

Peculiaridades da API (sondadas ao vivo em 2026-10-12 e 2026-10-13):
- O ``$filter`` vai na URL já codificado com ``quote(v, safe="")``. Passar os
  parâmetros pelo ``params`` do requests devolve HTTP 400.
- O histórico inteiro de um indicador vem numa resposta só (PIB Total: 32.740
  linhas, 3,6 MB, ~2 s). Não há paginação.
- ``Data`` é a data da pesquisa (aaaa-mm-dd). ``DataReferencia`` é o ano, em
  texto. Pesquisas antigas têm ``Minimo``/``Maximo`` nulos; ``Mediana`` e
  ``DesvioPadrao`` existem desde o início (PIB: 1999-07; IPCA 12m: 2001-12).
- Indicadores anuais usados: "Selic", "Câmbio" (R$/US$ no fim do ano) e
  "PIB Total". A inflação 12 meses à frente vem de um endpoint próprio
  (``Suavizada='S'``).

``hoje`` é sempre explícito: pesquisas com ``Data > hoje`` são descartadas.
"""

from __future__ import annotations

from datetime import date
from typing import NamedTuple
from urllib.parse import quote

import pandas as pd
import requests

URL_BASE = "https://olinda.bcb.gov.br/olinda/servico/Expectativas/versao/v1/odata/"
ENDPOINT_ANUAIS = "ExpectativasMercadoAnuais"
ENDPOINT_IPCA_12M = "ExpectativasMercadoInflacao12Meses"
INDICADOR_SELIC = "Selic"
INDICADOR_CAMBIO = "Câmbio"
INDICADOR_PIB = "PIB Total"
_TIMEOUT_HTTP = 60
_CAMPOS_ANUAIS = {"Data": "data", "DataReferencia": "ano_referencia",
                  "Mediana": "mediana", "DesvioPadrao": "desvio"}
_CAMPOS_IPCA_12M = {"Data": "data", "Mediana": "mediana", "DesvioPadrao": "desvio"}


class ErroFocus(RuntimeError):
    """Falha ao obter ou interpretar expectativas do Focus."""


class Expectativa12m(NamedTuple):
    """Mediana e desvio-padrão dos analistas 12 meses à frente, na pesquisa de ``data``."""

    mediana: float
    desvio: float          # NaN quando a pesquisa não traz desvio
    data: pd.Timestamp


def _url(endpoint: str, filtro: str, campos: dict[str, str]) -> str:
    return (f"{URL_BASE}{endpoint}?$format=json&$select={','.join(campos)}"
            f"&$filter={quote(filtro, safe='')}")


def _consultar(endpoint: str, filtro: str, campos: dict[str, str]) -> pd.DataFrame:
    """Linhas do OData com os campos renomeados. Levanta ErroFocus em falha de rede ou de formato."""
    try:
        resposta = requests.get(_url(endpoint, filtro, campos), timeout=_TIMEOUT_HTTP)
        resposta.raise_for_status()
        registros = resposta.json()["value"]
    except (requests.RequestException, ValueError, KeyError, TypeError) as erro:
        raise ErroFocus(f"Focus {endpoint} ({filtro}): {erro}") from erro
    bruto = pd.DataFrame.from_records(registros)
    faltando = sorted(set(campos) - set(bruto.columns))
    if faltando:
        raise ErroFocus(f"Focus {endpoint} ({filtro}) sem os campos {faltando}.")
    return bruto[list(campos)].rename(columns=campos)


def _limpar(tabela: pd.DataFrame, hoje: date) -> pd.DataFrame:
    """Converte tipos, descarta linhas sem data/mediana e pesquisas posteriores a ``hoje``."""
    tabela = tabela.copy()
    tabela["data"] = pd.to_datetime(tabela["data"], format="%Y-%m-%d", errors="coerce")
    for coluna in ("mediana", "desvio"):
        tabela[coluna] = pd.to_numeric(tabela[coluna], errors="coerce")
    tabela = tabela.dropna(subset=["data", "mediana"])
    tabela = tabela[tabela["data"] <= pd.Timestamp(hoje)]
    if tabela.empty:
        raise ErroFocus(f"Focus sem pesquisas válidas até {hoje}.")
    return tabela


def baixar_focus_anual(indicador: str, hoje: date) -> pd.DataFrame:
    """
    Pesquisas diárias de um indicador anual, até ``hoje``.

    Colunas: ``data`` (Timestamp), ``ano_referencia`` (int), ``mediana`` e
    ``desvio`` (float, nas unidades do Focus: % a.a. para Selic e PIB, R$/US$
    para o câmbio). Ordenadas por (data, ano_referencia).
    """
    filtro = f"Indicador eq '{indicador}' and baseCalculo eq 0"
    tabela = _consultar(ENDPOINT_ANUAIS, filtro, _CAMPOS_ANUAIS)
    tabela["ano_referencia"] = pd.to_numeric(tabela["ano_referencia"], errors="coerce")
    tabela = _limpar(tabela.dropna(subset=["ano_referencia"]), hoje)
    tabela["ano_referencia"] = tabela["ano_referencia"].astype(int)
    return (tabela.sort_values(["data", "ano_referencia"])
            .drop_duplicates(["data", "ano_referencia"], keep="last")
            .reset_index(drop=True))


def baixar_focus_ipca_12m(hoje: date) -> pd.DataFrame:
    """IPCA 12 meses à frente (série suavizada), até ``hoje``. Colunas: ``data``, ``mediana``, ``desvio``."""
    filtro = "Indicador eq 'IPCA' and Suavizada eq 'S' and baseCalculo eq 0"
    tabela = _limpar(_consultar(ENDPOINT_IPCA_12M, filtro, _CAMPOS_IPCA_12M), hoje)
    return tabela.sort_values("data").drop_duplicates("data", keep="last").reset_index(drop=True)


def peso_ano_corrente(data: pd.Timestamp | pd.Period) -> float:
    """
    Peso do ano corrente numa projeção 12 meses à frente: w = (12 - k)/12, com
    k = meses completos do ano na data. Janeiro: k = 0, w = 1. Outubro: k = 9, w = 3/12.
    """
    return (13 - data.month) / 12.0


def expectativa_12m(anual: pd.DataFrame, ate: pd.Timestamp) -> Expectativa12m:
    """
    Projeção 12 meses à frente na última pesquisa com ``data <= ate``:
    ``w·F(Y) + (1 - w)·F(Y+1)``, Y = ano da pesquisa. O desvio usa a mesma
    ponderação, o que supõe correlação 1 entre os dois anos (conservador).

    Levanta ErroFocus se não houver pesquisa até ``ate`` ou se ela não tiver
    o ano Y (ou o Y+1, quando o peso dele é positivo).
    """
    anteriores = anual[anual["data"] <= ate]
    if anteriores.empty:
        raise ErroFocus(f"Nenhuma pesquisa Focus até {ate:%Y-%m-%d}.")
    data = anteriores["data"].max()
    pesquisa = anteriores[anteriores["data"] == data].set_index("ano_referencia")
    peso = peso_ano_corrente(data)
    anos = [data.year] + ([data.year + 1] if peso < 1.0 else [])
    faltando = [ano for ano in anos if ano not in pesquisa.index]
    if faltando:
        raise ErroFocus(f"Pesquisa Focus de {data:%Y-%m-%d} sem os anos {faltando}.")
    if peso == 1.0:
        linha = pesquisa.loc[data.year]
        return Expectativa12m(float(linha["mediana"]), float(linha["desvio"]), data)
    atual, seguinte = pesquisa.loc[data.year], pesquisa.loc[data.year + 1]
    return Expectativa12m(
        mediana=float(peso * atual["mediana"] + (1 - peso) * seguinte["mediana"]),
        desvio=float(peso * atual["desvio"] + (1 - peso) * seguinte["desvio"]),
        data=data,
    )


def mediana_mensal_por_ano(anual: pd.DataFrame) -> pd.DataFrame:
    """
    Mediana da última pesquisa de cada mês, por ano de referência.
    Índice: PeriodIndex mensal ``mes``. Colunas: anos de referência (int).
    """
    ordenada = anual.sort_values("data")
    mes = ordenada["data"].dt.to_period("M").rename("mes")
    return ordenada.groupby([mes, "ano_referencia"])["mediana"].last().unstack("ano_referencia")


def mediana_mensal(serie_12m: pd.DataFrame) -> pd.Series:
    """Mediana da última pesquisa de cada mês (IPCA 12m). Índice: PeriodIndex mensal ``mes``."""
    ordenada = serie_12m.sort_values("data")
    mes = ordenada["data"].dt.to_period("M").rename("mes")
    return ordenada.groupby(mes)["mediana"].last()
