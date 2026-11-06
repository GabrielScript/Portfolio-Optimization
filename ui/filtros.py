"""
filtros.py - A "Filtros" button above a table: one multiselect per category column (the user's UI list of
2026-11-03). Streamlit 1.65's dataframe sorts, searches and hides columns, but has no category filter of its own.
UI strings are Portuguese, like the rest of the app.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from functools import cache

import pandas as pd
import streamlit as st

MAXIMO_DE_VALORES = 60          # a column with more distinct values than this makes too long a list
MAXIMO_DO_TEXTO = 40            # nor does long text (headlines, subjects): no one picks a headline from a list
# a table of stocks always gets a list for the stock and one for the sector, however many values (the user,
# 2026-11-06); a table with a stock column and no sector column gets a hidden sector to filter on
COLUNAS_DE_ACAO = ("Ticker", "Ação", "Ativo")
SETOR = "Setor"


@cache
def _setores_conhecidos() -> dict[str, str]:
    """Sector of each B3 stock and B3-listed US asset the app names (``assets``, ``ativos_eua``), by ``XXXX.SA``."""
    from assets import get_ticker_info
    from ativos_eua import get_ticker_info_eua
    return {ticker: dados["setor"] for ticker, dados in {**get_ticker_info(), **get_ticker_info_eua()}.items()
            if dados.get("setor")}


def com_setor_para_filtro(tabela: pd.DataFrame,
                          setores: Mapping[str, str] | None = None) -> tuple[pd.DataFrame, str | None]:
    """``tabela`` plus a ``Setor`` column looked up from its stock column in ``setores`` (``XXXX.SA`` keys; default
    the app's registry), and that column's name; ``(tabela, None)`` when it has its own sector column or no stocks."""
    acao = next((coluna for coluna in COLUNAS_DE_ACAO if coluna in tabela.columns), None)
    if acao is None or SETOR in tabela.columns:
        return tabela, None
    setores = _setores_conhecidos() if setores is None else setores
    ticker = tabela[acao].astype(str).str.removesuffix(".SA")
    return tabela.assign(**{SETOR: (ticker + ".SA").map(setores)}), SETOR


def _de_escolha(serie: pd.Series) -> bool:
    if serie.name in (*COLUNAS_DE_ACAO, SETOR):
        return serie.dropna().nunique() >= 1
    texto = (pd.api.types.is_object_dtype(serie) or pd.api.types.is_string_dtype(serie)
             or isinstance(serie.dtype, pd.CategoricalDtype))
    if not (texto or pd.api.types.is_integer_dtype(serie)) or pd.api.types.is_bool_dtype(serie):
        return False
    valores = serie.dropna().astype(str)
    return 1 < valores.nunique() <= MAXIMO_DE_VALORES and valores.str.len().max() <= MAXIMO_DO_TEXTO


def colunas_de_escolha(tabela: pd.DataFrame) -> list[str]:
    """The columns that get a choice list: short text or whole numbers (tickers, sectors, years) with 2 to
    ``MAXIMO_DE_VALORES`` distinct values; decimals (prices, returns) and long text get none."""
    return [coluna for coluna in tabela.columns if _de_escolha(tabela[coluna])]


def aplicar_filtros(tabela: pd.DataFrame, escolhas: Mapping[str, Sequence[str]]) -> pd.DataFrame:
    """The rows whose value is one of the chosen ones in every column with a choice; an empty choice keeps all."""
    mascara = pd.Series(True, index=tabela.index)
    for coluna, valores in escolhas.items():
        if valores:
            mascara &= tabela[coluna].astype(str).isin([str(valor) for valor in valores])
    return tabela[mascara]


def filtrar(tabela: pd.DataFrame, chave: str, so_para_filtro: Iterable[str] = ()) -> pd.DataFrame:
    """Draws the "Filtros" popover above every table, one multiselect per ``colunas_de_escolha`` column (widget keys
    ``filtro_<chave>_<coluna>``), and returns the filtered rows without the ``so_para_filtro`` columns (filters that
    are not shown as columns) nor the hidden sector of ``com_setor_para_filtro``. A named index (row labels such as
    the portfolios) is filtered like a column and kept."""
    indice = [nome for nome in tabela.index.names if nome is not None]
    plana, setor = com_setor_para_filtro(tabela.reset_index() if indice else tabela)
    ocultas = [*so_para_filtro, *([setor] if setor else [])]
    colunas = colunas_de_escolha(plana)
    colunas = [*[c for c in ocultas if c in colunas], *[c for c in colunas if c not in ocultas]]
    escolhas: dict[str, list[str]] = {}
    with st.popover("Filtros", icon=":material/filter_list:"):
        if not colunas:
            st.caption("Esta tabela não tem coluna com valores para escolher.")
        for coluna in colunas:
            opcoes = sorted(plana[coluna].dropna().astype(str).unique())
            escolhas[coluna] = st.multiselect(coluna, opcoes, key=f"filtro_{chave}_{coluna}", placeholder="Todos")
    saida = aplicar_filtros(plana, escolhas).drop(columns=ocultas, errors="ignore")
    return saida.set_index(indice) if indice else saida
