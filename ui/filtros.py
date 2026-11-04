"""
filtros.py - A "Filtros" button above a table: one multiselect per category column (the user's UI list of
2026-11-03). Streamlit 1.65's dataframe sorts, searches and hides columns, but has no category filter of its own.
UI strings are Portuguese, like the rest of the app.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence

import pandas as pd
import streamlit as st

MAXIMO_DE_VALORES = 60          # a column with more distinct values than this makes too long a list
MAXIMO_DO_TEXTO = 40            # nor does long text (headlines, subjects): no one picks a headline from a list


def _de_escolha(serie: pd.Series) -> bool:
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
    are not shown as columns). A named index (row labels such as the portfolios) is filtered like a column and kept."""
    indice = [nome for nome in tabela.index.names if nome is not None]
    plana = tabela.reset_index() if indice else tabela
    ocultas = list(so_para_filtro)
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
