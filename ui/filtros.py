"""
filtros.py - A "Filtros" button above a table: one multiselect per category column (the user's UI list of
2026-11-03). Streamlit 1.65's dataframe sorts, searches and hides columns, but has no category filter of its own.
UI strings are Portuguese, like the rest of the app.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence

import pandas as pd
import streamlit as st

MAXIMO_DE_VALORES = 60          # a text column with more distinct values than this (tickers, titles) is no category


def colunas_de_categoria(tabela: pd.DataFrame) -> list[str]:
    """The text columns whose values repeat: at most ``MAXIMO_DE_VALORES`` distinct ones and fewer than the rows."""
    colunas = []
    for coluna in tabela.columns:
        serie = tabela[coluna]
        if not (pd.api.types.is_object_dtype(serie) or pd.api.types.is_string_dtype(serie)
                or isinstance(serie.dtype, pd.CategoricalDtype)):
            continue
        distintos = serie.dropna().astype(str).nunique()
        if 1 < distintos <= MAXIMO_DE_VALORES and distintos < len(serie):
            colunas.append(coluna)
    return colunas


def aplicar_filtros(tabela: pd.DataFrame, escolhas: Mapping[str, Sequence[str]]) -> pd.DataFrame:
    """The rows whose value is one of the chosen ones in every column with a choice; an empty choice keeps all."""
    mascara = pd.Series(True, index=tabela.index)
    for coluna, valores in escolhas.items():
        if valores:
            mascara &= tabela[coluna].astype(str).isin([str(valor) for valor in valores])
    return tabela[mascara]


def filtrar(tabela: pd.DataFrame, chave: str, so_para_filtro: Iterable[str] = ()) -> pd.DataFrame:
    """Draws the "Filtros" popover over ``tabela``'s category columns (widget keys ``filtro_<chave>_<coluna>``) and
    returns the filtered rows without the ``so_para_filtro`` columns (filters that are not shown as columns)."""
    ocultas = list(so_para_filtro)
    colunas = colunas_de_categoria(tabela)
    colunas = [*[c for c in ocultas if c in colunas], *[c for c in colunas if c not in ocultas]]
    escolhas: dict[str, list[str]] = {}
    if colunas:
        with st.popover("Filtros", icon=":material/filter_list:"):
            for coluna in colunas:
                opcoes = sorted(tabela[coluna].dropna().astype(str).unique())
                escolhas[coluna] = st.multiselect(coluna, opcoes, key=f"filtro_{chave}_{coluna}",
                                                  placeholder="Todos")
    return aplicar_filtros(tabela, escolhas).drop(columns=ocultas, errors="ignore")
