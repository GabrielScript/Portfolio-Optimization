"""
formato.py - Every table in fixed Brazilian decimals (the whole-page finish review of 2026-11-06) and "—" instead of
"None" (the user, 2026-11-04). Streamlit 1.66's dataframe writes "None" for any null, even through a pandas Styler's
``na_rep``, and its printf formats write a decimal point. So a column with gaps is written as text ("23,21%" or "—",
right-aligned like numbers) and a complete column stays a number under a Styler that shows it in pt-BR, sorted by value.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping

import pandas as pd
import streamlit as st

TRACO = "—"
MENOS = "−"
# the per-year tables' first column ("2010 (desde 09/03)") was cut at the default width
COLUNA_DO_ANO = st.column_config.TextColumn("Ano", width="medium")


def _vazio(valor) -> bool:
    return valor is None or (isinstance(valor, float) and math.isnan(valor)) or pd.isna(valor)


def _numero(valor, casas: int, sinal: bool = False, sufixo: str = "", escala: float = 1.0) -> str:
    if _vazio(valor):
        return TRACO
    numero = float(valor) * escala
    texto = f"{abs(numero):,.{casas}f}".replace(",", "X").replace(".", ",").replace("X", ".") + sufixo
    if numero < 0 and texto.strip("0,.%") != "":
        return MENOS + texto
    return ("+" if sinal and numero > 0 else "") + texto


def percentual(valor, casas: int = 2) -> str:
    """A fraction in percent: 0.2321 -> "23,21%"."""
    return _numero(valor, casas, sufixo="%", escala=100.0)


def decimal(valor, casas: int = 2) -> str:
    return _numero(valor, casas)


def uma_casa(valor) -> str:
    return _numero(valor, 1)


def com_sinal(valor) -> str:
    """A signed score: "+0,45", "−0,14"."""
    return _numero(valor, 2, sinal=True)


def br(tabela: pd.DataFrame, colunas: Mapping[str, Callable], ajuda: Mapping[str, str] | None = None):
    """``(Styler, column_config)``: each of ``colunas`` written by its function; a column with a gap becomes text with
    "—", a complete one stays a number. The config carries only alignment and ``ajuda``, never a format, which would
    override the Styler."""
    ajuda = ajuda or {}
    saida, config, formatos = tabela.copy(), {}, {}
    for coluna, escrever in colunas.items():
        if coluna not in saida.columns:
            continue
        if saida[coluna].isna().any():
            saida[coluna] = [escrever(v) for v in saida[coluna]]
            config[coluna] = st.column_config.TextColumn(alignment="right", help=ajuda.get(coluna))
        else:
            formatos[coluna] = escrever
            if coluna in ajuda:
                config[coluna] = st.column_config.NumberColumn(help=ajuda[coluna])
    return saida.style.format(formatos), config
