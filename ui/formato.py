"""
formato.py - Number columns with gaps (the user, 2026-11-04: "—" instead of "None"). Streamlit 1.66's dataframe writes
"None" for any null, even through a pandas Styler's ``na_rep``, and has no option for it. So a column with gaps is
written as text ("23,21%" or "—", right-aligned like numbers) and a complete column stays a number, sorted by value.
"""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd
import streamlit as st

TRACO = "—"


def _brasileiro(valor: float, porcento: bool) -> str:
    return (f"{valor * 100:.2f}%" if porcento else f"{valor:.2f}").replace(".", ",")


def com_traco(tabela: pd.DataFrame, colunas: Sequence[str], porcento: bool) -> tuple[pd.DataFrame, dict]:
    """``tabela`` with each of ``colunas`` that has a gap as text, and the ``column_config`` for all of them."""
    saida, config = tabela.copy(), {}
    for coluna in colunas:
        if saida[coluna].isna().any():
            saida[coluna] = [TRACO if pd.isna(v) else _brasileiro(float(v), porcento) for v in saida[coluna]]
            config[coluna] = st.column_config.TextColumn(alignment="right")
        else:
            config[coluna] = st.column_config.NumberColumn(format="percent" if porcento else "%.2f")
    return saida, config
