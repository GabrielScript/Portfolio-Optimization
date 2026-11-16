"""
One book run as equal staggered sleeves (spec ``docs/superpowers/specs/2026-11-16-composto-tercos-design.md``).

Each sleeve starts with an equal share of the capital, earns the CDI until its own first session, then follows its
own curve; the sleeves are never rebalanced against each other.
"""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd


def livro_em_partes(sleeves: Sequence[pd.Series], acumulado_cdi: pd.Series) -> pd.Series:
    """
    Book value on ``acumulado_cdi``'s dates, starting at 1.

    ``sleeves``: each sleeve's value curve (any scale), starting on or after the book's first date and running to its
    last. ``acumulado_cdi``: the CDI accumulation G(t) on the book's dates.
    """
    indice = acumulado_cdi.index
    g = acumulado_cdi / acumulado_cdi.iloc[0]
    partes = []
    for curva in sleeves:
        inicio = curva.index[0]
        if inicio < indice[0] or curva.index[-1] != indice[-1]:
            raise ValueError(f"sleeve {inicio:%d/%m/%Y}-{curva.index[-1]:%d/%m/%Y} is outside the book's dates")
        propria = (curva / curva.iloc[0] * g.loc[inicio]).reindex(indice)
        partes.append(propria.where(indice >= inicio, g))
    if any(p.isna().any() for p in partes):
        raise ValueError("a sleeve misses some of the book's dates after its start")
    return sum(partes) / len(partes)
