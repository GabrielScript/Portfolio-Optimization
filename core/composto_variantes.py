"""
composto_variantes.py - Variants of the Composite's score for the pre-registered test
``docs/superpowers/specs/2026-11-11-composite-variants-design.md``. Research only: the live Composite
(``core.qubo_universo.composite_score``, ``core.composto``) is not changed.
"""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd

from core.valor import NotaDoMes
from core.views import escore_normal


def neutral_composite_score(conservadora_of_month: NotaDoMes, *others_of_month: NotaDoMes) -> NotaDoMes:
    """The Composite's score with a missing component counted as a normal score of 0: over the names with a
    Conservadora score, (its normal score + each other signal's normal score among those names, 0 where the name has
    none) / number of signals. The house score averages only the components a name has, which gives a name with
    fewer components a wider score distribution."""
    def _nota(mes: pd.Timestamp, candidatos: Sequence[str]) -> pd.Series:
        base = conservadora_of_month(mes, candidatos).dropna()
        escores = pd.DataFrame({0: escore_normal(base),
                                **{posicao: escore_normal(outra(mes, list(base.index)).dropna())
                                   for posicao, outra in enumerate(others_of_month, start=1)}})
        return escores.reindex(base.index).fillna(0.0).mean(axis=1)
    return _nota
