"""
setorial.py - Sector-relative scores (spec ``docs/superpowers/specs/2026-11-16-composto-setorial-design.md``): a raw
score becomes each name's percentile rank within its own sector, so a structurally cheap sector is not read as a cheap
firm.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence

import pandas as pd

MINIMO_POR_SETOR = 5

NotaDoMes = Callable[[pd.Timestamp, Sequence[str]], pd.Series]


def dentro_do_setor(nota_do_mes: NotaDoMes, setores: Mapping[str, str],
                    minimo: int = MINIMO_POR_SETOR) -> NotaDoMes:
    """``nota_do_mes`` turned into percentile ranks within each sector among the names with a score; a sector with
    fewer than ``minimo`` scored names, and a name without a sector, keep the percentile rank among all scored names."""
    def _nota(mes: pd.Timestamp, candidatos: Sequence[str]) -> pd.Series:
        bruta = nota_do_mes(mes, candidatos).dropna()
        geral = bruta.rank(pct=True)
        setor = pd.Series({papel: setores.get(papel) or "" for papel in bruta.index}, dtype=object)
        tamanho = setor.map(setor.value_counts())
        dentro = bruta.groupby(setor).rank(pct=True)
        usa_setor = (setor != "") & (tamanho >= minimo)
        return dentro.where(usa_setor, geral)
    return _nota
