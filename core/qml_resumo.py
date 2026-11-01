"""
qml_resumo.py - The solver bench's summary (docs/superpowers/specs/2026-11-01-qpo-hibrido-design.md, section 6): median and
interquartile range per size, statistical axis, solver and stage, with raw and repaired kept apart.
"""

from __future__ import annotations

import pandas as pd

GRUPO = ["n", "eixo", "solver", "etapa"]
MEDIDAS = ["viavel", "p_otimo", "razao_media", "razao_melhor"]


def resumo_da_bancada(linhas: pd.DataFrame) -> pd.DataFrame:
    grupos = linhas.groupby(GRUPO, sort=True)
    resumo = grupos[MEDIDAS].median()
    resumo["iqr_razao_media"] = grupos["razao_media"].quantile(0.75) - grupos["razao_media"].quantile(0.25)
    resumo["casos"] = grupos.size()
    return resumo.reset_index()
