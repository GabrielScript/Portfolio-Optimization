"""
qml_ler_bancada.py - Summarizes the solver bench's rows for the report and the screen (docs/superpowers/specs/2026-11-01-qpo-
hibrido-design.md, section 6). Descriptive: no N, no gate.

    python -X utf8 scripts/qml_ler_bancada.py D:/b3-datalake/exports/qml/b3
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.qml_resumo import resumo_da_bancada  # noqa: E402
from scripts.qml_ler_ic import DOCS  # noqa: E402


def main(argv: list[str]) -> int:
    linhas = pd.read_csv(Path(argv[0]) / "bancada.csv")
    resumo = resumo_da_bancada(linhas)
    resumo.to_parquet(DOCS / "qpo-hibrido-bancada.parquet", index=False)
    print(resumo.to_string(index=False, float_format=lambda valor: f"{valor:.3f}"))
    print(f"\nMedian seconds per training: {linhas[linhas['segundos'] > 0].groupby(['n', 'solver'])['segundos'].median().round(1).to_dict()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
