"""
lote3_b3.py - The one reading of three optimizers fed the Composite's alpha on the B3
(docs/superpowers/specs/2026-11-05-lote3-composto-design.md). Runs only after the pre-registration commit (`3c857f6`). No network.

    python -X utf8 scripts/lote3_b3.py > D:/b3-datalake/exports/lote3/b3/leitura.txt

Same engine, phases, frozen data and gate as ``lote2_b3``; writes docs/dados/lote3-{anos,portao}-b3.parquet.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.estrategias_lote3 import bracos_lote3  # noqa: E402
from scripts.lote2_b3 import ler  # noqa: E402

PASTA = Path("D:/b3-datalake/exports/lote3/b3")

if __name__ == "__main__":
    raise SystemExit(ler(sys.argv[1:], bracos_lote3, PASTA, "lote3"))
