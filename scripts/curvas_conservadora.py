"""
Saves the daily value curves of the Conservadora and BOVA11 from the frozen rerun (`scripts/rerun_conservadora.py`,
inputs `docs/dados/2026-10-22/`, prices cut at 16/10/2026), for the Fundo Alepha screen's results by period and
calendar year (`core/periodos.py`). Descriptive: the registered configuration, run as registered; N unchanged
(DECISOES 122 and 124). The rerun's text output goes to the second file and must match
`docs/conservadora-vol-2026-10-23.txt` apart from the CDI/Selic source lines.

Usage: .venv/Scripts/python -X utf8 scripts/curvas_conservadora.py
(writes docs/dados/conservadora-curvas-2026-10-23.parquet and docs/conservadora-curvas-2026-10-23.txt)
"""

from __future__ import annotations

import contextlib
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path

import pandas as pd

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))

CONSERVADORA = "Conservadora (15 nomes, pesos iguais, com regras)"     # scripts/avaliar_conservadora.CONSERVADORA
DESTINO = RAIZ / "docs" / "dados" / "conservadora-curvas-2026-10-23.parquet"
SAIDA = RAIZ / "docs" / "conservadora-curvas-2026-10-23.txt"


def quadro_das_curvas(capturadas: Sequence[tuple[int, Mapping[str, pd.Series]]]) -> pd.DataFrame:
    """One row per (phase, session): the Conservadora's and BOVA11's daily values."""
    partes = [pd.DataFrame({"fase": fase, "date": curvas[CONSERVADORA].index,
                            "Conservadora": curvas[CONSERVADORA].to_numpy(),
                            "BOVA11": curvas["BOVA11"].reindex(curvas[CONSERVADORA].index).to_numpy()})
              for fase, curvas in capturadas]
    return pd.concat(partes, ignore_index=True)


def main() -> int:
    from scripts import avaliar_conservadora, rerun_conservadora
    capturadas = []
    curvas_da_fase = avaliar_conservadora.curvas_da_fase

    def capturar(dados, mensal, fase, cdi, cfg):
        curvas = curvas_da_fase(dados, mensal, fase, cdi, cfg)
        capturadas.append((fase, {nome: curva.valores for nome, curva in curvas.items()}))
        return curvas
    avaliar_conservadora.curvas_da_fase = capturar
    with SAIDA.open("w", encoding="utf-8", newline="\n") as saida, contextlib.redirect_stdout(saida):
        codigo = rerun_conservadora.main()
    quadro_das_curvas(capturadas).to_parquet(DESTINO, index=False)
    print(f"{DESTINO.relative_to(RAIZ)}: fases {[fase for fase, _ in capturadas]}; saída em {SAIDA.relative_to(RAIZ)}")
    return codigo


if __name__ == "__main__":
    raise SystemExit(main())
