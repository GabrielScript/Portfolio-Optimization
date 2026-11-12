"""
Freeze the two DRL v3 inputs that come from the house engine (docs/superpowers/specs/2026-11-12-drl-v3-design.md):

* ``composite_score.parquet``: the Composite's score (Conservadora + value + profit growth, attempt 49) of every
  month and name of the frozen universe. A month's score uses nothing from that month on.
* ``reference_curves.parquet``: the Composite baseline curve (phase 0) and BOVA11, daily.

Run with the house code of the main checkout as the working directory, so ``core`` and ``scripts`` are the current
ones (this branch's copies predate the refiled-statements fix):

    cd <main checkout> && python -X utf8 <this file> <output folder> <curves parquet>
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import pandas as pd

from scripts import avaliar_conservadora, avaliar_robustez
from scripts.qpo_montar import FOLDER
from scripts.qubo_universo_avaliar import _value_score

FILES = ("composite_score.parquet", "reference_curves.parquet")


def scores() -> pd.DataFrame:
    avaliar_robustez.PASTA_PRECOS = avaliar_robustez.PASTA_UNIVERSO = FOLDER
    dados = avaliar_robustez.carregar()
    mensal = avaliar_conservadora.carregar_mensal(FOLDER)
    nota, _ = _value_score(mensal, com_crescimento=True)
    universo = dados.universo.assign(month=pd.to_datetime(dados.universo["month"]))
    partes = []
    for mes, grupo in universo.groupby("month"):
        valores = nota(mes, sorted(grupo["security"])).dropna()
        partes.append(pd.DataFrame({"month": mes, "security": valores.index, "score": valores.to_numpy(dtype=float)}))
    return pd.concat(partes, ignore_index=True)


def main(argv: list[str]) -> int:
    destino, curvas = Path(argv[0]), Path(argv[1])
    destino.mkdir(parents=True, exist_ok=True)
    notas = scores()
    notas.to_parquet(destino / FILES[0], index=False)
    referencias = pd.read_parquet(curvas, columns=["date", "Composite", "BOVA11"])
    referencias.to_parquet(destino / FILES[1], index=False)
    somas = "".join(f"{hashlib.sha256((destino / nome).read_bytes()).hexdigest()}  {nome}\n" for nome in FILES)
    (destino / "SHA256SUMS").write_bytes(somas.encode("ascii"))
    por_mes = notas.groupby("month").size()
    print(f"{len(notas):,} scores, months {por_mes.index.min():%Y-%m} to {por_mes.index.max():%Y-%m}, "
          f"names per month {por_mes.min()}-{por_mes.max()}; curves {referencias['date'].min()} to "
          f"{referencias['date'].max()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
