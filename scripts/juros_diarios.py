"""
juros_diarios.py - The CDI (SGS 12) and the Selic (SGS 11) of every published day since 2010, in decimal per day, to
docs/dados/juros-diarios.parquet: the columns the app's year tables compare with the strategies, with no network call
from the screen. ``scripts/curvas_ao_vivo.py`` refreshes it with the curves.
"""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
if str(RAIZ) not in sys.path:
    sys.path.insert(0, str(RAIZ))

from core import bcb  # noqa: E402

JUROS = RAIZ / "docs" / "dados" / "juros-diarios.parquet"
INICIO = date(2010, 1, 1)
SERIE_SELIC_DIARIA = 11        # Selic over, % per day; core.bcb has only the target (SGS 432)


def tabela_de_juros(hoje: date) -> pd.DataFrame:
    """One row per day either rate was published, in decimal per day."""
    juros = pd.DataFrame({"CDI": bcb.baixar_serie(bcb.SERIE_CDI, INICIO, hoje),
                          "Selic": bcb.baixar_serie(SERIE_SELIC_DIARIA, INICIO, hoje)}) / 100.0
    return juros.rename_axis("date").reset_index()


def escrever(hoje: date, arquivo: Path = JUROS) -> pd.Timestamp:
    """Writes the file and returns the last day in it."""
    tabela = tabela_de_juros(hoje)
    tabela.to_parquet(arquivo, index=False)
    return pd.Timestamp(tabela["date"].max())


if __name__ == "__main__":
    print(f"{JUROS.relative_to(RAIZ)} up to {escrever(date.today()):%d/%m/%Y}")
