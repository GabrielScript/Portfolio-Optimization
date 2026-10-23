"""
The app's liquidity snapshot: the last closed month of the lake's `monthly_metrics.parquet` in the final proof's
measures (`scripts/avaliar_execucao.py`: half the quoted spread, daily vol, median daily traded value), written to
`lago/liquidez.parquet` (~50 KB, versioned) so the deployed app prices each order with `custo_da_ordem` and the
no-trade zone has each name's daily vol. Refresh it by hand after a month closes (the app shows its month).

Usage: .venv/Scripts/python -X utf8 scripts/liquidez_do_app.py [folder of the lake's exports, default data/b3lake]
"""

from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path

import pandas as pd

RAIZ = Path(__file__).resolve().parent.parent
ORIGEM = RAIZ / "data" / "b3lake"
DESTINO = RAIZ / "lago" / "liquidez.parquet"


def instantaneo(metricas: pd.DataFrame, ate: date) -> pd.DataFrame:
    """The rows of the last month before the month of ``ate`` (the lake's last session), one per security."""
    meses = pd.to_datetime(metricas["month"])
    fechados = meses[meses < pd.Timestamp(ate).to_period("M").to_timestamp()]
    ultimo = fechados.max()
    linhas = metricas.loc[meses == ultimo]
    return pd.DataFrame({"security": linhas["security"].astype(str).to_numpy(),
                         "month": [ultimo.date()] * len(linhas),
                         "meio_spread": linhas["quoted_spread"].astype("float64").to_numpy() / 2,
                         "vol_diaria": linhas["vol_daily"].astype("float64").to_numpy(),
                         "volume_mediano": linhas["median_dtv_brl"].astype("float64").to_numpy()})


def main() -> int:
    origem = Path(sys.argv[1]) if len(sys.argv) > 1 else ORIGEM
    ate = date.fromisoformat(json.loads((origem / "manifest.json").read_text(encoding="utf-8"))["data_through"])
    metricas = pd.read_parquet(origem / "monthly_metrics.parquet",
                               columns=["month", "security", "quoted_spread", "vol_daily", "median_dtv_brl"])
    foto = instantaneo(metricas, ate)
    foto.to_parquet(DESTINO, index=False)
    print(f"{DESTINO.relative_to(RAIZ)}: {len(foto)} papéis, mês {foto['month'].iloc[0]:%m/%Y} (lago até {ate:%d/%m/%Y})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
