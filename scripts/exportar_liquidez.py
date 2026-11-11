"""Cut the monthly liquidity measures of the realistic replay from the lake (spec 2026-11-11-vol-target-motor).

    python -X utf8 scripts/exportar_liquidez.py [MONTHLY_METRICS] [UNIVERSE_MONTHLY]

Same three measures as ``docs/dados/2026-10-20-liquidez`` (half the quoted spread, daily volatility, median daily
traded value), from 2009-11 to 2026-10, for the names that enter the point-in-time universe in some month since
2009-12, plus BOVA11. Writes ``liquidez.parquet`` and ``SHA256SUMS`` in ``docs/dados/2026-11-11-liquidez``."""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import pandas as pd

RAIZ = Path(__file__).resolve().parent.parent
METRICAS = RAIZ.parent / "Portfolio-Optimization" / "data" / "b3lake" / "monthly_metrics.parquet"
UNIVERSO = RAIZ / "docs" / "dados" / "2026-10-22" / "universe_monthly.parquet"
DESTINO = RAIZ / "docs" / "dados" / "2026-11-11-liquidez"
PRIMEIRO, ULTIMO = pd.Timestamp("2009-11-01"), pd.Timestamp("2026-10-01")


def cortar(metricas: pd.DataFrame, universo: pd.DataFrame) -> pd.DataFrame:
    """One row per month and name in the window, with the replay's column names."""
    meses = pd.to_datetime(universo["month"])
    nomes = set(universo.loc[meses >= PRIMEIRO + pd.DateOffset(months=1), "security"]) | {"BOVA11"}
    m = metricas.assign(month=pd.to_datetime(metricas["month"]))
    m = m.loc[m["month"].between(PRIMEIRO, ULTIMO) & m["security"].isin(nomes)]
    return pd.DataFrame({"month": m["month"].dt.date, "security": m["security"],
                         "meio_spread": m["quoted_spread"] / 2.0, "vol_diaria": m["vol_daily"],
                         "volume_mediano": m["median_dtv_brl"]}).sort_values(["month", "security"]).reset_index(drop=True)


def main(argv: list[str]) -> int:
    metricas = Path(argv[0]) if argv else METRICAS
    universo = Path(argv[1]) if len(argv) > 1 else UNIVERSO
    tabela = cortar(pd.read_parquet(metricas, columns=["month", "security", "quoted_spread", "vol_daily",
                                                       "median_dtv_brl"]), pd.read_parquet(universo))
    DESTINO.mkdir(parents=True, exist_ok=True)
    arquivo = DESTINO / "liquidez.parquet"
    tabela.to_parquet(arquivo, index=False)
    soma = hashlib.sha256(arquivo.read_bytes()).hexdigest()
    (DESTINO / "SHA256SUMS").write_text(f"{soma} *liquidez.parquet\n", encoding="ascii", newline="\n")
    origem = hashlib.sha256(metricas.read_bytes()).hexdigest()
    print(f"{len(tabela)} rows, {tabela['security'].nunique()} names, {tabela['month'].min()} to "
          f"{tabela['month'].max()}; monthly_metrics sha256 {origem}; liquidez.parquet sha256 {soma}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
