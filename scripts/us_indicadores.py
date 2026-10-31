"""
us_indicadores.py - The screen's US indicators, from the lake's gold: each universe name's market value, risk and multiples
and the macro's last readings. Derived numbers only, no provider's price, like every US file the app reads (DECISOES 195).

    python -X utf8 scripts/us_indicadores.py

Writes docs/dados/us-indicadores.parquet and docs/dados/us-macro.parquet.
"""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.us_indicadores import tabela_de_indicadores, tabela_macro  # noqa: E402

RAIZ = Path(__file__).resolve().parent.parent
LAGO_US = Path("D:/b3-datalake/exports/us_stocks")
SAIDA = RAIZ / "docs" / "dados"


def gerar(pasta: Path, hoje: date) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The indicators of the last universe month and the macro table, from the gold in ``pasta``."""
    retornos = pd.read_parquet(pasta / "monthly_total_returns.parquet")
    retornos = retornos.set_index(pd.DatetimeIndex(pd.to_datetime(retornos.pop("month")))).astype("float64")
    fatores = pd.read_parquet(pasta / "factors_monthly.parquet")
    por_mes = fatores.set_index(pd.DatetimeIndex(pd.to_datetime(fatores["month"])))
    mercado = por_mes["mkt_rf"] + por_mes["rf"]
    universo = pd.read_parquet(pasta / "universe_monthly.parquet")
    ultimo = universo["month"].max()
    fundamentos = pd.read_parquet(pasta / "fundamentals.parquet",
                                  columns=["issuer", "account", "dt_refer", "value", "available_from", "sector"])
    for coluna in ("dt_refer", "available_from"):
        fundamentos[coluna] = pd.to_datetime(fundamentos[coluna])
    tabela = tabela_de_indicadores(retornos, mercado, universo.loc[universo["month"] == ultimo, "security"].tolist(),
                                   pd.read_parquet(pasta / "securities.parquet"), fundamentos,
                                   pd.read_parquet(pasta / "market_caps.parquet", columns=["issuer", "month", "market_cap"]),
                                   pd.read_parquet(pasta / "dividends.parquet", columns=["security", "ex_date", "yield"]), hoje)
    return tabela, tabela_macro(pd.read_parquet(pasta / "macro_monthly.parquet"), fatores)


def main() -> int:
    tabela, macro = gerar(LAGO_US, date.today())
    tabela.to_parquet(SAIDA / "us-indicadores.parquet", index=False)
    macro.to_parquet(SAIDA / "us-macro.parquet", index=False)
    print(f"docs/dados/us-indicadores.parquet: {len(tabela)} ações; docs/dados/us-macro.parquet: {len(macro)} indicadores")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
