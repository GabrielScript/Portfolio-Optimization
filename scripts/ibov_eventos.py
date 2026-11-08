"""
Freezes the Ibovespa's monthly compositions and rebalance events from the lake's CVM index-fund holdings (spec
``docs/superpowers/specs/2026-11-08-ibov-recomposicao-design.md``, data item 1). No price is read here.

Usage: python -X utf8 scripts/ibov_eventos.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))

from core import recomposicao as rc  # noqa: E402

LAGO = Path("D:/b3-datalake/exports")
SAIDA = RAIZ / "docs" / "dados" / "ibov-recomposicao"
PRINCIPAIS = [("10.406.511/0001-61", "2008-11-30", "2018-08-31"),      # BOVA11
              ("41.901.871/0001-63", "2021-07-31", "2026-12-31")]      # fully public index fund
OUTROS = ["04.699.738/0001-40", "07.096.546/0001-37", "03.370.003/0001-05", "05.589.418/0001-08",
          "03.394.711/0001-86", "08.973.918/0001-29"]               # index funds matching the IBOV of 2026-11


def aliases() -> dict[str, str]:
    """Every ticker the lake has seen for a security -> that security."""
    titulos = pd.read_parquet(LAGO / "securities.parquet", columns=["security", "tickers"])
    return {str(ticker).strip(): seguranca for seguranca, tickers in zip(titulos["security"], titulos["tickers"])
            for ticker in tickers}


def main() -> int:
    cnpjs = [cnpj for cnpj, _, _ in PRINCIPAIS] + OUTROS
    posicoes = pd.read_parquet(LAGO / "fund_stock_holdings.parquet", filters=[("cnpj", "in", cnpjs)],
                               columns=["cnpj", "month", "ticker"])
    posicoes["month"] = pd.to_datetime(posicoes["month"])
    posicoes = posicoes[posicoes["month"] >= "2008-11-30"]
    meses = rc.composicoes(posicoes, PRINCIPAIS, OUTROS, aliases())
    eventos = rc.eventos(meses)
    SAIDA.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{"mes": mes, "papel": papel} for mes, nomes in meses.items() for papel in sorted(nomes)]
                 ).to_csv(SAIDA / "composicoes.csv", index=False, lineterminator="\n", date_format="%Y-%m-%d")
    eventos.to_csv(SAIDA / "eventos.csv", index=False, lineterminator="\n", date_format="%Y-%m-%d")
    resumo = eventos.pivot_table(index="rebalanceamento", columns="tipo", values="papel", aggfunc="count",
                                 fill_value=0)
    tamanho = pd.Series({mes: len(nomes) for mes, nomes in meses.items()})
    print(f"{len(meses)} months, {tamanho.min()}-{tamanho.max()} names; {len(eventos)} events, "
          f"{(eventos['tipo'] == 'entrada').sum()} entrants")
    print(resumo.to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
