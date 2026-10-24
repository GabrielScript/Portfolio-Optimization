"""
The deployed app's news/sentiment and filings panel files, from the lake's forward log and CVM IPE filings on D:
(``core.sentimento``): ``lago/sentimento_diario.parquet`` (counts and Jev means per Brasília day, numbers only) and
``lago/fatos_relevantes.parquet`` (the material CVM filings of the app's stocks in the last ``DIAS_DE_FATOS`` days,
public CVM data) and ``lago/fundamentos.parquet`` (multiples and 12-month dividend yield of each stock, from the
lake's CVM statements). No headline or summary text goes to git: the raw text stays on D: (DECISOES 123); the local app
reads the headlines from there. The daily upkeep (``scripts/manutencao.py``) runs this and commits both files.

Usage: .venv/Scripts/python -X utf8 scripts/painel_do_app.py [lake sentiment folder, default D:/b3-datalake/sentiment]
"""

from __future__ import annotations

import sys
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))

PASTA_DO_LOG = Path("D:/b3-datalake/sentiment")
EXPORTS_DO_LAGO = RAIZ / "data" / "b3lake"            # the lake's exports in the app checkout (not in git)
DESTINO_DIARIO = RAIZ / "lago" / "sentimento_diario.parquet"
DESTINO_FATOS = RAIZ / "lago" / "fatos_relevantes.parquet"
DESTINO_FUNDAMENTOS = RAIZ / "lago" / "fundamentos.parquet"
DIAS_DE_FATOS = 90
INICIO_DO_LOG = date(2026, 10, 23)          # the log's first run (DECISOES 123)


def gerar(pasta: Path, exports: Path, tickers: list[str], hoje: date) -> tuple[pd.DataFrame, pd.DataFrame,
                                                                             pd.DataFrame]:
    """The panel tables: the daily tone since the log began, the recent material filings of ``tickers`` and their
    fundamentals (``core.indicadores.tabela_de_fundamentos``, from the lake's exports in ``exports``)."""
    from core.indicadores import tabela_de_fundamentos
    from core.sentimento import fatos_recentes, ler_log, tom_diario
    itens, notas = ler_log(pasta, INICIO_DO_LOG)
    diario = tom_diario(itens, notas).reset_index()
    ipe = pd.read_parquet(pasta / "ipe" / "ipe.parquet", columns=["codigo_cvm", "nome_companhia", "categoria",
                                                                  "assunto", "data_entrega", "link_download"])
    fatos = fatos_recentes(ipe, pd.read_parquet(exports / "sectors.parquet", columns=["issuer", "code_cvm"]),
                           tickers, hoje - timedelta(days=DIAS_DE_FATOS))
    fundamentos = pd.read_parquet(exports / "fundamentals.parquet",
                                  columns=["issuer", "account", "dt_refer", "value", "available_from", "sector"])
    for coluna in ("dt_refer", "available_from"):
        fundamentos[coluna] = pd.to_datetime(fundamentos[coluna])
    tabela = tabela_de_fundamentos(fundamentos, pd.read_parquet(exports / "market_caps.parquet",
                                                                columns=["issuer", "month", "market_cap"]),
                                   pd.read_parquet(exports / "dividends.parquet",
                                                   columns=["security", "ex_date", "yield"]), tickers, hoje)
    return diario, fatos, tabela


def main() -> int:
    from assets import get_all_tickers
    pasta = Path(sys.argv[1]) if len(sys.argv) > 1 else PASTA_DO_LOG
    diario, fatos, fundamentos = gerar(pasta, EXPORTS_DO_LAGO, get_all_tickers(), date.today())
    diario.to_parquet(DESTINO_DIARIO)
    fatos.to_parquet(DESTINO_FATOS)
    fundamentos.to_parquet(DESTINO_FUNDAMENTOS)
    print(f"{DESTINO_DIARIO.relative_to(RAIZ)}: {len(diario)} dias; {DESTINO_FATOS.relative_to(RAIZ)}: {len(fatos)} "
          f"fatos relevantes e comunicados; {DESTINO_FUNDAMENTOS.relative_to(RAIZ)}: {len(fundamentos)} ações")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
