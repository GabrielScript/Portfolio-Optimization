"""
lago.py - Preços diários de retorno total do lago de dados da B3 (projeto ``b3-datalake``).

O lago monta, a partir do COTAHIST e dos eventos da B3 e da CVM, um índice de retorno total
por papel ancorado no último fechamento (o último valor é o fechamento de verdade, como nos
preços ajustados do Yahoo) e o publica em ``lago/precos_diarios.parquet``: coluna ``date`` e
uma coluna por papel, com o ticker sem ``.SA``. Pregão sem negócio do papel fica vazio.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd

ARQUIVO_PRECOS = Path(__file__).resolve().parents[1] / "lago" / "precos_diarios.parquet"
SUFIXO_B3 = ".SA"


def ler_fechamentos(tickers: list[str], inicio: date, fim: date, arquivo: Path | None = None) -> pd.DataFrame:
    """Fechamentos de retorno total de ``tickers`` (formato do app, com ``.SA``) entre ``inicio`` e ``fim``.

    ``arquivo`` ausente = ``ARQUIVO_PRECOS``, lido na hora da chamada.
    """
    tabela = pd.read_parquet(arquivo or ARQUIVO_PRECOS)
    tabela = tabela.set_index(pd.to_datetime(tabela.pop("date")))
    tabela = tabela.loc[pd.Timestamp(inicio):pd.Timestamp(fim)]
    return tabela.rename(columns=lambda papel: papel + SUFIXO_B3).reindex(columns=tickers).astype("float64")


def versao_dos_dados() -> int:
    """Carimbo do arquivo (mtime em ns) para a chave do cache: muda a cada publicação do lago; 0 se ausente."""
    try:
        return ARQUIVO_PRECOS.stat().st_mtime_ns
    except FileNotFoundError:
        return 0
