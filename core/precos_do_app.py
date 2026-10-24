"""
precos_do_app.py - The screen's daily prices: the lake's ``lago/precos_diarios.parquet`` as committed, plus
``lago/precos_recentes.parquet``, the sessions after it that the daily upkeep (``scripts/manutencao.py``) commits. The
big file is committed only at the paper routines' rebalances (~3 MB each time), so without the small one the deployed
app would be up to a rebalance old.

The lake's file is a total-return index anchored on the last close: each ex-date rescales the earlier sessions. The
small file starts a few sessions before the base's last one (the overlap), and the base is rescaled to it on the last
overlap session where a stock traded in both.

Kept out of ``core/lago.py`` and ``core/market_data.py``, which are in the paper guards' paths: the paper routines
read the big file directly, as before.
"""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import pandas as pd

from core import market_data
from core.lago import ARQUIVO_PRECOS, SUFIXO_B3

ARQUIVO_RECENTES = ARQUIVO_PRECOS.with_name("precos_recentes.parquet")


def emendar(base: pd.DataFrame, recentes: pd.DataFrame) -> pd.DataFrame:
    """``base`` followed by the sessions of ``recentes`` after its last one (both: dates as index, one column per
    security). Each column of ``base`` is rescaled by ``recentes`` / ``base`` on the last session of the overlap where
    both have a price (an ex-date between the files); a column with no common price is kept as it is."""
    ultima = base.index.max()
    comuns = base.loc[base.index.isin(recentes.index)]
    if not comuns.empty:
        razoes = recentes.reindex(index=comuns.index, columns=base.columns) / comuns
        base = base * razoes.ffill().iloc[-1].fillna(1.0)
    return pd.concat([base, recentes.loc[recentes.index > ultima]])


def _ler(arquivo: Path) -> pd.DataFrame:
    tabela = pd.read_parquet(arquivo)
    return tabela.set_index(pd.to_datetime(tabela.pop("date"))).astype("float64")


def ler_fechamentos(tickers: list[str], inicio: date, fim: date, base: Path = ARQUIVO_PRECOS,
                    recentes: Path = ARQUIVO_RECENTES) -> pd.DataFrame:
    """``lago.ler_fechamentos`` on the two files spliced (``emendar``): ``tickers`` in the app's format (with
    ".SA"), between ``inicio`` and ``fim``. Without the small file, the base alone."""
    tabela = _ler(base)
    if recentes.exists():
        tabela = emendar(tabela, _ler(recentes))
    tabela = tabela.loc[pd.Timestamp(inicio):pd.Timestamp(fim)]
    return tabela.rename(columns=lambda papel: papel + SUFIXO_B3).reindex(columns=tickers).astype("float64")


def versao_dos_dados(base: Path = ARQUIVO_PRECOS, recentes: Path = ARQUIVO_RECENTES) -> int:
    """The cache key's stamp (``lago.versao_dos_dados`` of both files): it changes when either file is rewritten."""
    return sum(arquivo.stat().st_mtime_ns for arquivo in (base, recentes) if arquivo.exists())


def carregar_precos(tickers: list[str], anos: int, hoje: date | None = None,
                    cobertura_minima: float = 0.95) -> market_data.DadosPrecos:
    """``market_data.carregar_precos`` on the spliced files."""
    fim = hoje or date.today()
    inicio = fim - timedelta(days=int(anos * 365.25) + 10)
    try:
        fechamentos = ler_fechamentos(list(dict.fromkeys(tickers)), inicio, fim)
    except FileNotFoundError as erro:
        raise market_data.ErroDadosMercado(f"Arquivo de preços do lago ausente: {erro.filename}") from erro
    return market_data.limpar_precos(fechamentos, cobertura_minima=cobertura_minima)
