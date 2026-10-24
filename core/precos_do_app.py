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
FUSO_DA_B3 = "America/Sao_Paulo"


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


def com_fechamento_oficial(cotacoes: pd.DataFrame, fechamentos: pd.DataFrame) -> pd.DataFrame:
    """``cotacoes`` (``market_data.cotacoes_atuais``) with each 1-minute bar replaced by the ticker's last close in
    ``fechamentos`` (dates x tickers) when that close is from the bar's session or later. Yahoo's 1-minute bars stop
    before B3's closing auction (16:55-17:00), so once the lake has a session's close, that is its official price; a
    bar of a session the lake does not have yet (trading now, or closed but not yet in the upkeep's file) is kept."""
    corrigidas = cotacoes.astype({"horario": "object"})     # Yahoo's times are tz-aware, the lake's dates naive
    for ticker in corrigidas.index[corrigidas["fonte"] == market_data.FONTE_INTRADIARIA]:
        validos = fechamentos[ticker].dropna() if ticker in fechamentos.columns else pd.Series(dtype=float)
        if not validos.empty and validos.index[-1] >= _pregao(corrigidas.at[ticker, "horario"]):
            corrigidas.loc[ticker] = [float(validos.iloc[-1]), validos.index[-1], market_data.FONTE_FECHAMENTO]
    return corrigidas


def _pregao(horario: pd.Timestamp) -> pd.Timestamp:
    """The B3 session (a naive date) of a bar's time; Yahoo's times are tz-aware."""
    if horario.tzinfo is not None:
        horario = horario.tz_convert(FUSO_DA_B3).tz_localize(None)
    return horario.normalize()


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
