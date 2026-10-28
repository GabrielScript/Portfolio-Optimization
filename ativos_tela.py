"""
ativos_tela.py - The market panel's full list: every stock, unit, BDR and ETF that the lake exports with the app's daily
prices (traded in the last 15 days), from ``lago/ativos.parquet`` (``b3lake.export.app_names``: B3's short name, asset
type and the issuer's B3 sector).

Information only: the optimizer and the paper stay on ``assets.py`` (the 181) and the optional US leg
(``ativos_eua.py``). The app's own names and sectors win over the lake's; a BDR or ETF without a sector shows its type.
"""

from pathlib import Path

import pandas as pd

ARQUIVO_ATIVOS = Path(__file__).resolve().parent / "lago" / "ativos.parquet"
TIPOS = {"stock": "Ação", "unit": "Ação", "bdr": "BDR", "fund_etf": "ETF"}
SEM_SETOR = {"BDR": "BDR", "ETF": "ETF", "Ação": ""}


def info_curada(acoes: dict, eua: dict) -> dict:
    """The app's own names and sectors (``assets.get_ticker_info``, ``ativos_eua.get_ticker_info_eua``) with a type,
    so the panel's type filter keeps them even without the lake file: the 181 are stocks (units included), the US leg
    is BDRs plus its two index ETFs (the "11" tickers)."""
    tipos = {**{ticker: "Ação" for ticker in acoes},
             **{ticker: "ETF" if ticker.removesuffix(".SA").endswith("11") else "BDR" for ticker in eua}}
    return {ticker: {**dados, "tipo": tipos[ticker]} for ticker, dados in {**acoes, **eua}.items()}


def info_da_tela(nomes: pd.DataFrame, curados: dict) -> dict:
    """Ticker (app format, ".SA") -> {nome, setor, tipo} for every row of ``nomes`` (security, name, asset_type,
    sector): ``curados`` (``assets`` + ``ativos_eua``) names and sectors first, the lake's B3 name and sector else."""
    info = {}
    for linha in nomes.itertuples(index=False):
        ticker, tipo = f"{linha.security}.SA", TIPOS[linha.asset_type]
        proprio = curados.get(ticker, {})
        setor = linha.sector if isinstance(linha.sector, str) else SEM_SETOR[tipo]
        info[ticker] = {"nome": proprio.get("nome", linha.name), "setor": proprio.get("setor", setor), "tipo": tipo}
    return info


def acoes_da_tela(nomes: pd.DataFrame) -> list[str]:
    """The stocks and units of ``nomes`` (app format): the only ones that get CVM fundamentals and filings, matched by
    the ticker's first 4 letters (a BDR's prefix can be a B3 issuer's code)."""
    return [f"{papel}.SA" for papel in nomes.loc[nomes["asset_type"].isin(["stock", "unit"]), "security"]]



def ler_nomes(arquivo: Path = ARQUIVO_ATIVOS) -> pd.DataFrame:
    """``lago/ativos.parquet``; empty (same columns) without the file, so the panel falls back to its own list."""
    if not arquivo.exists():
        return pd.DataFrame(columns=["security", "name", "asset_type", "sector"])
    return pd.read_parquet(arquivo)
