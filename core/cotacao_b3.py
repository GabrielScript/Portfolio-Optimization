"""
cotacao_b3.py - The plan's quotes from B3's own public quote service (``cotacao.b3.com.br``, about 15 minutes late),
replacing Yahoo's 1-minute bars (the user's UI list of 2026-11-03; Yahoo is no source, DECISOES 224). After the
session the service's last print is the closing auction's (stamped 18:05 since 03/11/2026: continuous trading until
17:55, closing call 17:55-18:00), equal to the COTAHIST close: PETR4 50.71 on 03/11/2026 in both.

``core/market_data.py`` is a paper guard and stays as it is; this module only takes over the screen's quotes.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import requests

from core.market_data import FONTE_FECHAMENTO

URL = "https://cotacao.b3.com.br/mds/api/v1/DailyFluctuationHistory/{papel}"
FONTE_B3 = "B3 (atraso ~15 min)"
PRAZO_S = 15
CONEXOES = 8


def _baixar(papel: str) -> str:
    resposta = requests.get(URL.format(papel=papel), timeout=PRAZO_S, headers={"User-Agent": "Mozilla/5.0"})
    resposta.raise_for_status()
    return resposta.text


def ler_historico_do_dia(texto: str) -> tuple[float, pd.Timestamp] | None:
    """(last price, session date + time) of a ``DailyFluctuationHistory`` answer; None without a trade, with an error
    status or with something that is not its JSON."""
    try:
        dados = json.loads(texto)
        if dados["BizSts"]["cd"] != "OK":
            return None
        pregao = dados["TradgFlr"]
        ultimo = pregao["scty"]["lstQtn"][-1]
        preco = float(ultimo["closPric"])
    except (ValueError, KeyError, IndexError, TypeError):
        return None
    return (preco, pd.Timestamp(f"{pregao['date']} {ultimo['dtTm']}")) if preco > 0 else None


def _ultimo_fechamento(diarios: pd.DataFrame, ticker: str) -> tuple[float, pd.Timestamp, str]:
    validos = diarios[ticker].dropna() if ticker in diarios.columns else pd.Series(dtype=float)
    if validos.empty:
        return float("nan"), pd.NaT, FONTE_FECHAMENTO
    return float(validos.iloc[-1]), validos.index[-1], FONTE_FECHAMENTO


def cotacoes_b3(tickers: Iterable[str], diarios: pd.DataFrame,
                baixar: Callable[[str], str] | None = None) -> pd.DataFrame:
    """``preco``, ``horario`` and ``fonte`` per ticker (app format, ".SA"): B3's last trade of the session, or the
    last close of ``diarios`` (dates x tickers) when B3 fails or has no trade for it."""
    tickers, baixar = list(tickers), baixar or _baixar

    def um(ticker: str) -> tuple[float, pd.Timestamp, str]:
        try:
            lido = ler_historico_do_dia(baixar(ticker.removesuffix(".SA")))
        except (OSError, requests.RequestException):
            lido = None
        return (*lido, FONTE_B3) if lido is not None else _ultimo_fechamento(diarios, ticker)

    with ThreadPoolExecutor(max_workers=CONEXOES) as conexoes:
        linhas = dict(zip(tickers, conexoes.map(um, tickers)))
    return pd.DataFrame.from_dict(linhas, orient="index", columns=["preco", "horario", "fonte"])
