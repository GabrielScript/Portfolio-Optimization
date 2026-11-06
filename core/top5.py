"""
top5.py - The Composite top-5, a ghost strategy followed in the simulator (the user, 2026-11-06; DECISOES 290): at each
Composite record, the 5 names of highest Composite score (attempt 51's, ``scripts.paper_v3.nota_do_composto``), at
most 1 per sector, 20% each. One per sector is forced: 20% a name and the fixed 30% sector cap allow one.

Its record (``simulador/top5.csv``: ``data``, ``ticker``, ``peso``, ``base``) is what the simulator trades: ``base`` is
the Composite record it comes from and ``data`` the session it trades at (the launch record carries the 27/10 decision
and trades at the 06/11 close, the registration day). Nothing here touches the Composite, its record or paper-v3.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path

import pandas as pd

from core.backtest import Estrategia
from core.composto import estrategia_por_nota
from core.valor import NotaDoMes

CARTEIRA = "Top-5"                       # the simulator's name for the account
NOME = "Composite top-5"                 # the strategies comparison's name for the curve
NOMES = 5
POR_SETOR = 1
INICIO = pd.Timestamp("2026-11-06")       # registration day (DECISOES 290): its close is the first trade
COLUNAS = ["data", "ticker", "peso", "base"]


def estrategia(nota: NotaDoMes, setores: Mapping[str, str]) -> Estrategia:
    """The Composite's rule (``core.composto.estrategia_por_nota``) with 5 names and 1 per sector."""
    return estrategia_por_nota(nota, setores, nomes=NOMES, por_setor=POR_SETOR)


def vazio() -> pd.DataFrame:
    return pd.DataFrame({"data": pd.Series(dtype="datetime64[ns]"), "ticker": pd.Series(dtype=object),
                         "peso": pd.Series(dtype=float), "base": pd.Series(dtype="datetime64[ns]")})


def registros_devidos(composto: pd.DataFrame, feito: pd.DataFrame, pesos_do_dia: Callable[[pd.Timestamp], pd.Series],
                      inicio: pd.Timestamp = INICIO) -> pd.DataFrame:
    """The record rows to append: one top-5 record per Composite record date (``composto["data"]``) not yet in
    ``feito["base"]``, from the one in force on ``inicio`` on, each traded on ``max(base, inicio)``. ``pesos_do_dia``
    gives the top-5 weights decided on a Composite record's close (tickers with or without ".SA")."""
    bases = pd.DatetimeIndex(sorted(pd.to_datetime(composto["data"]).unique()))
    em_vigor = bases[bases <= inicio]
    bases = bases[bases >= em_vigor.max()] if len(em_vigor) else bases
    bases = bases[~bases.isin(pd.to_datetime(feito["base"]))]
    linhas = []
    for base in bases:
        pesos = pesos_do_dia(base)
        pesos = pesos[pesos > 0]
        linhas.append(pd.DataFrame({"data": max(base, inicio), "ticker": pesos.index.str.removesuffix(".SA"),
                                    "peso": pesos.to_numpy(dtype=float), "base": base}))
    return pd.concat(linhas, ignore_index=True)[COLUNAS] if linhas else vazio()


def ler(arquivo: Path) -> pd.DataFrame:
    if not arquivo.exists():
        return vazio()
    return pd.read_csv(arquivo, parse_dates=["data", "base"], dtype={"ticker": str})[COLUNAS]


def acrescentar(arquivo: Path, novas: pd.DataFrame) -> None:
    """Appends ``novas`` to the record file (header on a new file), dates as YYYY-MM-DD, LF line ends."""
    arquivo.parent.mkdir(parents=True, exist_ok=True)
    saida = novas[COLUNAS].assign(data=novas["data"].dt.strftime("%Y-%m-%d"), base=novas["base"].dt.strftime("%Y-%m-%d"))
    saida.to_csv(arquivo, mode="a", header=not arquivo.exists(), index=False, lineterminator="\n")


def pesos_do_dia(data: pd.Timestamp) -> pd.Series:
    """The top-5 decided on the close of ``data`` from the lake's exports, with the Composite's own score and training
    window (``scripts.paper_v3``); ``CarteiraRecusada`` if the fixed rules refuse it."""
    from scripts.paper_v2 import treino_do_dia
    from scripts.paper_v3 import ler_dados_do_composto, nota_do_composto
    dados = ler_dados_do_composto()
    nota = nota_do_composto(dados.lago.retornos_mensais, dados.lago.proventos, dados.balancos)
    pesos = estrategia(nota, dados.lago.setores)(treino_do_dia(dados.lago, data))
    return pesos[pesos > 0]
