"""The seen store (``simulador/sentinela_vistos.csv``): (ticker, item id) pairs already sent to the model for that
name. A macro item is new for every name until it has been in a call for that name; a company item only for its own."""

from __future__ import annotations

import csv
from collections.abc import Iterable
from datetime import datetime
from pathlib import Path

from core.sentinela.itens import MACRO, Item

COLUNAS = ["ticker", "id", "visto_em"]


def ler_vistos(arquivo: Path) -> set[tuple[str, str]]:
    if not arquivo.exists():
        return set()
    with arquivo.open(encoding="utf-8", newline="") as entrada:
        return {(linha["ticker"], linha["id"]) for linha in csv.DictReader(entrada)}


def novos_por_acao(itens: Iterable[Item], tickers: Iterable[str],
                   vistos: set[tuple[str, str]]) -> dict[str, list[Item]]:
    """Each name with something new: its own items, then the macro ones, in the order given; names with nothing new
    are left out (no new item, no model call)."""
    itens = list(itens)
    novos = {}
    for ticker in tickers:
        deste = [item for item in itens if item.ticker in (ticker, MACRO) and (ticker, item.id) not in vistos]
        deste.sort(key=lambda item: item.ticker == MACRO)
        if deste:
            novos[ticker] = deste
    return novos


def marcar(arquivo: Path, pares: Iterable[tuple[str, str]], agora: datetime) -> None:
    """Appends (LF): rows already written are never touched."""
    arquivo.parent.mkdir(parents=True, exist_ok=True)
    novo = not arquivo.exists()
    with arquivo.open("a", encoding="utf-8", newline="") as saida:
        escritor = csv.writer(saida, lineterminator="\n")
        if novo:
            escritor.writerow(COLUNAS)
        escritor.writerows((ticker, item, f"{agora:%Y-%m-%dT%H:%M:%S}") for ticker, item in pares)
