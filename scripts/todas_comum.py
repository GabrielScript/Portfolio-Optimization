"""
todas_comum.py - What the two readings of docs/superpowers/specs/2026-11-01-todas-as-estrategias-design.md share: an arm that finished
is never run again after a crash, and no arm starts inside the windows the lake's nightly job and the paper logs keep for themselves.
"""

from __future__ import annotations

import pickle
import time
from collections.abc import Callable
from datetime import datetime, time as hora
from pathlib import Path
from typing import TypeVar

Resultado = TypeVar("Resultado")

NOITE = (hora(21, 30), hora(0, 30))     # crosses midnight
PAPER = (hora(8, 25), hora(8, 50))
ESPERA_EM_SEGUNDOS = 60


def resultado_salvo(caminho: Path, rodar: Callable[[], Resultado]) -> Resultado:
    """The arm's result from the disk if a previous run finished it, or run now. The file only appears whole: these pickles are this
    script's own output, never anything downloaded."""
    if caminho.exists():
        with caminho.open("rb") as arquivo:
            return pickle.load(arquivo)
    resultado = rodar()
    parcial = caminho.with_suffix(".tmp")
    with parcial.open("wb") as arquivo:
        pickle.dump(resultado, arquivo)
    parcial.replace(caminho)
    return resultado


def na_janela_reservada(agora: datetime) -> bool:
    momento = agora.time()
    return momento >= NOITE[0] or momento < NOITE[1] or PAPER[0] <= momento < PAPER[1]


def esperar_a_janela_passar() -> None:
    while na_janela_reservada(datetime.now()):
        time.sleep(ESPERA_EM_SEGUNDOS)
