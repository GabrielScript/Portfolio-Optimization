"""
fermionica_fantasma.py - The "Fermiônica" fictitious account (the user, 2026-11-11; DECISOES 344 and 348;
docs/superpowers/specs/2026-11-11-fermionic-ghost-design.md): at each Composite record, 15 names among every scored
candidate by F(S) = 2 theta sum z + log det C_S at theta = 0.5 (``core.fermionica``), 4 per sector, 1/15 each. The
best of the six determinantal arms of DECISOES 344, which failed its gate: followed forward, not a decision.

Its record (``simulador/fermionica.csv``) has the Top-5's columns and dating (``core.top5``): ``base`` is the
Composite record and ``data`` the session it trades at, the first session after the registration day on.

The simulator's screen imports this module for the two constants, so the rule is imported only when it is used.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from core.backtest import Estrategia
    from core.valor import NotaDoMes

CARTEIRA = "Fermiônica"
THETA = 0.5
INICIO = pd.Timestamp("2026-11-12")       # registered on Wednesday 11/11 after the close: the next close trades


def estrategia(nota: NotaDoMes, setores: Mapping[str, str]) -> Estrategia:
    """The second reading's "LogDet theta 0.5" arm: every default of ``estrategia_fermionica`` but theta."""
    from core.fermionica import estrategia_fermionica
    return estrategia_fermionica(nota, setores, theta=THETA)


def pesos_do_dia(data: pd.Timestamp) -> pd.Series:
    """The account's portfolio decided on the close of ``data`` from the lake's exports, with the Composite's own
    score and training window (``scripts.paper_v3``); ``CarteiraRecusada`` if the fixed rules refuse it."""
    from scripts.paper_v2 import treino_do_dia
    from scripts.paper_v3 import ler_dados_do_composto, nota_do_composto
    dados = ler_dados_do_composto()
    nota = nota_do_composto(dados.lago.retornos_mensais, dados.lago.proventos, dados.balancos)
    return estrategia(nota, dados.lago.setores)(treino_do_dia(dados.lago, data))
