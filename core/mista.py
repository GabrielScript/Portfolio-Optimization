"""
mista.py - The 12/3 ghost (the user, 2026-11-07; DECISOES 298-299): at each Composite record, the Composite's rule with
12 names (4 per sector, its own score and training window) plus the 3 US BDRs of highest 12-1 momentum among the
eligible US set (``ativos_eua``), 1/15 each. The arm that led the 2021-2026 reading of DECISOES 298 without passing its
gate: followed forward, net of the BDR tax (the simulator's ``TRIBUTADOS_15``), not a decision.

Its record (``simulador/mista_12_3.csv``) has the Top-5's columns and dating (``core.top5``): ``base`` is the Composite
record and ``data`` the session it trades at, the first session after the registration day on.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import replace
from datetime import timedelta
from pathlib import Path

import pandas as pd

from core.backtest import Estrategia
from core.composto import estrategia_por_nota
from core.mix_bdr import escolher_eua
from core.risco import LIMITES, Limites
from core.valor import NotaDoMes

CARTEIRA = "12/3"
NOMES_BR = 12
NOMES_EUA = 3
NOMES = NOMES_BR + NOMES_EUA
INICIO = pd.Timestamp("2026-11-09")       # registered on Saturday 07/11: the next session's close is the first trade
GOLD = Path("D:/b3-datalake/gold/equity_adjusted.parquet")
DIAS_DE_HISTORIA = 800                    # calendar days read for the US names: 12 months of volume and 252 sessions


def limites_br() -> Limites:
    """The whole portfolio's caps on the BR sleeve's 12/15 share (DECISOES 298's addendum)."""
    escala = NOMES / NOMES_BR
    return replace(LIMITES, teto_setor=LIMITES.teto_setor * escala, peso_maximo_acao=LIMITES.peso_maximo_acao * escala,
                   peso_maximo_emissor=LIMITES.peso_maximo_emissor * escala)


def estrategia_br(nota: NotaDoMes, setores: Mapping[str, str]) -> Estrategia:
    return estrategia_por_nota(nota, setores, nomes=NOMES_BR, limites=limites_br())


def combinar(pesos_br: pd.Series, eua: Sequence[str]) -> pd.Series:
    """Each US pick 1/15; the BR sleeve the rest, in its own proportions (12/15 when all 3 US slots fill)."""
    br = pesos_br[pesos_br > 0]
    parte_eua = len(eua) / NOMES
    return pd.concat([br / br.sum() * (1.0 - parte_eua), pd.Series(1.0 / NOMES, index=list(eua))])


def _eua_no_lago(data: pd.Timestamp) -> tuple[pd.DataFrame, pd.DataFrame]:
    import pyarrow.parquet as pq

    from ativos_eua import ATIVOS_EUA
    tabela = pq.read_table(GOLD, columns=["date", "security", "tr_close", "volume_brl"],
                           filters=[("security", "in", [ativo["ticker"] for ativo in ATIVOS_EUA]),
                                    ("date", ">=", (data - timedelta(days=DIAS_DE_HISTORIA)).date()),
                                    ("date", "<=", data.date())]).to_pandas()
    tabela["date"] = pd.to_datetime(tabela["date"])
    precos = tabela.pivot(index="date", columns="security", values="tr_close").sort_index()
    return precos, tabela.pivot(index="date", columns="security", values="volume_brl").reindex(precos.index)


def pesos_do_dia(data: pd.Timestamp) -> pd.Series:
    """The 12/3 decided on the close of ``data``: the BR sleeve from the lake's exports with the Composite's own score
    and training window (``scripts.paper_v3``), the US picks from the lake's gold table up to ``data``."""
    from scripts.paper_v2 import treino_do_dia
    from scripts.paper_v3 import ler_dados_do_composto, nota_do_composto
    dados = ler_dados_do_composto()
    nota = nota_do_composto(dados.lago.retornos_mensais, dados.lago.proventos, dados.balancos)
    br = estrategia_br(nota, dados.lago.setores)(treino_do_dia(dados.lago, data))
    precos, volume = _eua_no_lago(data)
    return combinar(br, escolher_eua(precos, volume, data, NOMES_EUA))
