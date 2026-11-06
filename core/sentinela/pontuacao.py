"""Proper scoring rules (spec amendment 4, DECISOES 271): every stage-2 confidence and every judge probability is
scored with the Brier and log scores against what the stock did after 20 sessions.

- Decision (stage 2): the event is a **fall** beyond 1.5 sigma of the name's 20-session move (sigma = the daily
  standard deviation of the 63 sessions before the decision, times sqrt(20)); p = ``confianca`` for ``vender`` and
  ``reduzir_metade``, 1 - ``confianca`` for ``manter``.
- Judges: the event is a **move** beyond the same band, either way (materiality has no sign); p = ``p_material``.

The reference close is the last session on or before the decision's day. The score table is derived (rewritten
whole from the append-only log and the price files at every run).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

SESSOES = 20
SESSOES_CURTAS = 5
SESSOES_DO_DESVIO = 63
SIGMAS = 1.5
LIMIAR = 0.70
ACERTO_MINIMO = 0.55
EPSILON = 1e-6
VENDAS = ("vender", "reduzir_metade")


def brier(p: float, y: bool) -> float:
    return (float(p) - float(bool(y))) ** 2


def log_score(p: float, y: bool) -> float:
    """-ln of the probability given to what happened (lower is better), with p clipped away from 0 and 1."""
    p = min(max(float(p), EPSILON), 1.0 - EPSILON)
    return -math.log(p if y else 1.0 - p)


def p_de_queda(acao: str, confianca: float) -> float:
    return float(confianca) if acao in VENDAS else 1.0 - float(confianca)


@dataclass(frozen=True)
class Resultado:
    ret_5: float
    ret_20: float
    limiar: float          # 1.5 sigma of the 20-session move
    queda: bool
    movimento: bool


def resultado(serie: pd.Series, dia: pd.Timestamp) -> Resultado | None:
    """What the name did after ``dia``; None until 20 sessions have passed or without 63 sessions of history."""
    serie = serie.dropna()
    posicao = int(serie.index.searchsorted(pd.Timestamp(dia).normalize(), side="right")) - 1
    if posicao < SESSOES_DO_DESVIO or posicao + SESSOES >= len(serie):
        return None
    retornos = serie.pct_change().iloc[posicao - SESSOES_DO_DESVIO + 1:posicao + 1]
    limiar = SIGMAS * float(retornos.std()) * math.sqrt(SESSOES)
    base = float(serie.iloc[posicao])
    ret_5 = float(serie.iloc[posicao + SESSOES_CURTAS]) / base - 1.0
    ret_20 = float(serie.iloc[posicao + SESSOES]) / base - 1.0
    return Resultado(ret_5, ret_20, limiar, ret_20 < -limiar, abs(ret_20) > limiar)


def pontuar(registro: pd.DataFrame, precos: pd.DataFrame) -> pd.DataFrame:
    """The log's decision and judge rows (``hora``, ``ticker``, ``etapa``, ``acao``, ``confianca``, ``p``) with
    ret_5, ret_20, the event, its probability and both scores (NaN while the outcome is not known)."""
    linhas = []
    for _, linha in registro.iterrows():
        etapa = str(linha["etapa"])
        if etapa == "decisao" and pd.notna(linha.get("confianca")):
            p = p_de_queda(str(linha["acao"]), float(linha["confianca"]))
        elif etapa.startswith("juiz") and pd.notna(linha.get("p")):
            p = float(linha["p"])
        else:
            continue
        saida = linha.to_dict() | {"p_evento": p, "ret_5": np.nan, "ret_20": np.nan, "evento": None,
                                   "brier": np.nan, "log": np.nan}
        if linha["ticker"] in precos.columns:
            achado = resultado(precos[linha["ticker"]], pd.Timestamp(linha["hora"]))
            if achado is not None:
                evento = achado.queda if etapa == "decisao" else achado.movimento
                saida |= {"ret_5": achado.ret_5, "ret_20": achado.ret_20, "evento": evento,
                          "brier": brier(p, evento), "log": log_score(p, evento)}
        linhas.append(saida)
    return pd.DataFrame(linhas)


@dataclass(frozen=True)
class Autonomia:
    """The reading's rule (amendment 4): None while no decision has an outcome."""
    mantem: bool | None
    n: int
    acerto: float | None           # share of executed-grade calls (sale at >= 0.70) that saw the fall
    brier: float | None
    brier_base: float | None       # the constant base rate's Brier on the same decisions


def autonomia(tabela: pd.DataFrame) -> Autonomia:
    if tabela.empty:
        return Autonomia(None, 0, None, None, None)
    decisoes = tabela[(tabela["etapa"] == "decisao") & tabela["evento"].notna()]
    if decisoes.empty:
        return Autonomia(None, 0, None, None, None)
    eventos = decisoes["evento"].astype(bool)
    taxa = float(eventos.mean())
    nosso = float(decisoes["brier"].mean())
    base = float(((taxa - eventos.astype(float)) ** 2).mean())
    fortes = decisoes[decisoes["acao"].isin(VENDAS) & (decisoes["confianca"] >= LIMIAR)]
    acerto = float(fortes["evento"].astype(bool).mean()) if len(fortes) else None
    mantem = (acerto is None or acerto >= ACERTO_MINIMO) and nosso <= base
    return Autonomia(bool(mantem), len(decisoes), acerto, nosso, base)
