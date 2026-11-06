"""What the Sentinela's screen shows, read from its committed files (``ui/sentinela.py`` draws it): both curves of
the fictitious account, the alerts, the agent's orders, the calls' cost, the share of answers discarded by the checks
(the reading's first item: target under 5%, above 10% the agent stops) and the proper-score reading so far."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from core.sentinela import conta, pontuacao, registro
from core.sentinela.rodada import Arquivos

VEREDITOS = {"grave": "grave", "atencao": "atenção"}
ACOES = {"vender": "vender tudo", "reduzir_metade": "reduzir à metade"}
ALERTAS = 30


@dataclass(frozen=True)
class PainelSentinela:
    curvas: pd.DataFrame | None              # R$ per session: plain Composite, Composite + Sentinela, Selic
    alertas: pd.DataFrame                    # newest first
    acoes: pd.DataFrame                      # each action's latest state
    chamadas: int                            # DeepSeek calls so far
    custo_usd: float
    descartadas: float | None                # share of scans discarded by the checks
    ultima_rodada: pd.Timestamp
    autonomia: pontuacao.Autonomia


def painel(arquivos: Arquivos, precos: pd.DataFrame, selic: pd.Series) -> PainelSentinela | None:
    """None before the agent's first run."""
    log = registro.ler_log(arquivos.log)
    if log.empty:
        return None
    varreduras = log[log["etapa"] == "varredura"]
    marcados = varreduras[varreduras["veredito"].isin(VEREDITOS)].sort_values("hora", ascending=False,
                                                                                kind="stable").head(ALERTAS)
    alertas = pd.DataFrame({"Hora": marcados["hora"].dt.strftime("%d/%m/%Y %H:%M"), "Ação": marcados["ticker"],
                            "Veredito": marcados["veredito"].map(VEREDITOS), "Resumo": marcados["texto"]})
    acoes = registro.ler_acoes(arquivos.acoes)
    ultimas = acoes.groupby("id", sort=False).tail(1) if not acoes.empty else acoes
    tabela_acoes = pd.DataFrame({
        "Decidida em": ultimas["decidida_em"].dt.strftime("%d/%m/%Y %H:%M"), "Ação": ultimas["ticker"],
        "Ordem": ultimas["acao"].map(ACOES), "Confiança": ultimas["confianca"].astype(float),
        "Fechamento de": ultimas["executa_em"].dt.strftime("%d/%m/%Y"), "Situação": ultimas["estado"],
        "Motivo": ultimas["motivo"]})
    ordens = conta.ler(arquivos.ordens)
    curvas = conta.curvas(ordens, precos, selic) if not ordens.empty and not precos.empty else None
    pagas = log[log["etapa"].isin(registro.ETAPAS_PAGAS)]
    descartadas = float(varreduras["descarte"].notna().mean()) if len(varreduras) else None
    notas = pd.read_csv(arquivos.pontuacao) if arquivos.pontuacao.exists() else pd.DataFrame()
    return PainelSentinela(curvas, alertas.reset_index(drop=True), tabela_acoes.reset_index(drop=True), len(pagas),
                           float(pd.to_numeric(log["custo_usd"], errors="coerce").sum()), descartadas,
                           log["hora"].max(), pontuacao.autonomia(notas))
