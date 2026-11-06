"""The Sentinela's emails (Portuguese, like the app): one at once for every ``grave`` and its decision, one digest at
the 19h run with the day's ``atencao`` and the names left for the next session. Sent by ``core.alerta_email``."""

from __future__ import annotations

from collections.abc import Mapping

import pandas as pd

from core.assessor import Evidencia
from core.sentinela.checagem import Decisao, Varredura

RODAPE = ("A Sentinela só age na conta fictícia \"Composite + Sentinela\" (R$ 10.000 fictícios). No dinheiro real a "
          "decisão é sua; isto não é recomendação de investimento.")
NOMES_DAS_ACOES = {"vender": "vender tudo", "reduzir_metade": "reduzir à metade", "manter": "manter"}


def _citacoes(citacoes: list[dict], evidencias: Mapping[str, Evidencia]) -> list[str]:
    linhas = []
    for citacao in citacoes:
        evidencia = evidencias.get(citacao["id"].strip())
        fonte = evidencia.classe if evidencia else "?"
        linhas.append(f"- [{fonte}] {citacao['id']}: «{citacao['trecho'].strip()}»")
    return linhas


def mensagem_de_alerta(ticker: str, empresa: str, varredura: Varredura, evidencias: Mapping[str, Evidencia],
                       decisao: Decisao | None, acao: Mapping | None) -> tuple[str, str]:
    assunto = f"Sentinela: {ticker} marcada como grave"
    linhas = [f"A Sentinela marcou {ticker} ({empresa or ticker}) como GRAVE.", "", f"Resumo: {varredura.resumo}", "",
              "Citações (texto literal das fontes oficiais):", *_citacoes(varredura.citacoes, evidencias), ""]
    if decisao is None:
        linhas.append("Decisão: não pedida hoje (limite diário de chamadas de esforço máximo); fica para o próximo pregão.")
    else:
        linhas.append(f"Decisão para a conta fictícia: {NOMES_DAS_ACOES[decisao.acao]}, confiança "
                      f"{100 * decisao.confianca:.0f}%.")
        if decisao.descarte:
            linhas.append(f"A resposta não passou na checagem ({decisao.descarte}) e virou manter.")
        if decisao.motivo:
            linhas.append(f"Motivo: {decisao.motivo}")
        linhas += _citacoes(decisao.citacoes, evidencias)
        if acao is not None:
            if acao["estado"] == "aceita":
                assunto += f"; venda fictícia: {NOMES_DAS_ACOES[acao['acao']]}"
                linhas.append(f"Ordem fictícia aceita: executa no fechamento de "
                              f"{pd.Timestamp(acao['executa_em']):%d/%m/%Y}.")
            else:
                linhas.append(f"Ordem fictícia recusada: {acao['motivo']}.")
    return assunto, "\n".join([*linhas, "", RODAPE])


def mensagem_do_resumo(dia: pd.Timestamp, atencoes: pd.DataFrame, adiados: list[str]) -> tuple[str, str]:
    assunto = f"Sentinela: resumo de {dia:%d/%m/%Y} (atenção)"
    linhas = [f"Itens marcados como atenção em {dia:%d/%m/%Y}:"] if not atencoes.empty else []
    for _, linha in atencoes.iterrows():
        linhas.append(f"- {linha['ticker']}: {linha['texto']}")
    if adiados:
        linhas += ["", "Ficaram para o próximo pregão (limite diário de chamadas): " + ", ".join(sorted(set(adiados)))]
    return assunto, "\n".join([*linhas, "", RODAPE])
