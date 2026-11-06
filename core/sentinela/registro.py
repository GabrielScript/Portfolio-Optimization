"""The Sentinela's append-only files (spec, "Email and log"): rows already written are never touched (LF endings).

- ``simulador/sentinela_log.csv``: one row per model call (``varredura``, ``decisao``), per judge answer
  (``juiz:<name>``), per name left for the next session (``adiado``) and per digest sent (``resumo``), with tokens and
  cost. The 5- and 20-session outcomes and the Brier and log scores are derived from it (``core.sentinela.pontuacao``).
- ``simulador/sentinela_acoes.csv``: each stage-2 action that is not ``manter``, ``aceita`` or ``recusada`` (with
  the reason), then an ``executada`` row with the same id once the account trades it.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from datetime import date
from pathlib import Path

import pandas as pd

COLUNAS_LOG = ["hora", "ticker", "etapa", "veredito", "acao", "confianca", "p", "itens", "descarte", "texto",
               "citacoes", "entrada", "cache", "saida", "custo_usd", "erro"]
COLUNAS_ACOES = ["id", "decidida_em", "ticker", "acao", "confianca", "executa_em", "estado", "motivo"]
ETAPAS_PAGAS = ("varredura", "decisao")


def _acrescentar(arquivo: Path, linhas: Iterable[Mapping], colunas: list[str], datas: list[str]) -> None:
    quadro = pd.DataFrame(list(linhas)).reindex(columns=colunas)
    if quadro.empty:
        return
    for coluna in datas:
        quadro[coluna] = pd.to_datetime(quadro[coluna]).dt.strftime("%Y-%m-%dT%H:%M:%S")
    arquivo.parent.mkdir(parents=True, exist_ok=True)
    quadro.to_csv(arquivo, mode="a", header=not arquivo.exists(), index=False, lineterminator="\n", encoding="utf-8")


def _ler(arquivo: Path, colunas: list[str], datas: list[str]) -> pd.DataFrame:
    if not arquivo.exists():
        return pd.DataFrame(columns=colunas).astype({coluna: "datetime64[ns]" for coluna in datas})
    quadro = pd.read_csv(arquivo, encoding="utf-8")
    for coluna in datas:
        quadro[coluna] = pd.to_datetime(quadro[coluna])
    return quadro


def acrescentar_log(arquivo: Path, linhas: Iterable[Mapping]) -> None:
    _acrescentar(arquivo, linhas, COLUNAS_LOG, ["hora"])


def ler_log(arquivo: Path) -> pd.DataFrame:
    return _ler(arquivo, COLUNAS_LOG, ["hora"])


def usados_no_dia(log: pd.DataFrame, dia: date) -> dict[str, int]:
    """Model calls already made on ``dia``, per stage (the daily caps)."""
    do_dia = log[log["hora"].dt.date == dia] if not log.empty else log
    return {etapa: int((do_dia["etapa"] == etapa).sum()) for etapa in ETAPAS_PAGAS}


def acrescentar_acoes(arquivo: Path, linhas: Iterable[Mapping]) -> None:
    _acrescentar(arquivo, linhas, COLUNAS_ACOES, ["decidida_em", "executa_em"])


def ler_acoes(arquivo: Path) -> pd.DataFrame:
    return _ler(arquivo, COLUNAS_ACOES, ["decidida_em", "executa_em"])


def aceitas(acoes: pd.DataFrame) -> list[dict]:
    """Every accepted action, executed or not (they count for the one-action-a-day and cut-then-sell rules)."""
    return acoes[acoes["estado"] == "aceita"].to_dict("records")


def pendentes(acoes: pd.DataFrame) -> list[dict]:
    """Accepted actions not executed yet."""
    executadas = set(acoes.loc[acoes["estado"] == "executada", "id"])
    return [acao for acao in aceitas(acoes) if acao["id"] not in executadas]
