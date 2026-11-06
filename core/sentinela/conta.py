"""The fictitious account "Composite + Sentinela" (spec, "The fictitious account", amendments 5 and 6).

Two portfolios in one ledger (``simulador/sentinela_ordens.csv``, the simulator's columns), both starting with
R$ 10,000 on the same session from the Composite's record: ``BASE`` follows the Composite only; ``CONTA`` follows it
too and also takes the agent's sales. Rules:

- an action decided on session T before 17h executes at T's close, else at the next session's close;
- it is accepted only if ``acao`` is not ``manter`` and ``confianca >= 0.70``; one action per name a day; after
  ``reduzir_metade`` only ``vender``; after ``vender`` nothing, until the Composite's next record;
- a sale's cash earns the daily Selic (the simulator's replay); the agent never buys back: the next record returns
  the account to the Composite's weights (``core.simulador.ordens_devidas``);
- a sale on a record's trade day is merged into that day's rows (one row per name per day, as the replay needs).

Only fiction: nothing here reaches the real-money plan.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from datetime import datetime
from pathlib import Path

import pandas as pd

from core import simulador
from core.rebalanceio import CustoDaOrdem

CONTA = "Composto + Sentinela"
BASE = "Composto (base)"
CAPITAL = 10_000.0
LIMIAR = 0.70
HORA_DE_CORTE = 17
ACOES_QUE_VENDEM = {"reduzir_metade", "vender"}


def data_de_execucao(decidida_em: datetime, pregoes: pd.DatetimeIndex) -> pd.Timestamp | None:
    """T's close for a decision before 17h on session T, else the next session's; None past the calendar."""
    dia = pd.Timestamp(decidida_em.date())
    if dia in pregoes and decidida_em.hour < HORA_DE_CORTE:
        return dia
    seguintes = pregoes[pregoes > dia]
    return seguintes[0] if len(seguintes) else None


def recusa(acao: Mapping, aceitas: Iterable[Mapping], registro: pd.Timestamp) -> str | None:
    """Why ``acao`` is not executed (None: it is). ``aceitas``: the actions accepted before it; only those decided on
    or after ``registro`` (the Composite's last record) count."""
    if acao["acao"] not in ACOES_QUE_VENDEM:
        return "manter não move a conta"
    if float(acao["confianca"]) < LIMIAR:
        return f"confiança {float(acao['confianca']):.2f} abaixo de {LIMIAR:.2f}"
    dia = pd.Timestamp(acao["decidida_em"]).normalize()
    deste = [anterior for anterior in aceitas if anterior["ticker"] == acao["ticker"]
             and pd.Timestamp(anterior["decidida_em"]).normalize() >= registro.normalize()]
    if any(pd.Timestamp(anterior["decidida_em"]).normalize() == dia for anterior in deste):
        return "já houve uma ação neste nome hoje"
    feitas = {anterior["acao"] for anterior in deste}
    if "vender" in feitas:
        return "o nome já foi vendido até o próximo registro da Composite"
    if "reduzir_metade" in feitas and acao["acao"] != "vender":
        return "depois de reduzir à metade, só vender"
    return None


def vazio() -> pd.DataFrame:
    return pd.DataFrame(columns=simulador.COLUNAS).astype({"data": "datetime64[ns]"})


def _vender(conta: pd.DataFrame, acao: Mapping, dia: pd.Timestamp, precos: pd.DataFrame, cdi: pd.Series,
            custo: CustoDaOrdem, capital: float) -> pd.DataFrame:
    ate = precos.loc[:dia].ffill()
    trilha = simulador._trilha(conta, ate, simulador._fator_do_juro(cdi, ate.index), capital)
    quantidades = trilha.quantidades.astype("int64")
    ticker = acao["ticker"]
    tem = int(quantidades.get(ticker, 0))
    vende = tem if acao["acao"] == "vender" else tem // 2
    if vende <= 0:
        return conta
    cotacoes = ate.iloc[-1].reindex(quantidades.index)
    linhas = pd.DataFrame({"data": dia, "carteira": CONTA, "ticker": quantidades.index, "ordem": 0,
                           "preco": cotacoes.to_numpy(), "custo": 0.0}).set_index("ticker", drop=False)
    linhas.loc[ticker, "ordem"] = -vende
    linhas.loc[ticker, "custo"] = custo(ticker, vende * float(cotacoes[ticker]))
    do_dia = conta[conta["data"] == dia]
    if not do_dia.empty:
        juntas = pd.concat([do_dia, linhas.reset_index(drop=True)], ignore_index=True)
        linhas = juntas.groupby("ticker", sort=False).agg(
            data=("data", "first"), carteira=("carteira", "first"), ordem=("ordem", "sum"), preco=("preco", "first"),
            custo=("custo", "sum")).reset_index()
    linhas = linhas.reset_index(drop=True)[simulador.COLUNAS]
    return pd.concat([conta[conta["data"] != dia], linhas], ignore_index=True)


def ordens_devidas(ordens: pd.DataFrame, registro: pd.DataFrame, acoes: Iterable[Mapping], precos: pd.DataFrame,
                   cdi: pd.Series, custo: CustoDaOrdem, inicio: pd.Timestamp,
                   capital: float = CAPITAL) -> tuple[pd.DataFrame, list[str]]:
    """The ledger rows to append for both portfolios and the ids of the accepted actions executed now. ``acoes``:
    accepted, not yet executed (id, ticker, acao, decidida_em, executa_em); one whose close is not in ``precos`` yet
    waits. An action dated on or before the ledger's last written session executes at the next session in
    ``precos`` (rows already written are never changed)."""
    base = ordens[ordens["carteira"] == BASE]
    novas = [simulador.ordens_devidas(base, {BASE: registro}, precos, cdi, custo, inicio, capital)]
    conta = ordens[ordens["carteira"] == CONTA]
    escrito = conta["data"].max() if not conta.empty else pd.Timestamp.min
    feitas = []
    for acao in sorted(acoes, key=lambda acao: (pd.Timestamp(acao["executa_em"]), pd.Timestamp(acao["decidida_em"]))):
        piso = max(pd.Timestamp(acao["executa_em"]), escrito + pd.Timedelta(days=1))
        seguintes = precos.index[precos.index >= piso]           # a planned day with no session moves to the next one
        if not len(seguintes):
            continue
        dia = seguintes[0]
        conta = pd.concat([conta, simulador.ordens_devidas(conta, {CONTA: registro}, precos.loc[:dia], cdi, custo,
                                                           inicio, capital)], ignore_index=True)
        if conta.empty or conta["data"].min() > dia:
            continue
        conta = _vender(conta, acao, dia, precos, cdi, custo, capital)
        feitas.append(acao["id"])
    conta = pd.concat([conta, simulador.ordens_devidas(conta, {CONTA: registro}, precos, cdi, custo, inicio, capital)],
                      ignore_index=True)
    novas.append(conta[conta["data"] > escrito])
    saida = pd.concat([quadro for quadro in novas if not quadro.empty], ignore_index=True) if any(
        not quadro.empty for quadro in novas) else vazio()
    return saida.astype({"ordem": "int64"}), feitas


def curvas(ordens: pd.DataFrame, precos: pd.DataFrame, cdi: pd.Series, capital: float = CAPITAL) -> pd.DataFrame:
    """R$ per session of both portfolios and of the Selic on the whole capital, from the first buy."""
    janela = precos.loc[ordens["data"].min():].ffill()
    fator = simulador._fator_do_juro(cdi, janela.index)
    colunas = {carteira: simulador._trilha(ordens[ordens["carteira"] == carteira], janela, fator, capital).valores
               for carteira in (BASE, CONTA) if carteira in set(ordens["carteira"])}
    return pd.DataFrame(colunas | {simulador.JURO: capital * fator / fator.iloc[0]})


def ler(arquivo: Path) -> pd.DataFrame:
    return simulador.ler_ordens(arquivo)


def acrescentar(arquivo: Path, novas: pd.DataFrame) -> None:
    simulador.acrescentar_ordens(arquivo, novas)
