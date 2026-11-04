"""
The fictitious account (spec: ``docs/superpowers/specs/2026-10-31-simulador-design.md``): follows the paper-v3 records
from the 03/11/2026 close on. The ledger holds only orders; positions, cash and the curve are derived from it.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from core.execucao import Custos
from core.rebalanceio import CustoDaOrdem, Posicao, RegrasRebalanceio, plano_de_rebalanceio
from core.risco import LIMITES

INICIO = pd.Timestamp("2026-11-03")
CAPITAL_ACOES = 10_000.0          # the whole account in stocks (user, 2026-11-03, DECISOES 243)
CAPITAL_OPCOES = 0.0               # no options sleeve: the user invests only in stocks (2026-11-03)
# DeepSeek: the names of both rule portfolios it judges favorable; Controle: the same list in equal weights, no model
# (DECISOES 240 and 251)
CARTEIRAS = ("Composto", "Conservadora", "DeepSeek", "Controle")
REFERENCIAS = ("BOVA11",)              # comparison only, never invested (IVVB11 dropped, user 2026-11-03)
CUSTOS = Custos(emolumentos=0.0003, impacto=1.0)    # the same as the real-money plan in the app
CUSTO_BPS = 10.0                   # the plan's reserve, and the cost of a name outside the liquidity snapshot
COLUNAS = ["data", "carteira", "ticker", "ordem", "preco", "custo"]
LINHA_CAIXA = "Caixa (CDI)"
LINHA_OPCOES = "Reserva de opções (CDI)"


def _fator_do_cdi(cdi: pd.Series, dias: pd.DatetimeIndex) -> pd.Series:
    """Accumulated CDI per session. The last session's CDI is not out yet: it repeats the last known rate."""
    return (1.0 + cdi.reindex(dias).ffill().fillna(0.0)).cumprod()


@dataclass(frozen=True)
class Trilha:
    """The ledger of one portfolio replayed on the price files."""

    valores: pd.Series          # value per session; on a rebalance day, after its costs
    quantidades: pd.Series      # shares after the last rebalance
    fechamentos: pd.Series      # the close recorded on the last rebalance, for those shares
    caixa: float                # cash right after the last rebalance
    data: pd.Timestamp          # the last rebalance


def _trilha(ordens: pd.DataFrame, precos: pd.DataFrame, fator: pd.Series, capital: float) -> Trilha:
    """Each stretch is valued as quantity x recorded close x index(t) / index(rebalance), plus cash at the CDI. At the
    next rebalance the gap between that value and quantity x close (the dividends held in the index) becomes cash."""
    datas = list(ordens["data"].drop_duplicates().sort_values())
    quantidades = pd.Series(dtype=float)
    patrimonio, trechos = capital, []
    for posicao, data in enumerate(datas):
        ultimo = posicao + 1 == len(datas)
        fim = precos.index[-1] if ultimo else datas[posicao + 1]
        dia = ordens[ordens["data"] == data].set_index("ticker")
        quantidades = quantidades.add(dia["ordem"], fill_value=0.0)
        quantidades = quantidades[quantidades > 0]
        fechamentos = dia["preco"].reindex(quantidades.index)
        caixa = patrimonio - float((quantidades * fechamentos).sum()) - float(dia["custo"].sum())
        janela = precos.loc[data:fim, quantidades.index]
        trecho = ((janela / janela.iloc[0]) * quantidades * fechamentos).sum(axis=1)
        trecho = trecho + caixa * fator.loc[data:fim] / fator.loc[data]
        patrimonio = float(trecho.iloc[-1])
        trechos.append(trecho if ultimo else trecho.iloc[:-1])
    return Trilha(pd.concat(trechos), quantidades, fechamentos, caixa, datas[-1])


def _pendentes(registro: pd.DataFrame, feitas: pd.DataFrame, pregoes: pd.DatetimeIndex,
               inicio: pd.Timestamp) -> list[tuple[pd.Timestamp, pd.Series]]:
    """(trade date, weights) of each record not traded yet whose close is in the files, oldest first. A record trades
    at the first session on or after its date (the DeepSeek one is dated the day after its opinion, DECISOES 240); a
    portfolio whose first record comes after ``inicio`` starts then."""
    def pesos(data: pd.Timestamp) -> pd.Series:
        return registro[registro["data"] == data].set_index("ticker")["peso"]

    devidos = []
    if feitas.empty:
        anteriores = registro.loc[registro["data"] <= inicio, "data"]
        if not anteriores.empty:
            if inicio not in pregoes:
                return []
            devidos.append((inicio, pesos(anteriores.max())))
    ultima = inicio if feitas.empty else feitas["data"].max()
    novas = registro.loc[registro["data"] > ultima, "data"].drop_duplicates().sort_values()
    for data in novas:
        sessoes = pregoes[pregoes >= data]
        if len(sessoes):
            devidos.append((sessoes[0], pesos(data)))
    return devidos


def _rebalancear(feitas: pd.DataFrame, carteira: str, data: pd.Timestamp, pesos: pd.Series, precos: pd.DataFrame,
                 cdi: pd.Series, custo: CustoDaOrdem, capital: float) -> pd.DataFrame:
    ate = precos.loc[:data].ffill()
    cotacoes = ate.iloc[-1]
    posicoes: dict[str, Posicao] = {}
    patrimonio = capital
    if not feitas.empty:
        trilha = _trilha(feitas, ate, _fator_do_cdi(cdi, ate.index), capital)
        patrimonio = float(trilha.valores.iloc[-1])
        posicoes = {ticker: Posicao(int(quantidade), float(cotacoes[ticker]))
                    for ticker, quantidade in trilha.quantidades.items()}
    caixa = patrimonio - sum(posicao.quantidade * posicao.preco_medio for posicao in posicoes.values())
    regras = RegrasRebalanceio(LIMITES.peso_maximo_acao, CUSTO_BPS, quantidade_minima=1)
    plano = plano_de_rebalanceio(pesos, posicoes, cotacoes, caixa, regras, custo).ordens
    return pd.DataFrame({"data": data, "carteira": carteira, "ticker": plano.index, "ordem": plano["delta"].to_numpy(),
                         "preco": plano["cotacao"].to_numpy(), "custo": plano["custo"].to_numpy()})


def ordens_devidas(ordens: pd.DataFrame, registros: Mapping[str, pd.DataFrame], precos: pd.DataFrame, cdi: pd.Series,
                   custo: CustoDaOrdem, inicio: pd.Timestamp = INICIO, capital: float = CAPITAL_ACOES) -> pd.DataFrame:
    """The ledger rows to append: every rebalance due on ``precos`` (dates x tickers, total-return closes)."""
    novas = []
    for carteira, registro in registros.items():
        feitas = ordens[ordens["carteira"] == carteira]
        for data, pesos in _pendentes(registro, feitas, precos.index, inicio):
            linhas = _rebalancear(feitas, carteira, data, pesos, precos, cdi, custo, capital)
            feitas = linhas if feitas.empty else pd.concat([feitas, linhas], ignore_index=True)
            novas.append(linhas)
    return pd.concat(novas, ignore_index=True) if novas else pd.DataFrame(columns=COLUNAS)


def curvas(ordens: pd.DataFrame, precos: pd.DataFrame, cdi: pd.Series, capital: float = CAPITAL_ACOES,
           opcoes: float = CAPITAL_OPCOES) -> pd.DataFrame:
    """R$ per session from the first buy: each portfolio with its options sleeve at the CDI, then ``REFERENCIAS`` and
    the CDI as the whole capital on that day, with no costs."""
    janela = precos.loc[ordens["data"].min():].ffill()
    fator = _fator_do_cdi(cdi, janela.index)
    no_cdi = fator / fator.iloc[0]
    colunas = {carteira: _trilha(ordens[ordens["carteira"] == carteira], janela, fator, capital).valores + opcoes * no_cdi
               for carteira in CARTEIRAS if carteira in set(ordens["carteira"])}
    total = capital + opcoes
    colunas |= {nome: total * janela[nome] / janela[nome].iloc[0] for nome in REFERENCIAS}
    return pd.DataFrame(colunas | {"CDI": total * no_cdi})


def quedas(curvas_rs: pd.DataFrame) -> pd.DataFrame:
    """Each curve against its own peak so far."""
    return curvas_rs / curvas_rs.cummax() - 1.0


def retornos_recentes(curvas_rs: pd.DataFrame) -> pd.DataFrame:
    """One row per curve: the last session, the calendar week and month so far (from the last close before each began,
    or the first value when none exists), the whole period and the maximum drawdown."""
    ultimo = curvas_rs.iloc[-1]
    fim = curvas_rs.index[-1]

    def desde(corte: pd.Timestamp) -> pd.Series:
        antes = curvas_rs.loc[curvas_rs.index < corte]
        return ultimo / (antes.iloc[-1] if len(antes) else curvas_rs.iloc[0]) - 1.0

    return pd.DataFrame({"Dia": desde(fim), "Semana": desde(fim - pd.Timedelta(days=fim.weekday())),
                         "Mês": desde(fim.replace(day=1)), "Desde o início": ultimo / curvas_rs.iloc[0] - 1.0,
                         "Queda máx.": quedas(curvas_rs).min()})


def posicoes(ordens: pd.DataFrame, carteira: str, precos: pd.DataFrame, cdi: pd.Series, capital: float = CAPITAL_ACOES,
             opcoes: float = CAPITAL_OPCOES) -> pd.DataFrame:
    """The stocks of the portfolio's last rebalance at the last close, its cash and its options sleeve."""
    feitas = ordens[ordens["carteira"] == carteira]
    janela = precos.loc[feitas["data"].min():].ffill()
    fator = _fator_do_cdi(cdi, janela.index)
    trilha = _trilha(feitas, janela, fator, capital)
    movimento = (janela.iloc[-1] / janela.loc[trilha.data]).reindex(trilha.quantidades.index)
    acoes = pd.DataFrame({"Qtd.": trilha.quantidades, "Fechamento na troca (R$)": trilha.fechamentos,
                          "Valor (R$)": trilha.quantidades * trilha.fechamentos * movimento,
                          "Desde a troca": movimento - 1.0})
    no_cdi = {LINHA_CAIXA: trilha.caixa * fator.iloc[-1] / fator.loc[trilha.data]}
    if opcoes:
        no_cdi[LINHA_OPCOES] = opcoes * fator.iloc[-1] / fator.iloc[0]
    no_cdi = pd.DataFrame({"Valor (R$)": no_cdi})
    tabela = pd.concat([acoes, no_cdi])
    return tabela.assign(Peso=tabela["Valor (R$)"] / tabela["Valor (R$)"].sum())


@dataclass(frozen=True)
class Painel:
    """What the screen shows."""

    curvas: pd.DataFrame                    # ``curvas``
    posicoes: dict[str, pd.DataFrame]       # ``posicoes`` of each portfolio
    com_cdi: bool                           # False: the CDI series failed and cash shows no interest


def painel(ordens: pd.DataFrame, precos: pd.DataFrame, cdi: pd.Series) -> Painel:
    carteiras = [carteira for carteira in CARTEIRAS if carteira in set(ordens["carteira"])]
    return Painel(curvas(ordens, precos, cdi), {carteira: posicoes(ordens, carteira, precos, cdi) for carteira in carteiras},
                  not cdi.empty)


def ler_ordens(arquivo: Path) -> pd.DataFrame:
    """The ledger; empty, with its columns, before the first buy."""
    if not arquivo.exists():
        return pd.DataFrame(columns=COLUNAS).astype({"data": "datetime64[ns]"})
    return pd.read_csv(arquivo, parse_dates=["data"])


def acrescentar_ordens(arquivo: Path, novas: pd.DataFrame) -> None:
    """Appends to the ledger (LF, whole shares): rows already written are never touched."""
    arquivo.parent.mkdir(parents=True, exist_ok=True)
    novas.astype({"ordem": "int64"}).to_csv(arquivo, mode="a", header=not arquivo.exists(), index=False,
                                            lineterminator="\n", date_format="%Y-%m-%d")
