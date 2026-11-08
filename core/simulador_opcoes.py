"""
The fictitious options account (spec: ``docs/superpowers/specs/2026-11-05-simulador-opcoes-design.md``): buys a put on
a stretched stock near its high (the user's idea A) and a call on a volatile winner (idea B) on the first session of
each month, 10% of the account per cycle, held to expiry. Pure logic: the upkeep reads the lake's
``option_chain_recent.parquet`` and the price files and appends the ledger and the daily marks; the screen replays
them with ``curva``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

INICIO = pd.Timestamp("2026-11-06")          # the first session close after the code went live (2026-11-05)
CAPITAL = 10_000.0                           # a separate fictitious account, as each stocks account
ORCAMENTO = 0.10                             # of the account's value per cycle, half per idea
LOTE = 100
DU_ALVO, DU_MIN, DU_MAX = 21, 10, 45
VOLUME_MINIMO = 1e6                          # R$ a day of options traded over 21 sessions
TIPOS_DE_ACAO = ("stock", "unit")            # ETFs, BDRs and index options are out
ASK_MAXIMO = 2.0                             # x the bid: a lone ask with no real bid is not a price (dry run 05/11)
EMOLUMENTOS = 0.0003                        # fraction of the premium (the ask already pays the spread)
IR = 0.15                                    # swing-trade options, no monthly exemption
COLUNAS = ["data", "ideia", "ticker", "ativo", "tipo", "strike", "vencimento", "quantidade", "preco", "custo",
           "motivo"]
COLUNAS_MARCAS = ["data", "ticker", "valor", "vencida"]
IDEIAS = {"A": ("put", 0.95), "B": ("call", 1.05)}


def ciclos(pregoes: pd.DatetimeIndex, inicio: pd.Timestamp = INICIO) -> list[pd.Timestamp]:
    """The first session on or after ``inicio``, then the first session of each later month."""
    sessoes = pregoes[pregoes >= inicio]
    if sessoes.empty:
        return []
    primeiras = sessoes.to_series().groupby(sessoes.to_period("M")).min()
    return [sessoes[0], *[dia for dia in primeiras if dia.to_period("M") > sessoes[0].to_period("M")]]


def _historico(fechamentos: pd.DataFrame, candidatos: list[str], sessoes: int) -> pd.DataFrame:
    janela = fechamentos.reindex(columns=candidatos).iloc[-sessoes:]
    return janela.loc[:, janela.notna().all()] if len(janela) == sessoes else janela.iloc[:, :0]


def ideia_a(fechamentos: pd.DataFrame, candidatos: list[str]) -> str | None:
    """The highest 252-session return among the names whose last close is at least 95% of their 252-session high."""
    janela = _historico(fechamentos, candidatos, 253)
    if janela.empty:
        return None
    perto = janela.iloc[-1] >= 0.95 * janela.iloc[1:].max()
    retorno = (janela.iloc[-1] / janela.iloc[0] - 1.0)[perto]
    return None if retorno.empty else str(retorno.idxmax())


def ideia_b(fechamentos: pd.DataFrame, candidatos: list[str]) -> str | None:
    """The highest 12-1 momentum (t-252 to t-21) among the most volatile third by 63-session realized volatility."""
    janela = _historico(fechamentos, candidatos, 253)
    if janela.empty:
        return None
    vol = np.log(janela.iloc[-64:]).diff().iloc[1:].std()
    volateis = vol.sort_values(ascending=False).index[:math.ceil(len(vol) / 3)]
    momento = janela.iloc[-22][volateis] / janela.iloc[0][volateis] - 1.0
    return str(momento.idxmax())


def contrato(cadeia: pd.DataFrame, ativo: str, tipo: str, fator: float) -> pd.Series | None:
    """The option of ``ativo`` that traded with an ask, on the expiry nearest 21 business days within [10, 45] (ties:
    the earlier), at the strike nearest ``fator`` x the stock's close (ties: the lower strike)."""
    opcoes = cadeia[(cadeia["underlying"] == ativo) & (cadeia["option_type"] == tipo) & (cadeia["trades"] > 0)
                    & (cadeia["best_bid"] > 0) & (cadeia["best_ask"] >= cadeia["best_bid"])
                    & (cadeia["best_ask"] <= ASK_MAXIMO * cadeia["best_bid"])
                    & cadeia["du"].between(DU_MIN, DU_MAX)]
    if opcoes.empty:
        return None
    opcoes = opcoes.assign(dist_du=(opcoes["du"] - DU_ALVO).abs())
    vencimento = opcoes.sort_values(["dist_du", "expiry"])["expiry"].iloc[0]
    opcoes = opcoes[opcoes["expiry"] == vencimento]
    opcoes = opcoes.assign(dist_k=(opcoes["strike"] - fator * opcoes["spot_close"]).abs())
    return opcoes.sort_values(["dist_k", "strike"]).iloc[0]


def ordens_do_ciclo(data: pd.Timestamp, cadeia: pd.DataFrame, fechamentos: pd.DataFrame, acoes: list[str],
                    patrimonio: float) -> pd.DataFrame:
    """One ledger row per idea for the cycle on ``data`` (``cadeia`` = that session's option rows, ``fechamentos`` =
    total-return closes up to ``data``, ``acoes`` = the universe's stocks): the order, or quantity 0 and the reason."""
    dia = cadeia[cadeia["date"] == data]
    liquidos = sorted(set(dia.loc[dia["underlying_volume_21"] >= VOLUME_MINIMO, "underlying"]) & set(acoes))
    escolhas = {"A": ideia_a(fechamentos.loc[:data], liquidos), "B": ideia_b(fechamentos.loc[:data], liquidos)}
    verba = patrimonio * ORCAMENTO / len(IDEIAS)
    linhas = []
    for ideia, (tipo, fator) in IDEIAS.items():
        linha = dict.fromkeys(COLUNAS, None) | {"data": data, "ideia": ideia, "tipo": tipo, "quantidade": 0,
                                                "custo": 0.0, "motivo": ""}
        ativo = escolhas[ideia]
        escolhido = None if ativo is None else contrato(dia, ativo, tipo, fator)
        if ativo is None:
            linha["motivo"] = "nenhum ativo passou na regra"
        elif escolhido is None:
            linha |= {"ativo": ativo, "motivo": "sem opção negociada no prazo"}
        else:
            preco = float(escolhido["best_ask"])
            quantidade = int(verba // (preco * LOTE)) * LOTE
            linha |= {"ticker": escolhido["ticker"], "ativo": ativo, "strike": float(escolhido["strike"]),
                      "vencimento": escolhido["expiry"], "preco": preco, "quantidade": quantidade,
                      "custo": quantidade * preco * EMOLUMENTOS,
                      "motivo": "" if quantidade else f"um lote custa R$ {preco * LOTE:,.2f}, acima da verba"}
        linhas.append(linha)
    return pd.DataFrame(linhas, columns=COLUNAS)


def _valor_de_tela(linha: pd.Series) -> float | None:
    if 0 < linha["best_bid"] <= linha["best_ask"] <= ASK_MAXIMO * linha["best_bid"]:
        return (linha["best_bid"] + linha["best_ask"]) / 2
    return float(linha["close"]) if linha["close"] > 0 else None


def marcas_devidas(livro: pd.DataFrame, cadeia: pd.DataFrame, feitas: pd.DataFrame) -> pd.DataFrame:
    """The daily unit value of each bought option after its purchase, up to its expiry: the bid-ask mid, else its
    close; on the first session on or after expiry, the intrinsic value at the stock's close (``vencida``). Sessions
    already in ``feitas`` are skipped."""
    marcadas = set(zip(pd.to_datetime(feitas["data"]), feitas["ticker"]))
    linhas = []
    for compra in livro[livro["quantidade"] > 0].itertuples():
        sessoes = sorted(cadeia.loc[(cadeia["underlying"] == compra.ativo) & (cadeia["date"] > compra.data), "date"]
                         .unique())
        for dia in sessoes:
            dia = pd.Timestamp(dia)
            vencida = dia >= compra.vencimento
            if (dia, compra.ticker) not in marcadas:
                do_ativo = cadeia[(cadeia["underlying"] == compra.ativo) & (cadeia["date"] == dia)]
                if vencida:
                    spot = float(do_ativo["spot_close"].iloc[0])
                    valor = max(compra.strike - spot, 0.0) if compra.tipo == "put" else max(spot - compra.strike, 0.0)
                else:
                    propria = do_ativo[do_ativo["ticker"] == compra.ticker]
                    valor = None if propria.empty else _valor_de_tela(propria.iloc[0])
                if valor is not None:
                    linhas.append({"data": dia, "ticker": compra.ticker, "valor": valor, "vencida": vencida})
            if vencida:
                break
    return pd.DataFrame(linhas, columns=COLUNAS_MARCAS)


def atualizacao(livro: pd.DataFrame, marcas: pd.DataFrame, cadeia: pd.DataFrame, fechamentos: pd.DataFrame,
                acoes: list[str], selic: pd.Series, inicio: pd.Timestamp = INICIO,
                capital: float = CAPITAL) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The ledger rows and marks to append: every cycle in ``cadeia``'s sessions after the last one traded (the
    account's value that day sizes its budget), then the marks of everything bought."""
    pregoes = pd.DatetimeIndex(sorted(pd.to_datetime(cadeia["date"].unique())))
    ultimo = pd.to_datetime(livro["data"]).max() if len(livro) else None
    novas_ordens, novas_marcas = [], []

    def marcar(ate: pd.Timestamp | None) -> None:
        trecho = cadeia if ate is None else cadeia[cadeia["date"] <= ate]
        feitas = pd.concat([marcas, *novas_marcas]) if novas_marcas else marcas
        novas_marcas.append(marcas_devidas(livro, trecho, feitas))

    for dia in ciclos(pregoes, inicio):
        if ultimo is not None and dia <= ultimo:
            continue
        marcar(dia)
        patrimonio = capital
        if len(livro):
            todas = pd.concat([marcas, *novas_marcas])
            patrimonio = float(curva(livro, todas, selic, pregoes[pregoes <= dia], capital).iloc[-1])
        ordens = ordens_do_ciclo(dia, cadeia, fechamentos, acoes, patrimonio)
        livro = ordens if livro.empty else pd.concat([livro, ordens], ignore_index=True)
        novas_ordens.append(ordens)
    marcar(None)
    ordens = pd.concat(novas_ordens, ignore_index=True) if novas_ordens else pd.DataFrame(columns=COLUNAS)
    return ordens, pd.concat(novas_marcas, ignore_index=True)[COLUNAS_MARCAS]


@dataclass(frozen=True)
class PainelOpcoes:
    """What the screen shows."""

    curvas: pd.DataFrame        # "Opções" and "Selic" in R$ per session from the first cycle
    ordens: pd.DataFrame        # every cycle's row, with the reason when nothing was bought
    abertas: pd.DataFrame       # options not expired yet, at their last mark


def painel(livro: pd.DataFrame, marcas: pd.DataFrame, selic: pd.Series, capital: float = CAPITAL) -> PainelOpcoes:
    """The account on the Selic's sessions from its first cycle (the Selic is the account's cash and its reference)."""
    pregoes = selic.index[selic.index >= pd.to_datetime(livro["data"]).min()]
    contas = curva(livro, marcas, selic, pregoes, capital)
    fator = (1.0 + selic.reindex(pregoes).fillna(0.0)).cumprod()
    curvas = pd.DataFrame({"Opções": contas, "Selic": capital * fator / fator.iloc[0]})
    ordens = livro.rename(columns={"data": "Data", "ideia": "Ideia", "ticker": "Opção", "ativo": "Ação",
                                   "tipo": "Tipo", "strike": "Strike", "vencimento": "Vencimento",
                                   "quantidade": "Qtd.", "preco": "Prêmio (R$)", "custo": "Custo (R$)",
                                   "motivo": "Motivo"}).fillna({"Motivo": ""})
    marcas = marcas.assign(data=pd.to_datetime(marcas["data"]))
    vencidas = set(marcas.loc[marcas["vencida"].astype(bool), "ticker"])
    ultima = marcas.sort_values("data").groupby("ticker")["valor"].last()
    compradas = livro[(livro["quantidade"] > 0) & ~livro["ticker"].isin(vencidas)]
    unitario = compradas["ticker"].map(ultima).fillna(compradas["preco"])
    abertas = pd.DataFrame({"Opção": compradas["ticker"], "Ação": compradas["ativo"], "Tipo": compradas["tipo"],
                            "Strike": compradas["strike"], "Vencimento": compradas["vencimento"],
                            "Qtd.": compradas["quantidade"], "Prêmio (R$)": compradas["preco"],
                            "Valor (R$)": compradas["quantidade"] * unitario}).reset_index(drop=True)
    return PainelOpcoes(curvas, ordens, abertas)


def ler(arquivo, colunas: list[str], datas: list[str]) -> pd.DataFrame:
    """A ledger file; empty, with its columns, before it exists."""
    if not arquivo.exists():
        return pd.DataFrame(columns=colunas)
    return pd.read_csv(arquivo, parse_dates=datas)


def acrescentar(arquivo, novas: pd.DataFrame) -> None:
    """Appends to a ledger file (LF): rows already written are never touched."""
    arquivo.parent.mkdir(parents=True, exist_ok=True)
    novas.to_csv(arquivo, mode="a", header=not arquivo.exists(), index=False, lineterminator="\n",
                 date_format="%Y-%m-%d")


def curva(livro: pd.DataFrame, marcas: pd.DataFrame, selic: pd.Series, pregoes: pd.DatetimeIndex,
          capital: float = CAPITAL) -> pd.Series:
    """R$ per session: cash at the Selic (premiums and costs out on the purchase, the intrinsic value in on expiry, IR
    15% on each expiry session's net result above the losses carried) plus each open option at its last mark (the
    purchase price before the first mark)."""
    compras = livro[livro["quantidade"] > 0].set_index("ticker")
    pregoes = pregoes[pregoes >= pd.to_datetime(livro["data"]).min()] if len(livro) else pregoes
    marcas = marcas.assign(data=pd.to_datetime(marcas["data"]))
    tela = marcas.pivot_table(index="data", columns="ticker", values="valor").reindex(pregoes).ffill()
    vencidas = marcas[marcas["vencida"].astype(bool)].set_index("ticker")["data"]
    caixa, prejuizo, valores = capital, 0.0, []
    fator = 1.0 + selic.reindex(pregoes).ffill().fillna(0.0)
    for dia in pregoes:
        caixa *= fator.loc[dia] if valores else 1.0
        hoje = compras[pd.to_datetime(compras["data"]) == dia]
        caixa -= float((hoje["quantidade"] * hoje["preco"] + hoje["custo"]).sum())
        resultado, abertas = 0.0, 0.0
        for ticker, compra in compras.iterrows():
            if compra["data"] > dia:
                continue
            if vencidas.get(ticker) == dia:
                recebido = compra["quantidade"] * tela.loc[dia, ticker]
                caixa += recebido
                resultado += recebido - compra["quantidade"] * compra["preco"] - compra["custo"]
            elif ticker not in vencidas or vencidas[ticker] > dia:
                unitario = tela[ticker].get(dia) if ticker in tela else None
                abertas += compra["quantidade"] * (compra["preco"] if pd.isna(unitario) else unitario)
        if resultado:
            tributavel = resultado - prejuizo
            caixa -= IR * max(tributavel, 0.0)
            prejuizo = max(-tributavel, 0.0)
        valores.append(caixa + abertas)
    return pd.Series(valores, index=pregoes, name="Opções")
