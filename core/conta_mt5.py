"""
conta_mt5.py - The MT5 demo book on the Aleph Quant screen (spec ``2026-11-17-conta-mt5-no-app-design.md``): the pure
part, read from the two published files of ``paper-mt5/`` (``patrimonio.csv``: the 21:00 value of each day;
``fills.csv``: the book's deals). No ``MetaTrader5`` here and no import of ``scripts.mt5_executor`` (the order path):
it runs on Streamlit Cloud.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from core.mt5_book import cash_after
from core.resultado_diario import Linha, linha, serie_do_historico

INICIAL = 100_000.0              # the book's money (the user, 2026-11-12); a test holds it equal to the executor's
COLUNAS_POSICOES = ["Papel", "Qtd.", "Preço médio (R$)", "Custo (R$)"]


def posicoes(fills: pd.DataFrame) -> pd.DataFrame:
    """Net shares by name (the lot and fractional symbols merged), with the volume-weighted average buy price and the
    cost of the shares held at it; names sold to zero are left out."""
    if fills.empty:
        return pd.DataFrame(columns=COLUNAS_POSICOES)
    sinal = fills["side"].map({"buy": 1, "sell": -1})
    liquidas = (fills["volume"] * sinal).groupby(fills["name"]).sum()
    compras = fills[fills["side"] == "buy"]
    medio = (compras["volume"] * compras["price"]).groupby(compras["name"]).sum() / compras.groupby("name")["volume"].sum()
    detidas = liquidas[liquidas > 0].sort_index()
    preco = medio.reindex(detidas.index)
    return pd.DataFrame({"Papel": detidas.index, "Qtd.": detidas.astype(int).to_numpy(),
                         "Preço médio (R$)": preco.to_numpy(), "Custo (R$)": (detidas * preco).to_numpy()})


@dataclass(frozen=True)
class PainelMT5:
    curva: pd.Series             # the 21:00 value of each day
    linha: Linha                 # the last day: value, day and since-start changes
    caixa: float                 # the start minus buys, plus sells, minus commissions
    posicoes: pd.DataFrame       # ``COLUNAS_POSICOES``


def ler(pasta: Path) -> PainelMT5 | None:
    """The book from ``pasta``'s ``patrimonio.csv`` and ``fills.csv``; None without a valued day."""
    arquivo = pasta / "patrimonio.csv"
    if not arquivo.exists():
        return None
    historico = pd.read_csv(arquivo)
    if historico.empty:
        return None
    curva = serie_do_historico(historico)
    deals = pasta / "fills.csv"
    fills = pd.read_csv(deals) if deals.exists() else pd.DataFrame(columns=["name", "side", "volume", "price",
                                                                             "commission"])
    caixa = float(cash_after(INICIAL, fills[["side", "volume", "price", "commission"]].itertuples(index=False)))
    return PainelMT5(curva, linha(curva, INICIAL), caixa, posicoes(fills))


def lado_a_lado(mt5: pd.Series, composto: pd.Series | None) -> pd.DataFrame:
    """The book next to the Composite's simulator account, both at ``INICIAL`` on the simulator's last close before
    the book's first day (the book's money before its first buy); the book alone without such a close."""
    anteriores = composto.loc[composto.index < mt5.index[0]] if composto is not None else pd.Series(dtype=float)
    if anteriores.empty:
        return mt5.to_frame("MetaTrader 5")
    base = anteriores.index[-1]
    livro = pd.concat([pd.Series([INICIAL], index=[base]), mt5])
    simulado = composto.loc[base:] / float(anteriores.iloc[-1]) * INICIAL
    return pd.DataFrame({"MetaTrader 5": livro, "Composite (simulador)": simulado.reindex(livro.index)})
