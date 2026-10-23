"""
rebalanceio.py - Rebalance from the current positions (S4, 2026-10-23): the live orders of the real-money plan.

``plano_de_rebalanceio`` takes the held quantities to the purchase plan (``core.ordens.plano_de_compra``) of the whole
patrimony (positions at today's quotes + cash), with each order's cost, the month's IR and, as information, the
no-trade zone (``docs/zona-2026-10-23.txt``: pre-registered, it did not pass, so the plan sends every order).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass

import numpy as np
import pandas as pd

from core.config import LimiteGrupo
from core.execucao import Custos, custo_da_ordem
from core.impostos import apurar_mes
from core.ordens import LOTE_PADRAO, TOLERANCIA_REAIS, TOLERANCIA_SOMA_PESOS, RegrasCompra, plano_de_compra


@dataclass(frozen=True)
class Posicao:
    quantidade: int         # shares held
    preco_medio: float      # average cost per share, R$ (the IR base)


@dataclass(frozen=True)
class RegrasRebalanceio:
    """The purchase-plan rules applied to the whole patrimony (``RegrasCompra`` without the budget)."""

    peso_maximo: float
    custo_bps: float                            # reserved by the plan for the cost of the orders
    quantidade_minima: int = 2
    grupos: tuple[LimiteGrupo, ...] = ()


def ordens_do_rebalanceio(alvo: pd.Series, posicoes: Mapping[str, Posicao], cotacoes: pd.Series, caixa: float,
                          regras: RegrasRebalanceio) -> pd.DataFrame:
    """
    One row per ticker held or in ``alvo``: ``atual`` (held), ``alvo`` (the quantity of ``plano_de_compra`` with the
    patrimony = held x ``cotacoes`` + ``caixa`` as the budget), ``delta`` = alvo - atual, ``lado`` ("compra",
    "venda" or "" with no order), ``lote_padrao`` and ``fracionario`` of the order, ``cotacao`` and ``valor``. A held
    ticker outside ``alvo`` is sold in full.
    """
    atual = pd.Series({ticker: posicao.quantidade for ticker, posicao in posicoes.items()}, dtype="int64")
    patrimonio = float((atual * cotacoes.reindex(atual.index)).sum()) + caixa
    plano = plano_de_compra(alvo, cotacoes, RegrasCompra(patrimonio, regras.custo_bps, regras.peso_maximo,
                                                         regras.quantidade_minima, regras.grupos))
    tickers = list(dict.fromkeys([*plano.itens.index, *atual.index]))
    return _quadro(atual.reindex(tickers, fill_value=0),
                   plano.itens["quantidade"].reindex(tickers, fill_value=0).astype("int64"), cotacoes)


def _quadro(atual: pd.Series, alvo: pd.Series, cotacoes: pd.Series) -> pd.DataFrame:
    quadro = pd.DataFrame({"atual": atual, "alvo": alvo})
    quadro["delta"] = quadro["alvo"] - quadro["atual"]
    quadro["lado"] = np.select([quadro["delta"] > 0, quadro["delta"] < 0], ["compra", "venda"], "")
    tamanho = quadro["delta"].abs()
    quadro["lote_padrao"] = tamanho // LOTE_PADRAO * LOTE_PADRAO
    quadro["fracionario"] = tamanho % LOTE_PADRAO
    quadro["cotacao"] = cotacoes.reindex(quadro.index).astype(float)
    quadro["valor"] = tamanho * quadro["cotacao"]
    return quadro


CustoDaOrdem = Callable[[str, float], float]     # (ticker, order value in R$) -> cost in R$


@dataclass(frozen=True)
class MesFiscal:
    """The investor's month so far, outside this plan: the exemption counts all of the person's stock sales."""

    vendas_de_acoes: float = 0.0
    resultado_de_acoes: float = 0.0     # gains minus losses of those sales
    prejuizo_acumulado: float = 0.0     # stock-market loss carried from earlier months


@dataclass(frozen=True)
class ZonaSemNegociacao:
    """
    No-trade zone (noguer ch07, linear costs): a resize of a ticker held and kept in the target is skipped when its
    cost plus its marginal IR is above its expected gain, the certainty-equivalent risk the deviation would keep for
    the ``horizonte``: (``aversao`` / 2) x vol_diaria^2 x horizonte x (valor / patrimony)^2 x patrimony. Under equal
    weights the signal has no view among the held names, so a resize has no return edge; entries and exits always
    trade. A ticker without ``vol_diaria`` always trades.
    """

    vol_diaria: pd.Series       # daily standard deviation of returns per ticker
    aversao: float = 2.0        # the study's lambda (half-Kelly)
    horizonte: int = 63         # sessions to the next rebalance


@dataclass(frozen=True)
class Rebalanceio:
    ordens: pd.DataFrame            # ``ordens_do_rebalanceio`` + ``custo``, ``ganho_realizado``, ``ir_marginal``,
                                    # ``ganho_esperado`` and ``situacao`` (with a zone)
    caixa_final: float              # cash + sales - buys - costs
    ir_do_mes: float                # the month's DARF with these orders (stocks and ETFs)
    vendas_de_acoes_no_mes: float   # ``mes.vendas_de_acoes`` + this plan's stock sales


SITUACAO_EXECUTAR, SITUACAO_PULAR, SITUACAO_FORCADA = "executar", "pular: custo acima do ganho", "forçada: regra de risco"


def plano_de_rebalanceio(alvo: pd.Series, posicoes: Mapping[str, Posicao], cotacoes: pd.Series, caixa: float,
                         regras: RegrasRebalanceio, custo: CustoDaOrdem, *, mes: MesFiscal = MesFiscal(),
                         etfs: frozenset[str] = frozenset(), zona: ZonaSemNegociacao | None = None) -> Rebalanceio:
    """
    The orders of ``ordens_do_rebalanceio``, each with ``custo(ticker, valor)``, and the cash left after them. When the
    costs pass the plan's reserve (``regras.custo_bps``) and the cash would go negative, the largest buy loses one
    share at a time (a new position below ``quantidade_minima`` is not bought) until it does not.

    IR (``apurar_mes``): a sale realizes value - cost - quantity x average price. Stocks: the month's sales and result
    add ``mes``, with the R$ 20 thousand exemption and the carried loss. ``etfs``: 15% of their gain, no exemption,
    apart from the stocks (simplification: an ETF loss does not offset a stock gain here). ``ir_marginal`` of a sale =
    the month's IR with it minus without it: crossing R$ 20 thousand taxes the whole month's gain.

    ``zona``: the skipped orders keep the held quantity and the plan is redone without them (one pass).
    """
    medio = pd.Series({ticker: posicao.preco_medio for ticker, posicao in posicoes.items()}, dtype=float)
    ordens = ordens_do_rebalanceio(alvo, posicoes, cotacoes, caixa, regras)
    ordens, caixa_final = _no_caixa(ordens, caixa, cotacoes, regras, custo)
    ordens, ir_do_mes = _com_ir(ordens, medio, mes, etfs)
    if zona is not None:
        patrimonio = float((ordens["atual"] * ordens["cotacao"]).sum()) + caixa
        ganho = _ganho_esperado(ordens, zona, patrimonio)
        pular = ganho.notna() & (ordens["atual"] > 0) & (ordens["alvo"] > 0) & (
            ordens["custo"] + ordens["ir_marginal"] > ganho)
        forcadas = pular & _fura_um_teto(ordens, pular, patrimonio, regras)
        pular &= ~forcadas
        if pular.any():
            ordens, caixa_final = _no_caixa(_quadro(ordens["atual"], ordens["alvo"].where(~pular, ordens["atual"]),
                                                    cotacoes), caixa, cotacoes, regras, custo)
            ordens, ir_do_mes = _com_ir(ordens, medio, mes, etfs)
        ordens["ganho_esperado"] = ganho
        ordens["situacao"] = np.select([pular, forcadas, ordens["delta"] != 0],
                                       [SITUACAO_PULAR, SITUACAO_FORCADA, SITUACAO_EXECUTAR], "")
    etf = ordens.index.isin(list(etfs))
    return Rebalanceio(ordens, caixa_final, ir_do_mes,
                       mes.vendas_de_acoes + float(ordens.loc[(ordens["delta"] < 0) & ~etf, "valor"].sum()))


def _no_caixa(ordens: pd.DataFrame, caixa: float, cotacoes: pd.Series, regras: RegrasRebalanceio,
              custo: CustoDaOrdem) -> tuple[pd.DataFrame, float]:
    """``ordens`` with their costs, the largest buy shrunk one share at a time while the cash left is negative."""
    ordens, caixa_final = _custear(ordens, caixa, custo)
    while caixa_final < -TOLERANCIA_REAIS:
        compras = ordens.loc[ordens["delta"] > 0, "valor"]
        if compras.empty:
            raise ValueError(f"o caixa e as vendas não pagam os custos (faltam R$ {-caixa_final:,.2f})")
        maior = compras.idxmax()
        quantidade = int(ordens.loc[maior, "alvo"]) - 1
        alvos = ordens["alvo"].copy()
        alvos[maior] = quantidade if quantidade >= regras.quantidade_minima else int(ordens.loc[maior, "atual"])
        ordens, caixa_final = _custear(_quadro(ordens["atual"], alvos, cotacoes), caixa, custo)
    return ordens, caixa_final


def _com_ir(ordens: pd.DataFrame, medio: pd.Series, mes: MesFiscal,
            etfs: frozenset[str]) -> tuple[pd.DataFrame, float]:
    vendas = ordens["delta"] < 0
    ordens = ordens.assign(ganho_realizado=np.where(
        vendas, ordens["valor"] - ordens["custo"] + ordens["delta"] * medio.reindex(ordens.index).fillna(0.0), 0.0))
    etf = ordens.index.isin(list(etfs))
    ir_do_mes = _ir_do_mes(ordens, vendas, etf, mes)
    marginal = [ir_do_mes - _ir_do_mes(ordens, vendas & (ordens.index != ticker), etf, mes) if vendas[ticker] else 0.0
                for ticker in ordens.index]
    return ordens.assign(ir_marginal=marginal), ir_do_mes


def _fura_um_teto(ordens: pd.DataFrame, pular: pd.Series, patrimonio: float, regras: RegrasRebalanceio) -> pd.Series:
    """Per ticker: skipping ``pular`` would leave it above ``peso_maximo`` or in a group above its ceiling."""
    pesos = ordens["alvo"].where(~pular, ordens["atual"]) * ordens["cotacao"] / patrimonio
    fura = pesos > regras.peso_maximo + TOLERANCIA_SOMA_PESOS
    for grupo in regras.grupos:
        membros = ordens.index.isin(list(grupo.tickers))
        if pesos[membros].sum() > grupo.teto + TOLERANCIA_SOMA_PESOS:
            fura |= membros
    return fura


def _ganho_esperado(ordens: pd.DataFrame, zona: ZonaSemNegociacao, patrimonio: float) -> pd.Series:
    variancia = zona.vol_diaria.reindex(ordens.index) ** 2 * zona.horizonte
    return zona.aversao / 2.0 * variancia * (ordens["valor"] / patrimonio) ** 2 * patrimonio


def _ir_do_mes(ordens: pd.DataFrame, vendas: pd.Series, etf: np.ndarray, mes: MesFiscal) -> float:
    acoes, fundos = vendas & ~etf, vendas & etf
    imposto_acoes = apurar_mes(mes.vendas_de_acoes + float(ordens.loc[acoes, "valor"].sum()),
                               mes.resultado_de_acoes + float(ordens.loc[acoes, "ganho_realizado"].sum()),
                               mes.prejuizo_acumulado).imposto
    imposto_etfs = apurar_mes(float(ordens.loc[fundos, "valor"].sum()),
                              float(ordens.loc[fundos, "ganho_realizado"].sum()), 0.0, isencao=0.0).imposto
    return imposto_acoes + imposto_etfs


def _custear(quadro: pd.DataFrame, caixa: float, custo: CustoDaOrdem) -> tuple[pd.DataFrame, float]:
    custos = [custo(ticker, valor) if valor > 0 else 0.0 for ticker, valor in quadro["valor"].items()]
    quadro = quadro.assign(custo=custos)
    return quadro, caixa - float((np.sign(quadro["delta"]) * quadro["valor"]).sum()) - float(quadro["custo"].sum())


def posicoes_da_tabela(tabela: pd.DataFrame) -> dict[str, Posicao]:
    """The screen's table (``ticker``, ``quantidade``, ``preco_medio``) as positions: tickers trimmed, upper case and
    without ".SA"; rows without a ticker or with no shares are dropped; a ticker on two rows adds its quantities at
    their quantity-weighted average price."""
    posicoes: dict[str, Posicao] = {}
    for linha in tabela.itertuples(index=False):
        ticker = "" if pd.isna(linha.ticker) else str(linha.ticker).strip().upper().removesuffix(".SA")
        if ticker and pd.notna(linha.quantidade) and int(linha.quantidade) > 0:
            quantidade, medio = int(linha.quantidade), 0.0 if pd.isna(linha.preco_medio) else float(linha.preco_medio)
            antes = posicoes.get(ticker, Posicao(0, 0.0))
            total = antes.quantidade + quantidade
            posicoes[ticker] = Posicao(total, (antes.quantidade * antes.preco_medio + quantidade * medio) / total)
    return posicoes


def capital_minimo(pesos: pd.Series, cotacoes: pd.Series, quantidade_minima: int, custo_bps: float) -> float:
    """The smallest patrimony whose target value of every name of ``pesos`` buys ``quantidade_minima`` shares, with
    ``custo_bps`` reserved for the cost (``plano_de_compra``'s reserve): below it, names drop out of the plan."""
    pesos = pesos[pesos > 0]
    return float((quantidade_minima * cotacoes.reindex(pesos.index) / pesos).max() * (1.0 + custo_bps / 10_000))


def custo_com_liquidez(foto: pd.DataFrame | None, custos: Custos, custo_bps: float) -> CustoDaOrdem:
    """The screen's order cost: ``custo_da_ordem`` with the liquidity snapshot (``security``, ``meio_spread``,
    ``vol_diaria``, ``volume_mediano``; ticker with or without ".SA"); ``custo_bps`` for a name not in it or whose row
    is incomplete (a missing measure or no volume)."""
    colunas = ["meio_spread", "vol_diaria", "volume_mediano"]
    completas = None if foto is None else foto[foto[colunas].notna().all(axis=1) & (foto["volume_mediano"] > 0)]
    medidas = {} if completas is None else {str(papel): tuple(linha)
                                            for papel, linha in completas.set_index("security")[colunas].iterrows()}

    def custo(ticker: str, valor: float) -> float:
        medida = medidas.get(ticker.removesuffix(".SA"))
        return valor * custo_bps / 10_000 if medida is None else custo_da_ordem(valor, *medida, custos)
    return custo
