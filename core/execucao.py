"""
execucao.py - Execução realista de uma curva fora da amostra (S4): a mesma sequência de carteiras do motor, refeita em
reais, com custo por liquidez e o IR de pessoa física, e o valor líquido de saída (o que sobraria vendendo tudo).

Rebalance from the current positions (S4, 2026-10-23): ``ordens_do_rebalanceio`` takes the held quantities to the
purchase plan (``core.ordens.plano_de_compra``) of the whole patrimony (positions at today's quotes + cash).
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass

import numpy as np
import pandas as pd

from core.backtest import ResultadoBacktest
from core.config import LimiteGrupo
from core.impostos import ISENCAO_DE_VENDAS, apurar_mes
from core.ordens import LOTE_PADRAO, TOLERANCIA_REAIS, TOLERANCIA_SOMA_PESOS, RegrasCompra, plano_de_compra


@dataclass(frozen=True)
class Custos:
    emolumentos: float      # fração de cada ordem (B3: negociação + liquidação)
    impacto: float          # coeficiente da lei da raiz quadrada (x vol diária x raiz(ordem / volume))


SEM_CUSTOS = Custos(0.0, 0.0)


def custo_da_ordem(ordem: float, meio_spread: float, vol_diaria: float, volume_mediano: float, custos: Custos) -> float:
    """Custo em R$ de uma ordem de ``ordem`` reais (compra ou venda): emolumentos + meio spread + impacto x vol diária x
    raiz(ordem / volume mediano diário), a lei da raiz quadrada. Sem corretagem (a de hoje nas corretoras grandes)."""
    if ordem <= 0.0:
        return 0.0
    impacto = custos.impacto * vol_diaria * math.sqrt(ordem / volume_mediano)
    return ordem * (custos.emolumentos + meio_spread + impacto)


@dataclass(frozen=True)
class Liquidez:
    meio_spread: pd.DataFrame       # mês (1º dia) x papel, fração do preço
    vol_diaria: pd.DataFrame        # desvio-padrão dos retornos diários do mês
    volume_mediano: pd.DataFrame    # volume financeiro diário mediano do mês, R$


@dataclass(frozen=True)
class Refeita:
    valores: pd.Series      # patrimônio em R$ (antes do IR de saída)
    saida: pd.Series        # valor líquido de saída: patrimônio - IR se vendesse tudo naquele fechamento
    impostos: pd.Series     # IR pago no fim de cada mês
    custos: pd.Series       # custo de cada rebalanceio


def _custo_do_rebalanceio(ordens: pd.Series, mes: pd.Timestamp, liquidez: Liquidez | None, custos: Custos) -> float:
    """Soma do ``custo_da_ordem`` de cada papel; sem ``liquidez``, só os emolumentos. Papel sem medida no mês leva a
    mediana do mês entre os papéis; mês sem medida nenhuma é erro de dado."""
    if liquidez is None:
        return float(ordens.sum()) * custos.emolumentos
    return float(_custos_das_ordens(ordens, _medidas_do_mes(ordens.index, mes, liquidez), custos).sum())


def _medidas_do_mes(papeis: pd.Index, mes: pd.Timestamp, liquidez: Liquidez) -> dict[str, pd.Series]:
    medidas = {}
    for nome, quadro in (("spread", liquidez.meio_spread), ("vol", liquidez.vol_diaria),
                         ("volume", liquidez.volume_mediano)):
        if mes not in quadro.index:
            raise ValueError(f"Sem liquidez de {mes:%m/%Y} para o custo das ordens")
        linha = quadro.loc[mes]
        medidas[nome] = linha.reindex(papeis).fillna(linha.median())
    return medidas


def _custos_das_ordens(ordens: pd.Series, medidas: dict[str, pd.Series], custos: Custos) -> pd.Series:
    return pd.Series([custo_da_ordem(ordem, medidas["spread"][papel], medidas["vol"][papel], medidas["volume"][papel],
                                     custos) for papel, ordem in ordens.items()], index=ordens.index, dtype=float)


@dataclass(frozen=True)
class ZonaNoMotor:
    """
    The no-trade zone of ``ZonaSemNegociacao`` inside ``refazer``: at each rebalance, a ticker held and kept in the
    target skips its order when ``custo_da_ordem`` plus the sale's marginal IR in the month's apuração is above
    (``aversao`` / 2) x vol_diaria^2 x ``horizonte`` x (order / patrimony)^2 x patrimony, with the liquidity's
    ``vol_diaria`` of the month before the order; unless skipping leaves it above ``peso_maximo``. As in the live
    plan, the skipped positions keep their value and the others go to their own target value (weight x patrimony);
    what is left stays in cash at 0% until the next rebalance; when the skipped ones grew and the rest does not fit,
    the others shrink in proportion. The cost comes out of the cash first, then out of the others in proportion.
    """

    aversao: float = 2.0
    horizonte: int = 63
    peso_maximo: float = 0.20       # the fixed per-stock cap (core.risco.LIMITES.peso_maximo_acao)


def _trocar_pelo_preco_medio(posicoes: pd.Series, base: pd.Series, novas: pd.Series) -> tuple[pd.Series, float, float]:
    """De ``posicoes`` (custo ``base``) para ``novas``: a venda leva a base na proporção vendida e a compra soma a ela.
    Devolve a base nova, o total vendido e o resultado (ganho ou perda) das vendas."""
    vendido = (posicoes - novas).clip(lower=0.0)
    fracao = pd.Series(np.where(posicoes > 0.0, vendido / posicoes.where(posicoes > 0.0, 1.0), 0.0),
                       index=posicoes.index)
    nova_base = base * (1.0 - fracao) + (novas - posicoes).clip(lower=0.0)
    return nova_base, float(vendido.sum()), float((vendido - base * fracao).sum())


def refazer(resultado: ResultadoBacktest, retornos: pd.DataFrame, capital: float, custos: Custos = SEM_CUSTOS,
            liquidez: Liquidez | None = None, com_ir: bool = True, isencao: float = ISENCAO_DE_VENDAS,
            proventos: pd.DataFrame | None = None, zona: ZonaNoMotor | None = None) -> Refeita:
    """
    A sequência de carteiras do motor (``resultado.pesos_alvo``: cada linha vale do pregão do índice em diante) refeita
    com ``capital`` em reais: a compra é no fechamento anterior a esse pregão, com o patrimônio daquele fechamento menos
    o custo das ordens (``custo_da_ordem`` com a ``liquidez`` do mês anterior ao do fechamento, a já conhecida), e entre
    os rebalanceios cada posição anda com ``retornos`` (os do motor, ``retornos_da_carteira``).

    Com ``com_ir``: preço médio por papel (a venda leva a base na proporção vendida); no último pregão de cada mês (o
    último da amostra fica para a saída), ``apurar_mes`` sobre as vendas e o resultado do mês (``isencao``: R$ 20 mil
    para ações, 0 para ETF), e o IR sai do patrimônio na proporção de cada posição, sem contar como venda
    (simplificação). A saída de cada fechamento é o patrimônio menos o IR de vender tudo nele, somado às vendas e ao
    resultado do mês. ``proventos`` (pregão x papel, valor sobre o fechamento ex, já dentro de ``retornos``): o provento
    reinvestido entra na base, e não volta a pagar IR como ganho de capital.

    ``zona`` (needs ``liquidez``): the no-trade zone of ``ZonaNoMotor``; None = today's full rebalance, unchanged.
    """
    if zona is not None and liquidez is None:
        raise ValueError("a zona sem negociação precisa da liquidez (vol diária e custo de cada ordem)")
    datas = resultado.valores.index
    alvos = resultado.pesos_alvo.fillna(0.0)
    diarios = retornos.reindex(index=datas[1:], columns=alvos.columns).fillna(0.0)
    pagos = (pd.DataFrame(0.0, index=diarios.index, columns=diarios.columns) if proventos is None else
             proventos.reindex(index=diarios.index, columns=diarios.columns).fillna(0.0))
    fins_de_mes = {data for data, seguinte in zip(datas[:-1], datas[1:]) if data.month != seguinte.month}
    posicoes, base, caixa = pd.Series(0.0, index=alvos.columns), pd.Series(0.0, index=alvos.columns), float(capital)
    prejuizo, vendas_do_mes, resultado_do_mes = 0.0, 0.0, 0.0
    valores, saidas, gastos, impostos = [float(capital)], [float(capital)], {}, {}
    for anterior, (data, retorno) in zip(datas[:-1], diarios.iterrows()):
        if data in alvos.index:
            patrimonio = float(posicoes.sum()) + caixa
            alvo = alvos.loc[data]
            mes_da_liquidez = (anterior.to_period("M") - 1).to_timestamp()
            if zona is None:
                custo = _custo_do_rebalanceio((alvo * patrimonio - posicoes).abs(), mes_da_liquidez, liquidez, custos)
                novas = alvo * (patrimonio - custo)
            else:
                mes_fiscal = (vendas_do_mes, resultado_do_mes, prejuizo, isencao if com_ir else None)
                novas, custo, sobra = _rebalanceio_com_zona(alvo, posicoes, base, patrimonio,
                                                            _medidas_do_mes(alvo.index, mes_da_liquidez, liquidez),
                                                            custos, zona, mes_fiscal)
            base, vendas, ganho = _trocar_pelo_preco_medio(posicoes, base, novas)
            vendas_do_mes, resultado_do_mes = vendas_do_mes + vendas, resultado_do_mes + ganho
            posicoes, caixa = novas, (0.0 if zona is None else sobra)
            gastos[data] = custo
        posicoes = posicoes * (1.0 + retorno)
        base = base + posicoes * pagos.loc[data] / (1.0 + pagos.loc[data])
        if data in fins_de_mes:
            if com_ir:
                apuracao = apurar_mes(vendas_do_mes, resultado_do_mes, prejuizo, isencao)
                prejuizo = apuracao.prejuizo_acumulado
                if apuracao.imposto > 0.0:
                    fator = 1.0 - apuracao.imposto / (float(posicoes.sum()) + caixa)
                    posicoes, base, caixa = posicoes * fator, base * fator, caixa * fator
                    impostos[data] = apuracao.imposto
            vendas_do_mes, resultado_do_mes = 0.0, 0.0
        patrimonio = float(posicoes.sum()) + caixa
        valores.append(patrimonio)
        saida = apurar_mes(vendas_do_mes + float(posicoes.sum()), resultado_do_mes + float((posicoes - base).sum()),
                           prejuizo, isencao).imposto if com_ir else 0.0
        saidas.append(patrimonio - saida)
    return Refeita(pd.Series(valores, index=datas), pd.Series(saidas, index=datas), pd.Series(impostos, dtype=float),
                   pd.Series(gastos))


def _rebalanceio_com_zona(alvo: pd.Series, posicoes: pd.Series, base: pd.Series, patrimonio: float,
                          medidas: dict[str, pd.Series], custos: Custos, zona: ZonaNoMotor,
                          mes_fiscal: tuple[float, float, float, float | None]) -> tuple[pd.Series, float, float]:
    """The new positions, the cost and the cash left of one rebalance with ``zona`` (``mes_fiscal``: the month's
    sales, result, carried loss and exemption, or None for the exemption without IR)."""
    ordens = alvo * patrimonio - posicoes
    custo_de_cada = _custos_das_ordens(ordens.abs(), medidas, custos)
    ganho = zona.aversao / 2.0 * medidas["vol"] ** 2 * zona.horizonte * (ordens / patrimonio) ** 2 * patrimonio
    pular = ((posicoes > 0.0) & (alvo > 0.0) & (ordens != 0.0)
             & (custo_de_cada + _ir_marginal_no_motor(alvo * patrimonio, posicoes, base, mes_fiscal) > ganho)
             & ~(posicoes / patrimonio > zona.peso_maximo + 1e-12))
    if not pular.any():
        custo = float(custo_de_cada.sum())
        return alvo * (patrimonio - custo), custo, 0.0
    fixas = posicoes.where(pular, 0.0)
    alocavel = patrimonio - float(fixas.sum())
    livres = (alvo * patrimonio).where(~pular, 0.0)
    if livres.sum() > alocavel:
        livres = livres * (alocavel / livres.sum())
    custo = float(_custos_das_ordens((livres + fixas - posicoes).abs(), medidas, custos).sum())
    sobra = alocavel - float(livres.sum()) - custo
    if sobra < 0.0 and livres.sum() > 0.0:
        livres = livres * (1.0 + sobra / float(livres.sum()))
        sobra = 0.0
    return livres + fixas, custo, sobra


def _ir_marginal_no_motor(desejadas: pd.Series, posicoes: pd.Series, base: pd.Series,
                          mes_fiscal: tuple[float, float, float, float | None]) -> pd.Series:
    """Per sale: the month's IR with the full rebalance minus the IR without that one sale (0 without IR)."""
    vendas_do_mes, resultado_do_mes, prejuizo, isencao = mes_fiscal
    marginal = pd.Series(0.0, index=posicoes.index)
    if isencao is None:
        return marginal

    def imposto(novas: pd.Series) -> float:
        _, vendas, ganho = _trocar_pelo_preco_medio(posicoes, base, novas)
        return apurar_mes(vendas_do_mes + vendas, resultado_do_mes + ganho, prejuizo, isencao).imposto

    com_todas = imposto(desejadas)
    for papel in posicoes.index[desejadas < posicoes]:
        marginal[papel] = com_todas - imposto(desejadas.where(desejadas.index != papel, posicoes))
    return marginal


# ---------------------------------------------------------------- rebalance from the current positions

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
        pular = ganho.notna() & (ordens["atual"] > 0) & ordens.index.isin(alvo.index[alvo > 0]) & (
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
