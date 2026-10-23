"""
execucao.py - Execução realista de uma curva fora da amostra (S4): a mesma sequência de carteiras do motor, refeita em
reais, com custo por liquidez e o IR de pessoa física, e o valor líquido de saída (o que sobraria vendendo tudo).

The rebalance from the current positions (the live orders) is in ``core/rebalanceio.py``; ``ZonaNoMotor`` here is its
no-trade zone inside ``refazer``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

from core.backtest import ResultadoBacktest
from core.impostos import ISENCAO_DE_VENDAS, apurar_mes


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
