"""
valor.py - Sinais de valor do S2 (PLAN §5, estratégias 2 e 3): Fórmula Mágica de Greenblatt e Trending Value de
O'Shaughnessy, com os fundamentos da CVM do lago.

Dados do lago (S1-F, comando ``fundamentals``): ``fundamentos`` = ``fundamentals.parquet`` (``issuer``, ``account``,
``dt_refer``, ``value``, ``available_from``: a data de recebimento do documento na CVM; saldos no fim do período, fluxos
em 12 meses); ``valor_mercado`` = ``market_caps.parquet`` (``issuer``, ``month``, ``market_cap`` na decisão do mês com o
fechamento do mês anterior, ``shares_today_units``). O sinal do mês ``mes`` usa só documentos recebidos antes do 1º dia
dele, como o IC (retorno do próprio mês) e o walk-forward (decisão dentro do mês) pedem.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from core.backtest import CarteiraRecusada, Estrategia
from core.estimation import estimar_momentos
from core.risco import LIMITES, Limites, checar_carteira, emissor
from core.sinais import NOMES, POR_SETOR, _meses_antes, carteira_da_nota, dividend_yield_12m

MESES_DE_IDADE_MAXIMA = 18      # documento com fim de período mais velho que isso: a empresa parou de entregar
SETOR_FORA = "Financeiro"
FRACAO_BARATA = 0.30            # os 30% de maior composto de valor (PLAN §5, S2.3: fixo no pré-registro)
MESES_DO_MOMENTUM_CURTO = 6     # momentum de 6 meses de O'Shaughnessy
POSTO_NEUTRO = 0.5              # múltiplo que falta fica no meio da fila (O'Shaughnessy dá 50 de 100)


def fundamentos_no_mes(fundamentos: pd.DataFrame, mes: pd.Timestamp) -> pd.DataFrame:
    """Emissor x conta: o valor do documento de fim de período mais recente entre os recebidos antes de ``mes``, desde
    que esse fim de período tenha no máximo ``MESES_DE_IDADE_MAXIMA`` meses."""
    conhecidos = fundamentos[(fundamentos["available_from"] < mes)
                             & (fundamentos["dt_refer"] >= mes - pd.DateOffset(months=MESES_DE_IDADE_MAXIMA))]
    ultimos = conhecidos.sort_values("dt_refer").drop_duplicates(["issuer", "account"], keep="last")
    return ultimos.pivot(index="issuer", columns="account", values="value")


def _conta(fundamentos_mes: pd.DataFrame, conta: str, zero_se_falta: bool = False) -> pd.Series:
    valores = (fundamentos_mes[conta] if conta in fundamentos_mes.columns
               else pd.Series(np.nan, index=fundamentos_mes.index, dtype=float))
    return valores.fillna(0.0) if zero_se_falta else valores


def _fora_do_financeiro(fundamentos_mes: pd.DataFrame, setores: Mapping[str, str]) -> pd.DataFrame:
    """Emissores com setor e fora de ``SETOR_FORA`` (bancos e seguradoras: outro plano de contas, sem EBIT)."""
    return fundamentos_mes[[setores.get(codigo) not in (None, SETOR_FORA) for codigo in fundamentos_mes.index]]


def valor_da_firma(fundamentos_mes: pd.DataFrame, valor_mercado: pd.Series) -> pd.Series:
    """Valor de mercado + empréstimos (curto e longo prazo) + participação dos minoritários (valor contábil: o EBIT
    consolidado inclui a parte deles) - caixa e aplicações financeiras; conta que falta vale 0."""
    divida = _conta(fundamentos_mes, "short_term_debt", True) + _conta(fundamentos_mes, "long_term_debt", True)
    caixa = _conta(fundamentos_mes, "cash", True) + _conta(fundamentos_mes, "short_term_investments", True)
    minoritarios = _conta(fundamentos_mes, "minority_interest", True)
    return valor_mercado.reindex(fundamentos_mes.index) + divida + minoritarios - caixa


def formula_magica(fundamentos_mes: pd.DataFrame, valor_mercado: pd.Series, setores: Mapping[str, str]) -> pd.Series:
    """
    Nota da Fórmula Mágica de Greenblatt: a média dos postos percentuais de EBIT/EV (``valor_da_firma``) e do retorno
    sobre o capital, EBIT / (capital de giro sem caixa nem empréstimo de curto prazo + imobilizado); as maiores notas
    são a menor soma de postos do livro. Fora: Financeiro e sem setor, sem EBIT, sem ativo ou passivo circulante,
    EV <= 0 ou capital <= 0 (retorno sobre o capital sem sentido).
    """
    dentro = _fora_do_financeiro(fundamentos_mes, setores)
    ebit = _conta(dentro, "ebit")
    caixa = _conta(dentro, "cash", True) + _conta(dentro, "short_term_investments", True)
    giro = ((_conta(dentro, "current_assets") - caixa)
            - (_conta(dentro, "current_liabilities") - _conta(dentro, "short_term_debt", True)))
    capital = giro + _conta(dentro, "fixed_assets", True)
    ev = valor_da_firma(dentro, valor_mercado)
    medidas = pd.DataFrame({"rendimento": ebit / ev, "retorno": ebit / capital})[(ev > 0) & (capital > 0)].dropna()
    return (medidas["rendimento"].rank(pct=True) + medidas["retorno"].rank(pct=True)) / 2.0


def rendimentos_de_valor(fundamentos_mes: pd.DataFrame, valor_mercado: pd.Series,
                         setores: Mapping[str, str]) -> pd.DataFrame:
    """
    Os 5 múltiplos de O'Shaughnessy que saem do balanço, como rendimento (maior = mais barato; lucro negativo fica caro
    em vez de virar um P/L sem sentido): lucro/valor de mercado (o dos controladores, ou o lucro inteiro quando a conta
    não vem), patrimônio sem os minoritários/valor de mercado, receita/valor de mercado, EBITDA/EV (EBIT + |depreciação
    e amortização| da DVA; vazio com EV <= 0) e caixa operacional/valor de mercado. Fora: Financeiro e sem setor,
    emissor sem valor de mercado e emissor sem lucro (só balanço patrimonial: com o ``POSTO_NEUTRO`` nos que faltam,
    ficaria no meio da fila sem número nenhum de resultado).
    """
    dentro = _fora_do_financeiro(fundamentos_mes, setores)
    mercado = valor_mercado.reindex(dentro.index)
    lucro = _conta(dentro, "net_income_parent").fillna(_conta(dentro, "net_income"))
    patrimonio = _conta(dentro, "equity") - _conta(dentro, "minority_interest", True)
    ebitda = _conta(dentro, "ebit") + _conta(dentro, "depreciation", True).abs()
    ev = valor_da_firma(dentro, valor_mercado)
    rendimentos = pd.DataFrame({"lucro": lucro / mercado, "patrimonio": patrimonio / mercado,
                                "receita": _conta(dentro, "revenue") / mercado, "ebitda": (ebitda / ev).where(ev > 0),
                                "caixa_operacional": _conta(dentro, "operating_cash_flow") / mercado})
    return rendimentos[(mercado > 0) & lucro.notna()]


@dataclass(frozen=True)
class Balancos:
    fundamentos: pd.DataFrame       # fundamentals.parquet do lago
    valor_mercado: pd.DataFrame     # market_caps.parquet do lago: issuer, month, market_cap
    setores: Mapping[str, str]      # emissor -> setor do lago

    def no_mes(self, mes: pd.Timestamp) -> tuple[pd.DataFrame, pd.Series]:
        """Os fundamentos conhecidos antes de ``mes`` e o valor de mercado da decisão de ``mes`` (fechamento de m-1)."""
        do_mes = self.valor_mercado[pd.to_datetime(self.valor_mercado["month"]) == mes]
        return fundamentos_no_mes(self.fundamentos, mes), do_mes.set_index("issuer")["market_cap"]


def momentum_6m(retornos_mensais: pd.DataFrame, mes: pd.Timestamp) -> pd.Series:
    """Retorno total acumulado de m-6 a m-1 para o sinal de ``mes`` (O'Shaughnessy não pula o último mês); papel com
    algum desses meses vazio fica sem momentum."""
    janela = _meses_antes(retornos_mensais, mes, MESES_DO_MOMENTUM_CURTO)
    return (1.0 + janela).prod(min_count=MESES_DO_MOMENTUM_CURTO).sub(1.0).where(janela.notna().all())


def composto_de_valor(balancos: Balancos, proventos: pd.DataFrame, mes: pd.Timestamp,
                      papeis: Iterable[str]) -> pd.Series:
    """Composto de valor de O'Shaughnessy em ``mes``, nos ``papeis`` com os rendimentos do emissor
    (``rendimentos_de_valor``): a média dos postos percentuais, entre esses papéis, dos 5 rendimentos do emissor e do
    dividend yield de 12 meses do papel; o que falta vale ``POSTO_NEUTRO``. Maior = mais barato."""
    fundamentos_mes, valor_mercado = balancos.no_mes(mes)
    rendimentos = rendimentos_de_valor(fundamentos_mes, valor_mercado, balancos.setores)
    com_balanco = [papel for papel in papeis if emissor(papel) in rendimentos.index]
    medidas = rendimentos.reindex([emissor(papel) for papel in com_balanco]).set_axis(com_balanco)
    medidas["dividend_yield"] = dividend_yield_12m(proventos, mes, com_balanco)
    return medidas.rank(pct=True).fillna(POSTO_NEUTRO).mean(axis=1)


def trending_value(balancos: Balancos, retornos_mensais: pd.DataFrame, proventos: pd.DataFrame, mes: pd.Timestamp,
                   candidatos: Iterable[str]) -> pd.Series:
    """
    Nota do Trending Value de O'Shaughnessy para ``mes``, nos ``candidatos`` com o momentum de 6 meses completo e o
    ``composto_de_valor``: 1 para a fração ``FRACAO_BARATA`` de maior composto (as baratas) e 0 para as outras, mais o
    posto percentual do momentum dentro do próprio grupo. As ``k`` maiores notas são as ``k`` baratas de maior
    momentum, e a nota inteira ordena o universo para o IC, como na Conservadora.
    """
    papeis = [papel for papel in candidatos if papel in retornos_mensais.columns]
    momento = momentum_6m(retornos_mensais[papeis], mes).dropna()
    composto = composto_de_valor(balancos, proventos, mes, momento.index)
    momento = momento[composto.index]
    barata = composto.rank(ascending=False, method="first") <= int(len(composto) * FRACAO_BARATA)
    postos = pd.concat([momento[grupo].rank(pct=True) for grupo in (barata, ~barata)])
    return postos.reindex(composto.index) + barata.astype(float)


def formula_magica_no_mes(balancos: Balancos, mes: pd.Timestamp, candidatos: Iterable[str]) -> pd.Series:
    """A nota da ``formula_magica`` do emissor em cada papel dos ``candidatos`` (as classes de um emissor têm a mesma
    nota, e a carteira fica com uma só); papel de emissor sem nota fica fora."""
    fundamentos_mes, valor_mercado = balancos.no_mes(mes)
    nota = formula_magica(fundamentos_mes, valor_mercado, balancos.setores)
    papeis = [papel for papel in candidatos if emissor(papel) in nota.index]
    return pd.Series([nota[emissor(papel)] for papel in papeis], index=papeis, dtype=float)


NotaDoMes = Callable[[pd.Timestamp, Sequence[str]], pd.Series]     # (mês, candidatos) -> nota por papel


def estrategia_da_nota(nota_do_mes: NotaDoMes, setores: Mapping[str, str], nomes: int = NOMES,
                       por_setor: int = POR_SETOR, limites: Limites = LIMITES) -> Estrategia:
    """
    Uma nota de valor no motor de walk-forward, como a ``estrategia_conservadora``: na decisão (último pregão do
    treino), a ``carteira_da_nota`` da nota do mês dela nos papéis que o motor deixa elegíveis (``setores`` por papel),
    conferida por ``checar_carteira`` com a cov de Ledoit-Wolf do treino; violação = ``CarteiraRecusada`` (o motor
    mantém a atual).
    """
    def _estrategia(treino: pd.DataFrame) -> pd.Series:
        mes = treino.index[-1].to_period("M").to_timestamp()
        pesos = carteira_da_nota(nota_do_mes(mes, list(treino.columns)), setores, nomes, por_setor)
        if pesos.empty:
            raise ValueError(f"{treino.index[-1]:%Y-%m-%d}: nenhum papel elegível com nota")
        violacoes = checar_carteira(pesos, setores, estimar_momentos(treino[pesos.index], shrink_mu=False).cov,
                                    limites)
        if violacoes:
            raise CarteiraRecusada(violacoes)
        return pesos
    return _estrategia


def painel_da_nota(nota_do_mes: NotaDoMes, universo: pd.DataFrame, meses: Iterable[pd.Timestamp],
                   colunas: Sequence[str]) -> pd.DataFrame:
    """A nota de cada mês (linha) nos papéis do universo ponto no tempo daquele mês (``universo``: ``month``,
    ``security``, como o ``universe_monthly`` do lago); vazio fora do universo ou sem nota."""
    elegiveis = universo["security"].groupby(pd.to_datetime(universo["month"])).apply(list)
    notas = {mes: nota_do_mes(mes, elegiveis.get(mes, [])) for mes in meses}
    return pd.DataFrame(notas).T.reindex(columns=list(colunas)).astype(float)
