"""
classicas.py - Classic value strategies, compared with the Conservadora on the same engine: Carlisle's Acquirer's
Multiple and Piotroski's F-score (Greenblatt's Magic Formula is ``core.valor.formula_magica``), on the lake's CVM
statements point-in-time (``core.valor``: documents received before the month, flows over 12 months).

Kept out of ``core/valor.py`` and ``core/sinais.py`` (the latter is in the paper-v2 guard's path): new code only.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping

import pandas as pd

from core.risco import emissor
from core.valor import MESES_DE_IDADE_MAXIMA, Balancos, _conta, _fora_do_financeiro, valor_da_firma

PESO_DO_DESEMPATE = 0.5         # book-to-market percentile x this: orders firms of equal F-score, never across them


def balancos_em_tres_anos(fundamentos: pd.DataFrame,
                          mes: pd.Timestamp) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Issuer x account at the issuer's latest period end among the documents received before ``mes`` (at most
    ``MESES_DE_IDADE_MAXIMA`` months old), and at the same period end one and two years earlier (the same quarter, so
    the balances compare without seasonality), each also received before ``mes``. An issuer without one of those
    period ends is missing from that table.
    """
    conhecidos = fundamentos[fundamentos["available_from"] < mes]
    ultimo = conhecidos.groupby("issuer")["dt_refer"].max()
    ultimo = ultimo[ultimo >= mes - pd.DateOffset(months=MESES_DE_IDADE_MAXIMA)]
    tabelas = []
    for anos in (0, 1, 2):
        alvo = (ultimo - pd.DateOffset(years=anos)).rename("alvo")
        linhas = conhecidos.join(alvo, on="issuer", how="inner")
        linhas = linhas[linhas["dt_refer"] == linhas["alvo"]]
        tabelas.append(linhas.pivot_table(index="issuer", columns="account", values="value", aggfunc="last"))
    return tabelas[0], tabelas[1], tabelas[2]


TOLERANCIA_DE_ACOES = 0.001     # split-adjusted counts move by rounding: a rise up to 0.1% is not an issue


def f_score(atual: pd.DataFrame, um_ano: pd.DataFrame, dois_anos: pd.DataFrame, acoes: pd.Series,
            acoes_um_ano: pd.Series) -> pd.Series:
    """
    Piotroski's F-score (0-9), as in Walkshäusl (2020): one point each for (1) net income > 0, (2) operating cash flow
    > 0, (3) ROA up (net income over the year-earlier total assets), (4) operating cash flow > net income, (5)
    long-term debt / total assets down, (6) current ratio up, (7) no new shares (``acoes`` vs ``acoes_um_ano``, split-
    adjusted), (8) gross margin up, (9) asset turnover up (revenue over the year-earlier total assets). The tables are
    ``balancos_em_tres_anos``; an issuer without any input is left out (missing long-term debt = no such debt).
    """
    def conta(tabela: pd.DataFrame, nome: str, zero_se_falta: bool = False) -> pd.Series:
        return _conta(tabela.reindex(atual.index), nome, zero_se_falta)

    lucro, caixa = conta(atual, "net_income"), conta(atual, "operating_cash_flow")
    ativo, ativo_1, ativo_2 = conta(atual, "total_assets"), conta(um_ano, "total_assets"), conta(dois_anos,
                                                                                              "total_assets")
    receita, receita_1 = conta(atual, "revenue"), conta(um_ano, "revenue")
    sinais = pd.DataFrame({
        "lucro": lucro > 0,
        "caixa": caixa > 0,
        "roa": lucro / ativo_1 > conta(um_ano, "net_income") / ativo_2,
        "accruals": caixa > lucro,
        "alavancagem": (conta(atual, "long_term_debt", True) / ativo
                        < conta(um_ano, "long_term_debt", True) / ativo_1),
        "liquidez": (conta(atual, "current_assets") / conta(atual, "current_liabilities")
                     > conta(um_ano, "current_assets") / conta(um_ano, "current_liabilities")),
        "emissao": acoes.reindex(atual.index) <= acoes_um_ano.reindex(atual.index) * (1 + TOLERANCIA_DE_ACOES),
        "margem": conta(atual, "gross_profit") / receita > conta(um_ano, "gross_profit") / receita_1,
        "giro": receita / ativo_1 > receita_1 / ativo_2,
    })
    entradas = pd.concat([lucro, caixa, ativo, ativo_1, ativo_2, receita, receita_1, conta(um_ano, "net_income"),
                          conta(atual, "current_assets"), conta(atual, "current_liabilities"),
                          conta(um_ano, "current_assets"), conta(um_ano, "current_liabilities"),
                          conta(atual, "gross_profit"), conta(um_ano, "gross_profit"),
                          acoes.reindex(atual.index), acoes_um_ano.reindex(atual.index)], axis=1)
    return sinais[entradas.notna().all(axis=1)].sum(axis=1).astype(int)


def _por_papel(nota: pd.Series, candidatos: Iterable[str]) -> pd.Series:
    """The issuer's score on each of its stocks among ``candidatos``; a stock of an issuer without a score is out."""
    papeis = [papel for papel in candidatos if emissor(papel) in nota.index]
    return pd.Series([nota[emissor(papel)] for papel in papeis], index=papeis, dtype=float)


def piotroski_no_mes(balancos: Balancos, mes: pd.Timestamp, candidatos: Iterable[str]) -> pd.Series:
    """
    Piotroski score of each stock among ``candidatos`` for ``mes``: the issuer's ``f_score`` plus its book-to-market
    percentile (equity / market value) times ``PESO_DO_DESEMPATE``, so the highest F-scores come first and the
    cheapest leads among equals. Shares are ``balancos.valor_mercado["shares_today_units"]`` at ``mes`` and twelve
    months before. Out: Financeiro and no sector, equity <= 0 (Walkshäusl 2020), no market value.
    """
    atual, um_ano, dois_anos = balancos_em_tres_anos(balancos.fundamentos, mes)
    atual = _fora_do_financeiro(atual, balancos.setores)
    acoes = {data: balancos.valor_mercado[balancos.valor_mercado["month"] == data].set_index("issuer")
             for data in (mes, mes - pd.DateOffset(years=1))}
    valor_mercado = acoes[mes]["market_cap"].reindex(atual.index)
    patrimonio = _conta(atual, "equity")
    atual = atual[(patrimonio > 0) & (valor_mercado > 0)]
    nota = f_score(atual, um_ano, dois_anos, acoes[mes]["shares_today_units"],
                   acoes[mes - pd.DateOffset(years=1)]["shares_today_units"])
    book_to_market = (patrimonio / valor_mercado).reindex(nota.index)
    return _por_papel(nota + book_to_market.rank(pct=True) * PESO_DO_DESEMPATE, candidatos)


def multiplo_do_adquirente(fundamentos_mes: pd.DataFrame, valor_mercado: pd.Series,
                           setores: Mapping[str, str]) -> pd.Series:
    """
    Carlisle's Acquirer's Multiple, EV / operating earnings (EBIT), as a percentile rank of EBIT / EV: the highest
    score is the lowest multiple, an operating loss ranks last. Out: Financeiro and no sector, no EBIT, EV <= 0.
    """
    dentro = _fora_do_financeiro(fundamentos_mes, setores)
    ev = valor_da_firma(dentro, valor_mercado)
    rendimento = (_conta(dentro, "ebit") / ev)[ev > 0].dropna()
    return rendimento.rank(pct=True)


def multiplo_do_adquirente_no_mes(balancos: Balancos, mes: pd.Timestamp, candidatos: Iterable[str]) -> pd.Series:
    """The issuer's ``multiplo_do_adquirente`` for ``mes`` on each stock among ``candidatos``."""
    fundamentos_mes, valor_mercado = balancos.no_mes(mes)
    return _por_papel(multiplo_do_adquirente(fundamentos_mes, valor_mercado, balancos.setores), candidatos)
