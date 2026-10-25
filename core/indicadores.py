"""
indicadores.py - The screen's per-stock table: valuation multiples from the lake's CVM statements (``core.valor``'s
accounts: balances at the period end, flows over 12 months) and risk measures from the daily prices. Information only
(DECISOES 124): the Alepha plan is the registered Conservadora, which uses none of these.

Known data limit (README): a few market values right after a listing are a thousand times too small (the reverse
slip of DECISOES 119), so their multiples are wrong by the same factor.
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

from core.sinais import dividend_yield_12m
from core.valor import SETOR_FORA, _conta, fundamentos_no_mes


def multiplos(fundamentos_mes: pd.DataFrame, valor_mercado: pd.Series) -> pd.DataFrame:
    """Per issuer (``fundamentos_mes``: issuer x account, ``core.valor.fundamentos_no_mes``): P/L (empty with a loss),
    P/VP, ROE, net margin and net debt / EBITDA (EBIT + |depreciation|; empty when EBITDA <= 0)."""
    mercado = valor_mercado.reindex(fundamentos_mes.index)
    lucro = _conta(fundamentos_mes, "net_income_parent").fillna(_conta(fundamentos_mes, "net_income"))
    patrimonio = _conta(fundamentos_mes, "equity") - _conta(fundamentos_mes, "minority_interest", True)
    divida = _conta(fundamentos_mes, "short_term_debt", True) + _conta(fundamentos_mes, "long_term_debt", True)
    caixa = _conta(fundamentos_mes, "cash", True) + _conta(fundamentos_mes, "short_term_investments", True)
    ebitda = _conta(fundamentos_mes, "ebit") + _conta(fundamentos_mes, "depreciation", True).abs()
    return pd.DataFrame({"P/L": (mercado / lucro).where(lucro > 0),
                         "P/VP": (mercado / patrimonio).where(patrimonio > 0),
                         "ROE": (lucro / patrimonio).where(patrimonio > 0),
                         "Margem líquida": (lucro / _conta(fundamentos_mes, "revenue")).replace([np.inf, -np.inf],
                                                                                              np.nan),
                         "Dív. líq./EBITDA": ((divida - caixa) / ebitda).where(ebitda > 0)})


PREGOES_NO_ANO = 252


def risco_por_acao(precos: pd.DataFrame, referencia: str) -> pd.DataFrame:
    """Per column of ``precos`` (daily total-return closes, NaN on a session without trades): the last 252 sessions'
    return, annualized volatility and beta against ``referencia`` (BOVA11), and the maximum drawdown over the whole
    window given. A stock with no price at the start of the year has no 12-month return; the beta uses the sessions
    where both the stock and ``referencia`` have a return."""
    retornos = precos.pct_change(fill_method=None).iloc[-PREGOES_NO_ANO:]
    corridos = precos.ffill()
    mercado = retornos[referencia]

    def beta(serie: pd.Series) -> float:
        juntos = pd.concat([serie, mercado], axis=1).dropna()
        return float(juntos.iloc[:, 0].cov(juntos.iloc[:, 1]) / juntos.iloc[:, 1].var())
    return pd.DataFrame({"Retorno 12m": corridos.iloc[-1] / corridos.iloc[-PREGOES_NO_ANO - 1] - 1,
                         "Vol anual": retornos.std() * np.sqrt(PREGOES_NO_ANO),
                         "Beta": retornos.apply(beta),
                         "Queda máx.": (precos.ffill() / precos.ffill().cummax() - 1).min()})


def tabela_por_acao(precos: pd.DataFrame, info: dict, fundamentos: pd.DataFrame | None,
                    referencia: str) -> pd.DataFrame:
    """Per stock with a price (index: ticker without ".SA"; ``referencia`` left out): name and sector (``info``), the
    last close and its date, ``risco_por_acao`` (drawdown over the window given, "Queda máx. (3 anos)" for the panel's
    three years) and the columns of ``fundamentos`` (ticker -> multiples, "Balanço de")."""
    com_dados = precos.loc[:, precos.notna().any()]
    risco = risco_por_acao(com_dados, referencia).drop(index=referencia).rename(
        columns={"Queda máx.": "Queda máx. (3 anos)"})
    ultimos = com_dados.ffill().iloc[-1]
    datas = com_dados.apply(lambda serie: serie.last_valid_index())
    tabela = pd.DataFrame({"Empresa": [info.get(ticker, {}).get("nome", "") for ticker in risco.index],
                           "Setor": [info.get(ticker, {}).get("setor", "") for ticker in risco.index],
                           "Preço (R$)": ultimos.reindex(risco.index).to_numpy(),
                           "Fechamento de": datas.reindex(risco.index).to_numpy()},
                          index=[ticker.removesuffix(".SA") for ticker in risco.index]).join(
        risco.rename(index=lambda ticker: ticker.removesuffix(".SA")))
    return tabela if fundamentos is None else tabela.join(fundamentos)


def tabela_de_fundamentos(fundamentos: pd.DataFrame, valor_mercado: pd.DataFrame, proventos: pd.DataFrame,
                          tickers: list[str], hoje: date) -> pd.DataFrame:
    """Per ticker of ``tickers`` (index without ".SA") whose issuer has a statement: ``multiplos`` with the statements
    received up to ``hoje`` (``core.valor.fundamentos_no_mes``) and the latest market value, the 12-month dividend
    yield (``core.sinais.dividend_yield_12m``) and the period end of the latest statement. Banks and insurers
    (``sector`` "Financeiro") get no multiples: the lake's equity line is not their equity. Inputs: the lake's
    ``fundamentals``, ``market_caps`` and ``dividends`` exports."""
    amanha = pd.Timestamp(hoje) + pd.Timedelta(days=1)
    do_mes = fundamentos_no_mes(fundamentos, amanha)
    ultimo_mes = pd.to_datetime(valor_mercado["month"]).max()
    mercado = valor_mercado[pd.to_datetime(valor_mercado["month"]) == ultimo_mes].set_index("issuer")["market_cap"]
    por_emissor = multiplos(do_mes, mercado)
    conhecidos = fundamentos[fundamentos["available_from"] < amanha]
    balanco = conhecidos.groupby("issuer")["dt_refer"].max()
    papeis = [ticker.removesuffix(".SA") for ticker in tickers if ticker.removesuffix(".SA")[:4] in por_emissor.index]
    tabela = por_emissor.reindex([papel[:4] for papel in papeis]).set_axis(papeis)
    if "sector" in fundamentos.columns:      # banks and insurers: another chart of accounts (core.valor.SETOR_FORA)
        financeiros = set(fundamentos.loc[fundamentos["sector"] == SETOR_FORA, "issuer"])
        tabela.loc[[papel[:4] in financeiros for papel in papeis]] = np.nan
    tabela["DY 12m"] = dividend_yield_12m(proventos, amanha, papeis)
    tabela["Balanço de"] = balanco.reindex([papel[:4] for papel in papeis]).to_numpy()
    return tabela
