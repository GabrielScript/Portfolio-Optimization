"""
us_indicadores.py - The US stocks' indicators and macro for the screen. Information only, like the B3 panel (DECISOES 124).
Multiples are the app's own (``core.indicadores.tabela_de_fundamentos``) on the US lake's gold, which has the same shape;
risk reads monthly returns, since the dead names have no daily history. Portuguese labels, shown as they are.
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

from core.indicadores import tabela_de_fundamentos
from core.valor import SETOR_FORA

MESES_NO_ANO = 12
MESES_DA_JANELA = 36
MINIMO_PARA_VOL = 6
MINIMO_PARA_BETA = 24
COLUNAS = ["Ticker", "Empresa", "Setor", "Valor de mercado", "Retorno 12m", "Vol anual", "Beta", "Queda máx. (3 anos)", "P/L",
           "P/VP", "ROE", "Margem líquida", "Dív. líq./EBITDA", "DY 12m", "Balanço de"]
SERIES_DO_FRED = (("Juro de 3 meses", "DGS3MO", "% ao ano"), ("Juro de 10 anos", "DGS10", "% ao ano"),
                  ("Prêmio de crédito BAA", "BAA10Y", "p.p. sobre 10 anos"), ("Desemprego", "UNRATE", "%"),
                  ("Volatilidade VIX", "VIXCLS", "índice"), ("Condições financeiras NFCI", "NFCI", "índice, negativo = folgadas"),
                  ("Recessão NBER", "USREC", "1 = em recessão"))
FATORES = (("Tamanho em 12 meses (SMB)", "smb"), ("Valor em 12 meses (HML)", "hml"), ("Rentabilidade em 12 meses (RMW)", "rmw"),
           ("Investimento em 12 meses (CMA)", "cma"), ("Momentum em 12 meses", "mom"))


def _beta(retorno: pd.Series, mercado: pd.Series) -> float:
    juntos = pd.concat([retorno, mercado], axis=1, join="inner").dropna()
    if len(juntos) < MINIMO_PARA_BETA:
        return float("nan")
    return float(juntos.iloc[:, 0].cov(juntos.iloc[:, 1]) / juntos.iloc[:, 1].var())


def risco_mensal(retornos: pd.DataFrame, mercado: pd.Series) -> pd.DataFrame:
    """Per column of month x security: the 12-month return (all 12 known), annual volatility (6 of the last 12), beta over
    the 36 months (24 in common with ``mercado``) and the maximum drawdown of those 36 months."""
    janela = retornos.iloc[-MESES_DA_JANELA:]
    ano = retornos.iloc[-MESES_NO_ANO:]
    corridos = (1.0 + janela.fillna(0.0)).cumprod()
    return pd.DataFrame({"Retorno 12m": (1.0 + ano).prod(min_count=MESES_NO_ANO).where(ano.notna().all()) - 1.0,
                         "Vol anual": ano.std().where(ano.notna().sum() >= MINIMO_PARA_VOL) * np.sqrt(MESES_NO_ANO),
                         "Beta": pd.Series({nome: _beta(janela[nome], mercado) for nome in janela.columns}, dtype=float),
                         "Queda máx. (3 anos)": (corridos / corridos.cummax() - 1.0).min().where(janela.notna().any())})


def tabela_de_indicadores(retornos: pd.DataFrame, mercado: pd.Series, universo: list[str], papeis: pd.DataFrame,
                          fundamentos: pd.DataFrame, valor_mercado: pd.DataFrame, proventos: pd.DataFrame,
                          hoje: date) -> pd.DataFrame:
    """One row per security of ``universo``, the largest first: who it is, its last market value, risk and multiples."""
    meus = papeis.set_index("security").reindex(universo)
    ultimo = pd.to_datetime(valor_mercado["month"]).max()
    capital = valor_mercado[pd.to_datetime(valor_mercado["month"]) == ultimo].set_index("issuer")["market_cap"]
    tabela = pd.DataFrame({"Ticker": meus["symbol"], "Empresa": meus["name"], "Setor": meus["sector"],
                           "Valor de mercado": capital.reindex([papel[:4] for papel in universo]).to_numpy()}, index=universo)
    risco = risco_mensal(retornos.reindex(columns=universo), mercado)
    # without `sector`: the B3 blanks banks because the CVM's line is not their equity, and EDGAR's is
    multiplos = tabela_de_fundamentos(fundamentos.drop(columns="sector", errors="ignore"), valor_mercado, proventos, universo, hoje)
    junta = tabela.join(risco).join(multiplos).reindex(columns=COLUNAS)
    junta.loc[junta["Setor"] == SETOR_FORA, "Dív. líq./EBITDA"] = np.nan    # a bank has no EBITDA to speak of
    return junta.sort_values("Valor de mercado", ascending=False, na_position="last").reset_index(drop=True)


def _ultimo(serie: pd.Series) -> tuple[float, pd.Timestamp] | None:
    conhecidos = serie.dropna()
    return None if conhecidos.empty else (float(conhecidos.iloc[-1]), conhecidos.index[-1])


def tabela_macro(macro: pd.DataFrame, fatores: pd.DataFrame) -> pd.DataFrame:
    """`Indicador, Valor, Unidade, Mês`: each FRED series' last known month and French's factors over 12 months."""
    series = macro.set_index(pd.DatetimeIndex(pd.to_datetime(macro["month"]))).sort_index()
    linhas: list[tuple] = []
    def incluir(nome: str, serie: pd.Series, unidade: str) -> None:
        achado = _ultimo(serie)
        if achado is not None:
            linhas.append((nome, achado[0], unidade, achado[1]))
    precos = series["CPIAUCSL"].dropna()
    incluir("Inflação em 12 meses (CPI)", (precos / precos.shift(MESES_NO_ANO, freq="MS").reindex(precos.index) - 1.0) * 100.0,
            "%")
    for nome, coluna, unidade in SERIES_DO_FRED:
        incluir(nome, series[coluna], unidade)
    incluir("Juro de 10 anos menos 3 meses", series["DGS10"] - series["DGS3MO"], "p.p.")
    meses = fatores.set_index(pd.DatetimeIndex(pd.to_datetime(fatores["month"]))).sort_index().iloc[-MESES_NO_ANO:]
    def em_12_meses(retornos: pd.Series) -> float:
        return float((1.0 + retornos).prod() - 1.0) * 100.0
    fim = meses.index[-1]
    linhas.append(("Mercado em 12 meses (French)", em_12_meses(meses["mkt_rf"] + meses["rf"]), "%", fim))
    linhas.extend((nome, em_12_meses(meses[coluna]), "%", fim) for nome, coluna in FATORES)
    return pd.DataFrame(linhas, columns=["Indicador", "Valor", "Unidade", "Mês"])
