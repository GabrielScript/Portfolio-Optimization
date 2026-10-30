"""
periodos.py - The Conservadora's frozen backtest results by period (whole run, last 5 and 3 years) and by calendar
year, from the saved daily value curves of the pre-registered rerun (`scripts/curvas_conservadora.py`). Descriptive
only: the same configuration as the registered run, so N does not change (DECISOES 122).

Kept out of `core/metrics.py` and `core/lago.py`, which are in the paper guards' paths.
"""

from __future__ import annotations

import pandas as pd

from core.metrics import cagr, comparar_benchmark, cvar_historico, drawdown, sharpe, sortino
from core.volatilidade import volatilidades


def desde_inicio_comum(precos: pd.DataFrame) -> pd.DataFrame:
    """``precos`` (dates x securities) from the first session where every column has a price, each column divided by
    its price on that session (all start at 1); a missing day later keeps the last price."""
    inicio = precos.dropna().index[0]
    curvas = precos.loc[inicio:].ffill()
    return curvas / curvas.iloc[0]


def retornos_por_ano(curvas: pd.DataFrame) -> pd.DataFrame:
    """Each calendar year's simple return of every column: last session of the year over the last session of the
    year before (the first year starts at the curve's first value)."""
    fechamentos = curvas.groupby(curvas.index.year).last()
    anteriores = fechamentos.shift(1)
    anteriores.iloc[0] = curvas.iloc[0]
    return fechamentos / anteriores - 1


def _medidas(curvas: pd.DataFrame) -> dict[tuple[str, str], float]:
    medidas = {}
    for nome in curvas.columns:
        valores = curvas[nome].dropna()
        medidas[(nome, "CAGR")] = cagr(valores)
        medidas[(nome, "Vol anual")] = volatilidades(valores).anual
        medidas[(nome, "MDD")] = drawdown(valores)[1]
    return medidas


def medidas_por_periodo(curvas: pd.DataFrame, anos: tuple[int, ...] = (5, 3)) -> pd.DataFrame:
    """CAGR, annualized volatility and maximum drawdown of every column over the whole curve and over the last
    ``anos`` years (sessions strictly after the last date minus n years), one row per period."""
    fim = curvas.index[-1]
    trechos = {"Período inteiro": curvas,
               **{f"Últimos {n} anos": curvas.loc[curvas.index > fim - pd.DateOffset(years=n)] for n in anos}}
    linhas = {rotulo: {("Início", ""): trecho.index[0], ("Fim", ""): trecho.index[-1], **_medidas(trecho)}
              for rotulo, trecho in trechos.items()}
    tabela = pd.DataFrame(linhas).T
    tabela.columns = pd.MultiIndex.from_tuples(tabela.columns)
    return tabela


DIAS_POR_ANO = 252


def _medidas_do_ano(valores: pd.Series, cdi: pd.Series, referencia: pd.Series) -> dict[str, float]:
    """One column's year: ``valores`` = its base value (the year before's last close, or its first price) followed
    by the year's sessions; ``cdi`` and ``referencia`` (the benchmark's daily returns) on the same sessions."""
    diarios = valores.pct_change().iloc[1:]
    excesso = diarios - cdi.reindex(diarios.index).fillna(0.0)
    retorno = float(valores.iloc[-1] / valores.iloc[0] - 1)
    vol = float(diarios.std(ddof=1) * DIAS_POR_ANO ** 0.5)
    queda = float((excesso.clip(upper=0.0) ** 2).mean() ** 0.5 * DIAS_POR_ANO ** 0.5)
    pior_queda = float((valores / valores.cummax() - 1).min())
    fechamentos = valores.groupby(valores.index.to_period("M")).last()
    meses = pd.concat([valores.iloc[:1], fechamentos.iloc[(0 if valores.index[0].year < valores.index[-1].year
                                                          or len(fechamentos) == 0 else 0):]])
    mensais = fechamentos / pd.Series([valores.iloc[0], *fechamentos.iloc[:-1]], index=fechamentos.index) - 1
    alinhado = pd.concat([diarios, referencia.reindex(diarios.index)], axis=1).dropna()
    beta = float(alinhado.cov().iloc[0, 1] / alinhado.iloc[:, 1].var()) if len(alinhado) > 1 else float("nan")
    del meses
    return {"Retorno": retorno,
            "Excesso sobre o CDI": retorno - float((1 + cdi.reindex(diarios.index).fillna(0.0)).prod() - 1),
            "Vol": vol,
            "Sharpe": float(excesso.mean() * DIAS_POR_ANO / vol) if vol > 0 else float("nan"),
            "Sortino": float(excesso.mean() * DIAS_POR_ANO / queda) if queda > 0 else float("nan"),
            "Queda máx.": pior_queda,
            "Calmar": retorno / abs(pior_queda) if pior_queda < 0 else float("nan"),
            "Pior mês": float(mensais.min()),
            "Beta (BOVA11)": beta,
            "Sessões": len(diarios)}


def medidas_por_ano(curvas: pd.DataFrame, cdi: pd.Series, referencia: str = "BOVA11") -> pd.DataFrame:
    """(year, column) -> return, excess over the CDI, annualized vol, Sharpe and Sortino (daily returns over the
    day's CDI, x252), maximum drawdown, Calmar (return / |drawdown|), worst month, beta to ``referencia`` and the
    sessions measured. Each year starts at the column's last value of the year before (or at its first price, for a
    column that starts that year). ``cdi``: decimal per day, indexed by date."""
    retornos_ref = curvas[referencia].pct_change()
    linhas = {}
    for nome in curvas.columns:
        serie = curvas[nome].dropna()
        for ano in sorted(set(serie.index.year)):
            antes = serie.loc[serie.index.year < ano]
            do_ano = serie.loc[serie.index.year == ano]
            valores = pd.concat([antes.iloc[-1:], do_ano]) if len(antes) else do_ano
            if len(valores) > 1:
                linhas[(ano, nome)] = _medidas_do_ano(valores, cdi, retornos_ref)
    tabela = pd.DataFrame(linhas).T
    tabela.index.names = ["Ano", "Estratégia"]
    return tabela


def resumo_do_periodo(curvas: pd.DataFrame, cdi: pd.Series, referencia: str = "BOVA11") -> pd.DataFrame:
    """One row per column, from its own first value to the end: CAGR, vol, Sharpe and Sortino over the CDI, maximum
    drawdown, Calmar (CAGR / |drawdown|), the CAPM against ``referencia`` (beta, Jensen's annual alpha with its
    Newey-West t, information ratio, ``core.metrics.comparar_benchmark``), the daily 95% CVaR, the best and worst
    calendar year and the years above ``referencia`` (``medidas_por_ano``). ``cdi``: decimal per day."""
    anos = medidas_por_ano(curvas, cdi, referencia)["Retorno"].unstack()
    linhas = {}
    for nome in curvas.columns:
        valores = curvas[nome].dropna()
        diarios = valores.pct_change().dropna()
        excesso = diarios - cdi.reindex(diarios.index).fillna(0.0)
        crescimento, queda = cagr(valores), drawdown(valores)[1]
        capm = comparar_benchmark(valores, curvas[referencia].dropna(), cdi)
        ambos = pd.DataFrame({"x": anos[nome], "ref": anos[referencia]}).dropna()
        linhas[nome] = {"Desde": valores.index[0], "CAGR": crescimento, "Vol": volatilidades(valores).anual,
                        "Sharpe": sharpe(excesso), "Sortino": sortino(excesso), "Queda máx.": queda,
                        "Calmar": crescimento / abs(queda) if queda < 0 else float("nan"),
                        "Beta (BOVA11)": capm.beta, "Alfa a.a. (BOVA11)": capm.alfa_anual,
                        "t do alfa": capm.t_stat_alfa, "Information ratio": capm.information_ratio,
                        "CVaR 95% diário": cvar_historico(diarios),
                        "Melhor ano": float(anos[nome].max()), "Pior ano": float(anos[nome].min()),
                        "Anos acima do BOVA11": f"{int((ambos['x'] > ambos['ref']).sum())} de {len(ambos)}"}
    return pd.DataFrame(linhas).T
