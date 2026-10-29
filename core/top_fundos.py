"""
top_fundos.py - Copy the top-return equity funds (docs/superpowers/specs/2026-10-29-top-fundos-copia-design.md).

Fund monthly returns from the CVM daily report (the lake's month-end quotas), always within one quota series; the
12-month trailing return m-13 .. m-2 (m-1 skipped: its last report may be filed after m's first session); the top
funds by it; and the stock score = the mean of their CDA issuer weights.
"""

from __future__ import annotations

from collections.abc import Iterable

import pandas as pd

DIAS_NO_FIM_DO_MES = 7         # a month-end quota counts when reported within the month's last 7 calendar days
RETORNO_MINIMO, RETORNO_MAXIMO = -0.60, 1.50   # outside: a data error, the month is missing
MESES_DA_JANELA = 12
MESES_PULADOS = 1              # m-1 is not used


def retornos_mensais(cotas: pd.DataFrame) -> pd.DataFrame:
    """(cnpj, month, r, nav) from the lake's month-end quotas (cnpj, subclass, month = 1st day, date, quota, nav).
    r(m) = quota_s(m) / quota_s(m-1) - 1 within one series s: the class row (subclass null) when quoted at m and m-1,
    else the subclass with the largest nav at m-1 among those quoted at both; nav is s's nav at m."""
    fim_do_mes = cotas["month"] + pd.offsets.MonthEnd(0)
    validas = cotas[(fim_do_mes - cotas["date"]).dt.days < DIAS_NO_FIM_DO_MES]
    validas = validas.assign(serie=validas["subclass"].fillna(""))[["cnpj", "serie", "month", "quota", "nav"]]
    anteriores = validas.assign(month=validas["month"] + pd.DateOffset(months=1)).rename(
        columns={"quota": "quota_antes", "nav": "nav_antes"})
    pares = validas.merge(anteriores, on=["cnpj", "serie", "month"], how="inner")
    pares = pares.assign(e_classe=pares["serie"] == "", r=pares["quota"] / pares["quota_antes"] - 1.0)
    escolhidas = (pares.sort_values(["cnpj", "month", "e_classe", "nav_antes", "serie"],
                                    ascending=[True, True, False, False, True])
                  .drop_duplicates(["cnpj", "month"], keep="first"))
    escolhidas = escolhidas[escolhidas["r"].between(RETORNO_MINIMO, RETORNO_MAXIMO)]
    return escolhidas[["cnpj", "month", "r", "nav"]].reset_index(drop=True)


def retorno_12m(retornos: pd.DataFrame, mes: pd.Timestamp) -> pd.Series:
    """cnpj -> the compounded return over m-13 .. m-2, only for funds with all 12 months."""
    fim = (mes.to_period("M") - 1 - MESES_PULADOS).to_timestamp()
    inicio = (fim.to_period("M") - (MESES_DA_JANELA - 1)).to_timestamp()
    janela = retornos[(retornos["month"] >= inicio) & (retornos["month"] <= fim)]
    por_fundo = janela.groupby("cnpj")["r"]
    completos = por_fundo.count() == MESES_DA_JANELA
    acumulado = (1.0 + janela["r"]).groupby(janela["cnpj"]).prod() - 1.0
    return acumulado[completos]


def top_fundos(acumulado: pd.Series, nav: pd.Series, n: int) -> list[str]:
    """The n funds with the highest trailing return; ties by the larger nav, then the cnpj."""
    tabela = pd.DataFrame({"R": acumulado, "nav": nav.reindex(acumulado.index)}).rename_axis("cnpj").reset_index()
    ordem = tabela.sort_values(["R", "nav", "cnpj"], ascending=[False, False, True], na_position="last")
    return ordem["cnpj"].head(n).tolist()


def nota_media(pesos: Iterable[dict[str, float]], emissores: Iterable[str]) -> pd.Series:
    """issuer -> the mean of the funds' issuer weights (0 where a fund does not hold it), over ``emissores``."""
    indice = pd.Index(list(emissores), name="issuer")
    tabelas = [pd.Series(p, dtype=float).reindex(indice).fillna(0.0) for p in pesos]
    if not tabelas:
        return pd.Series(0.0, index=indice)
    return pd.concat(tabelas, axis=1).mean(axis=1)
