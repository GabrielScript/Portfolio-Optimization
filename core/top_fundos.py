"""
top_fundos.py - Copy the top-return equity funds (docs/superpowers/specs/2026-10-29-top-fundos-copia-design.md).

Fund monthly returns from the CVM daily report (the lake's month-end quotas), always within one quota series; the
12-month trailing return m-13 .. m-2 (m-1 skipped: its last report may be filed after m's first session); the top
funds by it; and the stock score = the mean of their CDA issuer weights.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import pandas as pd

from core.melhores_ideias import carteira_usada
from core.risco import emissor

DIAS_NO_FIM_DO_MES = 7         # a month-end quota counts when reported within the month's last 7 calendar days
RETORNO_MINIMO, RETORNO_MAXIMO = -0.60, 1.50   # outside: a data error, the month is missing
MESES_DA_JANELA = 12
MESES_PULADOS = 1              # m-1 is not used
TOP_N = 15
FUNDOS_MINIMOS = 30            # a month with fewer qualifying funds, or fewer than TOP_N ranked ones, is not read


@dataclass(frozen=True)
class Carteiras:
    """The CDA side exactly as attempt 52 builds it (core.melhores_ideias)."""
    ideias: pd.DataFrame       # ideias_por_carteira: (cnpj, portfolio month-end) -> qualifica, pesos, ...
    publico: pd.Series         # publico_a_partir: (cnpj, portfolio month-end) -> first month it is public
    bench: pd.DataFrame        # referencia: (portfolio month-end, issuer, w_b)
    corte: pd.Timestamp        # portfolios after this month-end are not used


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


def fim_da_janela(mes: pd.Timestamp) -> pd.Timestamp:
    """The last month of the ranking window: m-2 (m-1 is skipped)."""
    return (mes.to_period("M") - 1 - MESES_PULADOS).to_timestamp()


def inicio_da_janela(mes: pd.Timestamp) -> pd.Timestamp:
    """The first month of the ranking window: m-13."""
    return (fim_da_janela(mes).to_period("M") - (MESES_DA_JANELA - 1)).to_timestamp()


def retorno_12m(retornos: pd.DataFrame, mes: pd.Timestamp) -> pd.Series:
    """cnpj -> the compounded return over m-13 .. m-2, only for funds with all 12 months."""
    janela = retornos[retornos["month"].between(inicio_da_janela(mes), fim_da_janela(mes))]
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


def fundos_qualificados(carteiras: Carteiras, mes: pd.Timestamp) -> pd.DataFrame:
    """The qualifying funds at ``mes``, each at its latest portfolio public by then (as attempt 52's panel):
    index cnpj, columns month (the portfolio month-end) and pesos."""
    usadas = carteira_usada(carteiras.publico, mes, carteiras.corte)
    chaves = [(cnpj, t) for cnpj, t in usadas.items() if (cnpj, t) in carteiras.ideias.index]
    escolhidas = carteiras.ideias.loc[chaves] if chaves else carteiras.ideias.iloc[:0]
    escolhidas = escolhidas[escolhidas["qualifica"]]
    return escolhidas.reset_index(level="month")[["month", "pesos"]]


def peso_ativo_medio(fundos: pd.DataFrame, bench: pd.DataFrame, emissores: Iterable[str]) -> pd.Series:
    """issuer -> the mean over ``fundos`` (columns month, pesos) of w_f - w_b, each fund against the benchmark of its
    own portfolio month (0 where neither holds the issuer)."""
    indice = pd.Index(list(emissores), name="issuer")
    por_mes = {t: grupo.set_index("issuer")["w_b"] for t, grupo in bench.groupby("month")}
    ativos = [pd.Series(pesos, dtype=float).reindex(indice).fillna(0.0)
              - por_mes.get(t, pd.Series(dtype=float)).reindex(indice).fillna(0.0)
              for t, pesos in zip(fundos["month"], fundos["pesos"])]
    if not ativos:
        return pd.Series(0.0, index=indice)
    return pd.concat(ativos, axis=1).mean(axis=1)


def painel_top_fundos(carteiras: Carteiras, retornos: pd.DataFrame, universo: pd.DataFrame,
                      meses: Iterable[pd.Timestamp], n: int = TOP_N) -> pd.DataFrame:
    """(month, issuer, score, score_all, active, qualifying, with_quotas, ranked), one row per issuer of each month's
    universe. score = the mean issuer weight of the n ranked qualifying funds with the best 12-month return;
    score_all = the same mean over every ranked qualifying fund; active (reported only) = the mean of w_f - w_b over
    the n; with_quotas = qualifying funds with any return in the window (coverage). Computed for every month: the
    reader drops the thin ones by the counts."""
    emissores_do_mes = (universo.assign(month=pd.to_datetime(universo["month"]),
                                        issuer=universo["security"].map(emissor))
                        .groupby("month")["issuer"].apply(lambda s: sorted(set(s))))
    partes = []
    for mes in meses:
        fundos = fundos_qualificados(carteiras, mes)
        dos_fundos = retornos[retornos["cnpj"].isin(fundos.index)]
        acumulado = retorno_12m(dos_fundos, mes)
        fim = fim_da_janela(mes)
        na_janela = dos_fundos[dos_fundos["month"].between(inicio_da_janela(mes), fim)]
        no_fim = dos_fundos[dos_fundos["month"] == fim]
        melhores = top_fundos(acumulado, no_fim.set_index("cnpj")["nav"], n)
        emissores = emissores_do_mes.get(mes, [])
        partes.append(pd.DataFrame({
            "month": mes, "issuer": emissores,
            "score": nota_media(fundos.loc[melhores, "pesos"], emissores).to_numpy(),
            "score_all": nota_media(fundos.loc[acumulado.index, "pesos"], emissores).to_numpy(),
            "active": peso_ativo_medio(fundos.loc[melhores], carteiras.bench, emissores).to_numpy(),
            "qualifying": len(fundos), "with_quotas": na_janela["cnpj"].nunique(), "ranked": len(acumulado)}))
    return pd.concat(partes, ignore_index=True)


def meses_cheios(painel: pd.DataFrame) -> list[pd.Timestamp]:
    """The panel's months with at least ``FUNDOS_MINIMOS`` qualifying funds and ``TOP_N`` ranked ones."""
    contagens = painel.groupby("month")[["qualifying", "ranked"]].first()
    cheios = (contagens["qualifying"] >= FUNDOS_MINIMOS) & (contagens["ranked"] >= TOP_N)
    return list(contagens.index[cheios])
