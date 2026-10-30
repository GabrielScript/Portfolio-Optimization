"""
tres_fundos.py - The three funds the user named on 2026-10-30 (Capyvara FIA, AMI FIF Ações, Trend Bolsa Americana
Alavancado; DECISOES 189): their 7 combinations, the return of investing in them (their CVM quotas) and of copying
their public CDA holdings. Descriptive only: the funds were picked by name after their returns were seen, so nothing
here is a trial or a gate (N does not change).

The copy follows attempt 52's publication rule (``core.melhores_ideias``): at month m each member's latest book that
is entirely public by m, its stock and BDR positions weighted by value, the members in equal parts.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from itertools import combinations

import pandas as pd
from scipy import stats

from core.melhores_ideias import carteira_usada, publico_a_partir
from core.metrics import _erro_padrao_intercepto_newey_west      # the same HAC error as the daily CAPM

APLICACOES = ("Ações", "Ações e outros TVM cedidos em empréstimo", "Brazilian Depository Receipt - BDR")
SEM_CORTE = pd.Timestamp("2100-12-31")


def combinacoes(fundos: Sequence[str]) -> list[tuple[str, ...]]:
    """All of them together, then every pair, then each alone (in ``fundos``' order)."""
    return [grupo for tamanho in range(len(fundos), 0, -1) for grupo in combinations(fundos, tamanho)]


def retorno_da_combinacao(retornos: pd.DataFrame, membros: Iterable[str]) -> pd.Series:
    """Equal parts of each member, rebalanced monthly (``retornos``: months x funds), over the months in which every
    member has a return, from the first such month to the first one where any is missing."""
    tabela = retornos[list(membros)]
    completas = tabela.notna().all(axis=1)
    if not completas.any():
        return pd.Series(dtype=float)
    inicio = completas.idxmax()
    depois = completas.loc[inicio:]
    fim = depois.index[-1] if depois.all() else depois[~depois].index[0] - pd.DateOffset(months=1)
    return tabela.loc[inicio:fim].mean(axis=1)


def pesos_copiados(posicoes: pd.DataFrame, mes: pd.Timestamp, membros: Iterable[str]) -> pd.Series:
    """ticker -> weight of the copy at ``mes``: each member's latest book public by then (``posicoes``: cnpj, month
    month-end, ticker, value, confid_until), by value, the members that have one in equal parts; empty if none."""
    usadas = carteira_usada(publico_a_partir(posicoes), mes, SEM_CORTE)
    livros = []
    for cnpj in membros:
        if cnpj not in usadas.index:
            continue
        livro = posicoes[(posicoes["cnpj"] == cnpj) & (posicoes["month"] == usadas[cnpj])]
        valores = livro.groupby("ticker")["value"].sum()
        livros.append(valores / valores.sum())
    if not livros:
        return pd.Series(dtype=float)
    return (pd.concat(livros, axis=1).fillna(0.0).sum(axis=1) / len(livros)).rename(None)


def retorno_da_copia(pesos: pd.Series, retornos: pd.Series) -> float:
    """The month's return of ``pesos`` over the names that have a return in ``retornos`` (weights renormalized)."""
    comuns = pesos[pesos.index.isin(retornos.dropna().index)]
    if comuns.empty or comuns.sum() <= 0:
        return float("nan")
    return float((comuns * retornos[comuns.index]).sum() / comuns.sum())


def linhas_do_blc4(linhas: Iterable[str], cnpjs: set[str]) -> pd.DataFrame:
    """The stock and BDR positions (``APLICACOES``, lent shares included) of ``cnpjs`` in a CVM CDA ``BLC_4`` file
    (lines, header first; the pre-CVM 175 header too): cnpj, month (month-end), ticker, value (R$ at market) and
    confid_until."""
    iterador = iter(linhas)
    colunas = next(iterador).rstrip("\r\n").split(";")
    colunas = ["CNPJ_FUNDO_CLASSE" if nome == "CNPJ_FUNDO" else nome for nome in colunas]    # before CVM 175
    pos = {nome: colunas.index(nome) for nome in ("CNPJ_FUNDO_CLASSE", "DT_COMPTC", "TP_APLIC", "CD_ATIVO",
                                                    "VL_MERC_POS_FINAL", "DT_CONFID_APLIC")}
    registros = []
    for linha in iterador:
        if not any(cnpj in linha for cnpj in cnpjs):
            continue
        campos = linha.rstrip("\r\n").split(";")
        if campos[pos["CNPJ_FUNDO_CLASSE"]] not in cnpjs or campos[pos["TP_APLIC"]] not in APLICACOES:
            continue
        registros.append({"cnpj": campos[pos["CNPJ_FUNDO_CLASSE"]], "month": pd.Timestamp(campos[pos["DT_COMPTC"]]),
                          "ticker": campos[pos["CD_ATIVO"]].strip(), "value": float(campos[pos["VL_MERC_POS_FINAL"]]),
                          "confid_until": pd.Timestamp(campos[pos["DT_CONFID_APLIC"]]) if campos[
                              pos["DT_CONFID_APLIC"]] else pd.NaT})
    return pd.DataFrame(registros, columns=["cnpj", "month", "ticker", "value", "confid_until"])



def medidas_mensais(retornos: pd.Series, bova: pd.Series, cdi: pd.Series) -> dict[str, float]:
    """Monthly ``retornos`` (and BOVA11's and the CDI's on the same months): total return, CAGR, vol x sqrt(12),
    Sharpe over the CDI, maximum drawdown from the starting R$ 1, and the CAPM on the excesses over the CDI (beta,
    12 x alpha, alpha's Newey-West t, as ``core.metrics.comparar_benchmark`` does on daily data)."""
    meses = retornos.dropna().index
    r, b, c = retornos[meses], bova.reindex(meses), cdi.reindex(meses).fillna(0.0)
    riqueza = (1 + r).cumprod()
    vol = float(r.std(ddof=1) * 12 ** 0.5)
    regressao = stats.linregress((b - c).to_numpy(), (r - c).to_numpy())
    erro = _erro_padrao_intercepto_newey_west((b - c).to_numpy(), (r - c).to_numpy(), regressao.intercept,
                                              regressao.slope)
    return {"Meses": len(meses), "Retorno total": float(riqueza.iloc[-1] - 1),
            "CAGR": float(riqueza.iloc[-1] ** (12 / len(meses)) - 1), "Vol": vol,
            "Sharpe": float((r - c).mean() * 12 / vol) if vol > 0 else float("nan"),
            "Queda máx.": float((riqueza / riqueza.cummax().clip(lower=1.0) - 1).min()),
            "Beta (BOVA11)": float(regressao.slope), "Alfa a.a. (BOVA11)": float(regressao.intercept * 12),
            "t do alfa": float(regressao.intercept / erro) if erro > 0 else float("nan")}
