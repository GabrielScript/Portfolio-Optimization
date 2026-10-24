"""
sentimento.py - The screen's news, sentiment and filings panel, from the lake's forward log (``b3lake.newslog``, Jev
scores of the questions ``items-v1``) and the CVM IPE filings (``b3lake.ipe``). Information only: the user asked to
see it on 2026-10-23 (DECISOES 124); nothing here reaches the optimizer, the paper routines or the Fundo Alepha plan,
and the pre-registered tests of ``docs/preregistro-sentimento-2026-10-23.md`` are unchanged.

The Jev questions have no ticker: per-stock coverage comes from the CVM filings, which name the company.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import pandas as pd

VERSAO_DAS_PERGUNTAS = "items-v1"
TOPICOS = ("election_politics", "monetary_policy", "fiscal_policy", "company_specific", "commodities_global",
           "currency", "personal_finance", "other")


def notas_por_item(linhas: Iterable[dict]) -> pd.DataFrame:
    """One row per scored item (index ``key``): ``relevante`` = P(relevant), ``tom`` = P(positive) - P(negative) of
    the market tone, ``eleicao`` = P(about the 2026 election), ``forte`` = P(intensity Major or Extraordinary), and
    the probability of each topic. Rows of other question versions are skipped."""
    registros = {}
    for linha in linhas:
        if linha.get("target") != "item" or linha.get("questions_version") != VERSAO_DAS_PERGUNTAS:
            continue
        respostas = linha["answers"]
        tom = respostas["market_tone"]["probabilities"]
        intensidade = respostas["intensity"]["probabilities"]
        registros[linha["key"]] = {
            "relevante": respostas["relevant"]["noul"], "tom": tom["positive"] - tom["negative"],
            "eleicao": respostas["election"]["noul"], "forte": intensidade["2"] + intensidade["3"],
            **{topico: respostas["topic"]["probabilities"].get(topico, 0.0) for topico in TOPICOS}}
    colunas = ["relevante", "tom", "eleicao", "forte", *TOPICOS]
    return pd.DataFrame.from_dict(registros, orient="index", columns=colunas).astype(float).rename_axis("key")


LIMIAR_RELEVANTE = 0.5                  # the pre-registration's "relevant >= 0.5" (Test B)
FUSO = "America/Sao_Paulo"
TOPICOS_DO_PAINEL = ("election_politics", "monetary_policy", "fiscal_policy", "currency", "commodities_global",
                     "company_specific")


def tom_diario(itens: pd.DataFrame, notas: pd.DataFrame) -> pd.DataFrame:
    """Per Brasília calendar day of ``fetched_at``: ``itens`` (all logged), ``relevantes`` (scored, P(relevant) >=
    0.5), and over the relevant ones the mean ``tom``, the share about the election (``eleicao`` >= 0.5), the mean
    ``forte`` and the mean probability of each topic of ``TOPICOS_DO_PAINEL``. Days without a relevant item have
    NaN means."""
    dia = pd.to_datetime(itens["fetched_at"], utc=True).dt.tz_convert(FUSO).dt.tz_localize(None).dt.normalize()
    quadro = itens.assign(dia=dia.to_numpy()).join(notas, on="key")
    relevantes = quadro[quadro["relevante"] >= LIMIAR_RELEVANTE]
    por_dia = relevantes.groupby("dia")
    medias = por_dia[["tom", "forte", *TOPICOS_DO_PAINEL]].mean()
    medias["eleicao"] = por_dia["eleicao"].apply(lambda serie: float((serie >= LIMIAR_RELEVANTE).mean()))
    contagens = pd.DataFrame({"itens": quadro.groupby("dia").size(), "relevantes": por_dia.size()}).fillna(0)
    return contagens.astype(int).join(medias).rename_axis("dia")


CATEGORIAS_MATERIAIS = ("Fato Relevante", "Comunicado ao Mercado")     # b3lake.jevscore.MATERIAL_CATEGORIES


def fatos_recentes(ipe: pd.DataFrame, setores: pd.DataFrame, tickers: Iterable[str], desde) -> pd.DataFrame:
    """The material CVM filings (``CATEGORIAS_MATERIAIS``) delivered on or after ``desde`` by the issuers of
    ``tickers`` (app format, issuer = first 4 letters), newest first: ``ticker``, ``empresa``, ``categoria``,
    ``assunto``, ``data`` and ``link`` (the filing on the CVM's site). ``setores``: the lake's issuer-CVM table."""
    por_emissor = {ticker.removesuffix(".SA")[:4]: ticker.removesuffix(".SA") for ticker in tickers}
    # a renamed company keeps one CVM code under two prefixes (ELET and AXIA): the app lists one of them
    do_app = setores.dropna(subset=["code_cvm"]).assign(ticker=lambda quadro: quadro["issuer"].map(por_emissor))
    ticker_do_codigo = do_app.dropna(subset=["ticker"]).drop_duplicates("code_cvm").set_index("code_cvm")["ticker"]
    materiais = ipe[ipe["categoria"].isin(CATEGORIAS_MATERIAIS) & (ipe["data_entrega"] >= desde)]
    ticker = materiais["codigo_cvm"].map(ticker_do_codigo)
    fatos = pd.DataFrame({"ticker": ticker, "empresa": materiais["nome_companhia"],
                          "categoria": materiais["categoria"], "assunto": materiais["assunto"],
                          "data": materiais["data_entrega"], "link": materiais["link_download"]}).dropna(
        subset=["ticker"])
    return fatos.sort_values(["data", "ticker"], ascending=[False, True], kind="stable").reset_index(drop=True)


ROTULOS_DOS_TOPICOS = {"election_politics": "Eleição e política", "monetary_policy": "Juros e inflação",
                       "fiscal_policy": "Fiscal", "company_specific": "Empresas", "commodities_global":
                       "Commodities e exterior", "currency": "Câmbio", "personal_finance": "Finanças pessoais",
                       "other": "Outros"}


def manchetes(itens: pd.DataFrame, notas: pd.DataFrame, quantas: int = 50) -> pd.DataFrame:
    """The ``quantas`` newest relevant scored items: ``quando`` (Brasília), ``fonte``, ``titulo``, ``tema`` (the most
    likely topic), ``tom`` and ``link``. Titles and links only, never the summary text (the lake keeps the raw text on
    its own disk, DECISOES 123)."""
    quadro = itens.join(notas, on="key")
    quadro = quadro[quadro["relevante"] >= LIMIAR_RELEVANTE]
    quando = pd.to_datetime(quadro["fetched_at"], utc=True).dt.tz_convert(FUSO).dt.tz_localize(None)
    lista = pd.DataFrame({"quando": quando, "fonte": quadro["source"], "titulo": quadro["title"],
                          "tema": quadro[list(TOPICOS)].idxmax(axis=1).map(ROTULOS_DOS_TOPICOS),
                          "tom": quadro["tom"], "link": quadro["url"]})
    return lista.sort_values("quando", ascending=False, kind="stable").head(quantas).reset_index(drop=True)


COLUNAS_DO_ITEM = ("key", "source", "kind", "fetched_at", "title", "url")


def _linhas(arquivo: Path):
    with arquivo.open(encoding="utf-8") as entrada:
        for linha in entrada:
            try:
                yield json.loads(linha)
            except ValueError:
                continue        # a killed run's partial line (the lake's collector skips it too)


def ler_log(pasta: Path, desde: date) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The lake's log under ``pasta`` (``D:/b3-datalake/sentiment``): the items of the daily files from ``desde`` on
    (``COLUNAS_DO_ITEM``, no summary text) and the Jev scores of every file (``notas_por_item``)."""
    linhas = [linha for arquivo in sorted((pasta / "raw" / "items").glob("*.jsonl"))
              if date.fromisoformat(arquivo.stem) >= desde for linha in _linhas(arquivo)]
    itens = pd.DataFrame([{coluna: linha.get(coluna) for coluna in COLUNAS_DO_ITEM} for linha in linhas],
                         columns=list(COLUNAS_DO_ITEM)).drop_duplicates("key")
    notas = notas_por_item(linha for arquivo in sorted((pasta / "scores").glob("*.jsonl"))
                           for linha in _linhas(arquivo))
    return itens, notas


# The election-window study (scripts/evento_eleicao.py, docs/eleicao-2026-10-23.txt, pre-registered): 10 rounds of
# 2006-2022; the volatility of sessions 0 to +4 after the vote against sessions -60 to -11: x1.43, p = 0.0013.
# Direction was printed, never tested.
TURNOS_DE_2026 = (date(2026, 10, 4), date(2026, 10, 25))
RAZAO_DE_VOL_DO_ESTUDO, P_DO_ESTUDO, RODADAS_DO_ESTUDO = 1.43, 0.0013, 10
SESSOES_DA_JANELA = 5


@dataclass(frozen=True)
class JanelaEleitoral:
    turno: date             # the vote (a Sunday)
    dias_ate: int           # calendar days from today (negative after it)
    janela_inicio: date     # session 0: the first weekday after the vote
    janela_fim: date        # session +4
    dentro: bool            # today is in sessions 0 to +4


def janela_eleitoral(hoje: date) -> JanelaEleitoral | None:
    """The next round of 2026 whose studied window (sessions 0 to +4) has not ended, or None after the last one."""
    for turno in TURNOS_DE_2026:
        sessoes = pd.bdate_range(pd.Timestamp(turno) + pd.Timedelta(days=1), periods=SESSOES_DA_JANELA)
        inicio, fim = sessoes[0].date(), sessoes[-1].date()
        if hoje <= fim:
            return JanelaEleitoral(turno, (turno - hoje).days, inicio, fim, inicio <= hoje <= fim)
    return None
