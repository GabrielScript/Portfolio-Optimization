"""
sentimento.py - The screen's news, sentiment and filings panel, from the lake's forward log (``b3lake.newslog``, Jev
scores of the questions ``items-v1``) and the CVM IPE filings (``b3lake.ipe``). Information only: the user asked to
see it on 2026-10-23 (DECISOES 124); nothing here reaches the optimizer, the paper routines or the Alepha plan,
and the pre-registered tests of ``docs/preregistro-sentimento-2026-10-23.md`` are unchanged.

The Jev questions have no ticker: per-stock coverage comes from the CVM filings, which name the company.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from urllib.parse import urlparse

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


VERSAO_DAS_PERGUNTAS_DE_FATOS = "filings-v1"
COLUNAS_DA_NOTA_DE_FATO = ("tom_do_fato", "p_ruim", "p_bom", "p_forte")


def notas_por_fato(linhas: Iterable[dict]) -> pd.DataFrame:
    """One row per Jev-scored CVM filing (index ``key`` = the filing's ``link_download``): ``tom_do_fato`` = the most
    likely of bad/neutral/good/unclear, ``p_ruim`` = P(bad), ``p_bom`` = P(good), ``p_forte`` = P(intensity Major or
    Extraordinary). Rows of other targets or question versions are skipped."""
    registros = {}
    for linha in linhas:
        if linha.get("target") != "filing" or linha.get("questions_version") != VERSAO_DAS_PERGUNTAS_DE_FATOS:
            continue
        tom = linha["answers"]["filing_tone"]["probabilities"]
        intensidade = linha["answers"]["filing_intensity"]["probabilities"]
        registros[linha["key"]] = {"tom_do_fato": max(tom, key=tom.get), "p_ruim": float(tom["bad"]),
                                   "p_bom": float(tom["good"]), "p_forte": float(intensidade["2"] + intensidade["3"])}
    notas = pd.DataFrame.from_dict(registros, orient="index", columns=list(COLUNAS_DA_NOTA_DE_FATO))
    return notas.astype({"tom_do_fato": object, "p_ruim": float, "p_bom": float, "p_forte": float}).rename_axis("key")


def com_notas_dos_fatos(fatos: pd.DataFrame, notas: pd.DataFrame) -> pd.DataFrame:
    """``fatos`` (``fatos_recentes``) with the columns of ``notas_por_fato`` joined by ``link``; a filing Jev has not
    scored yet (backfilled, or waiting for the next run) keeps its row with NaN scores."""
    return fatos.join(notas, on="link")


LIMIAR_FORTE = 0.5          # P(intensity Major or Extraordinary) for a filing to change a stock's light


def semaforo(fatos: pd.DataFrame, tickers: Iterable[str], desde) -> pd.DataFrame:
    """The light of each of ``tickers`` (index, app format) from its material filings (``com_notas_dos_fatos``)
    delivered on or after ``desde``: ``cor`` "vermelho" for a filing whose most likely tone is bad with P(strong) >=
    ``LIMIAR_FORTE``, with ``motivo`` (dd/mm + subject), ``data`` and ``link`` of the newest such filing. Information
    only: nothing here reaches the weights (DECISOES 132)."""
    # a panel file written before the scores were joined has no score columns: every filing reads as unscored
    fatos = fatos.reindex(columns=list(dict.fromkeys([*fatos.columns, *COLUNAS_DA_NOTA_DE_FATO])))
    luzes = {}
    for ticker in tickers:
        do_ticker = fatos[(fatos["ticker"] == ticker.removesuffix(".SA")) & (fatos["data"] >= desde)]
        fortes = do_ticker[do_ticker["p_forte"] >= LIMIAR_FORTE]
        for tom, cor in (("bad", "vermelho"), ("good", "verde")):
            do_tom = fortes[fortes["tom_do_fato"] == tom]
            if len(do_tom):
                fato = do_tom.sort_values("data", kind="stable").iloc[-1]
                luzes[ticker] = {"cor": cor, "motivo": _rotulo_do_fato(fato), "data": fato["data"],
                                 "link": fato["link"]}
                break
        else:
            pontuados = int(do_ticker["tom_do_fato"].notna().sum())
            luzes[ticker] = {"cor": "amarelo", "data": None, "link": None, "motivo":
                             f"neutro ({pontuados} fato{'s' if pontuados > 1 else ''} pontuado"
                             f"{'s' if pontuados > 1 else ''})" if pontuados else "sem fato pontuado"}
    return pd.DataFrame.from_dict(luzes, orient="index", columns=["cor", "motivo", "data", "link"])


def _rotulo_do_fato(fato: pd.Series) -> str:
    return f"{fato['data']:%d/%m} {fato['assunto'] if isinstance(fato['assunto'], str) else ''}".strip()


LIMIAR_RELEVANTE = 0.5                 # the pre-registration's "relevant >= 0.5" (Test B)
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
                          "data": materiais["data_entrega"],
                          "link": materiais["link_download"].map(link_seguro)}).dropna(subset=["ticker"])
    return fatos.sort_values(["data", "ticker"], ascending=[False, True], kind="stable").reset_index(drop=True)


def ultimo_fato(fatos: pd.DataFrame) -> pd.Series:
    """Ticker -> "dd/mm subject" of its newest filing (``fatos_recentes``), date and subject from the same row; a
    filing without a subject shows only its date."""
    ultimos = fatos.sort_values("data", kind="stable").drop_duplicates("ticker", keep="last").set_index("ticker")
    return pd.Series({ticker: f"{linha['data']:%d/%m} {linha['assunto'] if isinstance(linha['assunto'], str) else ''}"
                     .strip() for ticker, linha in ultimos.iterrows()}, dtype="object")


def link_seguro(url) -> str | None:
    """``url`` when it is an http(s) address, else None: the links come from third-party feeds and files, and a
    ``javascript:`` or ``data:`` URL must never become a clickable link on the screen."""
    try:
        return url if isinstance(url, str) and urlparse(url).scheme in ("http", "https") else None
    except ValueError:
        return None


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
                          "tom": quadro["tom"], "link": quadro["url"].map(link_seguro)})
    return lista.sort_values("quando", ascending=False, kind="stable").head(quantas).reset_index(drop=True)


COLUNAS_DO_ITEM = ("key", "source", "kind", "fetched_at", "title", "url")


def _linhas(arquivo: Path):
    with arquivo.open(encoding="utf-8") as entrada:
        for linha in entrada:
            try:
                yield json.loads(linha)
            except ValueError:
                continue        # a killed run's partial line (the lake's collector skips it too)


def _arquivos_desde(pasta: Path, desde: date) -> list[Path]:
    """The daily ``YYYY-MM-DD.jsonl`` files of ``pasta`` from ``desde`` on; any other file name is skipped."""
    arquivos = []
    for arquivo in sorted(pasta.glob("*.jsonl")):
        try:
            dia = date.fromisoformat(arquivo.stem)
        except ValueError:
            continue
        if dia >= desde:
            arquivos.append(arquivo)
    return arquivos


def ler_log(pasta: Path, desde: date) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The lake's log under ``pasta`` (``D:/b3-datalake/sentiment``): the items of the daily files from ``desde`` on
    (``COLUNAS_DO_ITEM``, no summary text) and their Jev scores (``notas_por_item``). Score files are dated by the
    scoring day, never before the fetch day, so only those from ``desde`` on are read."""
    linhas = [linha for arquivo in _arquivos_desde(pasta / "raw" / "items", desde) for linha in _linhas(arquivo)]
    itens = pd.DataFrame([{coluna: linha.get(coluna) for coluna in COLUNAS_DO_ITEM} for linha in linhas],
                         columns=list(COLUNAS_DO_ITEM)).drop_duplicates("key")
    notas = notas_por_item(linha for arquivo in _arquivos_desde(pasta / "scores", desde)
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


def proxima_troca(calendario: pd.DatetimeIndex, registrada: date) -> date:
    """About when paper-v2 registers the next portfolio: ``core.paper.PASSO`` lake sessions after ``registrada``,
    counted as ``core.paper.rebalanceio_devido`` counts them, the sessions still to come as weekdays (holidays push
    the real date later; paper-v2's own calendar decides)."""
    from core.paper import PASSO
    faltam = max(PASSO - int((calendario > pd.Timestamp(registrada)).sum()), 0)
    return (calendario[-1] + pd.offsets.BDay(faltam)).date()


def alertas_antes_da_troca(luzes: pd.DataFrame, registrada: date, proxima: date) -> list[str]:
    """One line per stock of ``semaforo`` whose red filing came after the portfolio registered on ``registrada``:
    the plan keeps its weights until paper-v2's next registration (about ``proxima``)."""
    vermelhos = luzes[(luzes["cor"] == "vermelho") & (luzes["data"] > registrada)]
    return [f"{ticker.removesuffix('.SA')} ficou vermelho em {linha['data']:%d/%m}, depois da carteira de "
            f"{registrada:%d/%m}; próxima troca ~{proxima:%d/%m/%Y}" for ticker, linha in vermelhos.iterrows()]


def ler_notas_dos_fatos(pasta: Path, desde: date) -> pd.DataFrame:
    """The Jev scores of the CVM filings (``notas_por_fato``) in the lake's score files under ``pasta`` from
    ``desde`` on (the scorer writes items and filings to the same daily files)."""
    return notas_por_fato(linha for arquivo in _arquivos_desde(pasta / "scores", desde) for linha in _linhas(arquivo))
