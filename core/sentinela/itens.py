"""Items: what each official source says, as short verbatim texts the model may cite (spec, "Sources").

Each builder is pure (DataFrames in, items out); the fetching lives in ``core.sentinela.fontes``. An item's id is
stable for the same fact, so the seen store recognizes it on the next hourly run.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Mapping
from dataclasses import dataclass

import pandas as pd

MACRO = "macro"                    # the ticker of an item shared by every name (BCB, IBGE, Tesouro)
CARACTERES = 600                   # an item's text at most
CARACTERES_DO_CORPO = 3000         # a Fato Relevante's document text after its subject (amendment 7)
ROTULO_DO_CORPO = " Texto do documento: "
SIGMAS_DO_PRECO = 2.5              # a close move beyond this many daily sigmas of the last 63 sessions
SESSOES_DO_DESVIO = 63
VEZES_DA_MEDIANA = 3.0             # volume, shares lent or lending rate beyond 3x the last 21 sessions' median
SESSOES_DA_MEDIANA = 21


@dataclass(frozen=True)
class Item:
    """One fact: ``fonte`` is its official class (cvm, mercado, bcb, ibge, tesouro); ``data`` is YYYY-MM-DD."""
    id: str
    fonte: str
    data: str
    ticker: str
    texto: str

    def linha(self) -> str:
        return f"{self.id} | {self.fonte} | {self.data} | {self.texto}"


def _id(fonte: str, *partes: object) -> str:
    return f"{fonte}-" + hashlib.sha1("|".join(map(str, partes)).encode("utf-8")).hexdigest()[:12]


def _br(valor: float, casas: int) -> str:
    """Brazilian writing: 1.234.567,89."""
    return f"{valor:,.{casas}f}".replace(",", "_").replace(".", ",").replace("_", ".")


def _publicado(valor: float) -> str:
    """A released value as published, decimal comma and no padding: 15 -> "15", 0.82 -> "0,82"."""
    return f"{valor:.10g}".replace(".", ",")


def itens_cvm(fatos: pd.DataFrame, tickers: Iterable[str]) -> list[Item]:
    """One item per CVM filing (``fatos``: ticker, empresa, categoria, assunto, data, link, optionally corpo, the
    document's text) of ``tickers``."""
    nomes = set(tickers)
    itens = []
    for _, fato in fatos[fatos["ticker"].isin(nomes)].iterrows():
        data = pd.Timestamp(fato["data"])
        assunto = fato["assunto"] if isinstance(fato["assunto"], str) and fato["assunto"].strip() else "sem assunto"
        texto = f"{fato['empresa']} ({fato['ticker']}), {fato['categoria']} de {data:%d/%m/%Y}: {assunto.strip()}"
        chave = fato["link"] if isinstance(fato.get("link"), str) and fato["link"] else (fato["ticker"], data, assunto)
        texto = texto[:CARACTERES]
        corpo = fato.get("corpo")
        if isinstance(corpo, str) and corpo.strip():
            texto += ROTULO_DO_CORPO + corpo.strip()[:CARACTERES_DO_CORPO]
        itens.append(Item(_id("cvm", chave), "cvm", f"{data:%Y-%m-%d}", fato["ticker"], texto))
    return itens


def _vezes_a_mediana(serie: pd.Series, posicao: int) -> float | None:
    """The value at ``posicao`` over the median of the 21 values before it; None without 21 values or a median."""
    anteriores = serie.iloc[max(0, posicao - SESSOES_DA_MEDIANA):posicao].dropna()
    if len(anteriores) < SESSOES_DA_MEDIANA or not anteriores.median() > 0 or pd.isna(serie.iloc[posicao]):
        return None
    return float(serie.iloc[posicao] / anteriores.median())


def _ultimas(indice: pd.Index, sessoes: int) -> range:
    return range(max(0, len(indice) - sessoes), len(indice))


def itens_mercado(fechamentos: pd.DataFrame, volumes: pd.DataFrame, sessoes: int = 5) -> list[Item]:
    """Price and volume anomalies of the last ``sessoes`` sessions (dates x tickers: total-return closes, R$ traded)."""
    itens = []
    for ticker in fechamentos.columns:
        retornos = fechamentos[ticker].pct_change(fill_method=None)
        for posicao in _ultimas(retornos.index, sessoes):
            dia = retornos.index[posicao]
            anteriores = retornos.iloc[max(1, posicao - SESSOES_DO_DESVIO):posicao].dropna()
            retorno = retornos.iloc[posicao]
            if len(anteriores) >= SESSOES_DO_DESVIO and pd.notna(retorno):
                desvio = float(anteriores.std())
                if abs(retorno) > SIGMAS_DO_PRECO * desvio:
                    texto = (f"{ticker}: variação de {_br(100 * retorno, 2)}% no pregão de {dia:%d/%m/%Y}; o desvio "
                             f"diário dos {SESSOES_DO_DESVIO} pregões anteriores é {_br(100 * desvio, 2)}%, limite de "
                             f"{_br(SIGMAS_DO_PRECO, 1)} desvios")
                    itens.append(Item(_id("mercado", ticker, dia, "preco"), "mercado", f"{dia:%Y-%m-%d}", ticker, texto))
        if ticker in volumes.columns:
            volume = volumes[ticker]
            for posicao in _ultimas(volume.index, sessoes):
                vezes = _vezes_a_mediana(volume, posicao)
                if vezes is not None and vezes > VEZES_DA_MEDIANA:
                    dia = volume.index[posicao]
                    mediana = volume.iloc[posicao] / vezes
                    texto = (f"{ticker}: volume de R$ {_br(volume.iloc[posicao], 0)} no pregão de {dia:%d/%m/%Y}, "
                             f"{_br(vezes, 1)} vezes a mediana dos {SESSOES_DA_MEDIANA} pregões anteriores "
                             f"(R$ {_br(mediana, 0)})")
                    itens.append(Item(_id("mercado", ticker, dia, "volume"), "mercado", f"{dia:%Y-%m-%d}", ticker, texto))
    return itens


def itens_aluguel(saldos: pd.DataFrame, taxas: pd.DataFrame, sessoes: int = 5) -> list[Item]:
    """B3 BDI lending jumps (dates x tickers): shares lent out and the borrower's average rate. The rate's unit is not
    published (read as a fraction a year, DECISOES 264), so only its relative jump is used."""
    itens = []
    for quadro, nome, como in ((saldos, "saldo", "saldo de ações emprestadas de {valor}"),
                               (taxas, "taxa", "taxa média do tomador no aluguel de {valor}")):
        for ticker in quadro.columns:
            serie = quadro[ticker]
            for posicao in _ultimas(serie.index, sessoes):
                vezes = _vezes_a_mediana(serie, posicao)
                if vezes is not None and vezes > VEZES_DA_MEDIANA:
                    dia, valor = serie.index[posicao], float(serie.iloc[posicao])
                    escrito = _br(valor, 0) if nome == "saldo" else _publicado(valor)
                    mediana = valor / vezes
                    texto = (f"{ticker}: {como.format(valor=escrito)} em {dia:%d/%m/%Y} (BDI da B3), {_br(vezes, 1)} "
                             f"vezes a mediana dos {SESSOES_DA_MEDIANA} pregões anteriores "
                             f"({_br(mediana, 0) if nome == 'saldo' else _publicado(round(mediana, 8))})")
                    itens.append(Item(_id("mercado", ticker, dia, nome), "mercado", f"{dia:%Y-%m-%d}", ticker, texto))
    return itens


@dataclass(frozen=True)
class Serie:
    """A macro series: ``valores`` dated by reference period; ``mensal`` writes the period as MM/YYYY; ``nomes`` (date ->
    the issuer's own period name, e.g. "2º trimestre 2026") overrides that writing."""
    nome: str
    fonte: str
    rotulo: str
    valores: pd.Series
    mensal: bool = True
    nomes: Mapping[pd.Timestamp, str] | None = None

    def periodo(self, dia: pd.Timestamp) -> str:
        if self.nomes and dia in self.nomes:
            return self.nomes[dia]
        return f"{dia:%m/%Y}" if self.mensal else f"{dia:%d/%m/%Y}"


def itens_macro(series: Iterable[Serie]) -> list[Item]:
    """The latest release of each series, with the one before it; the id changes only with a new reference period."""
    itens = []
    for serie in series:
        valores = serie.valores.dropna().sort_index()
        if valores.empty:
            continue
        dia, valor = valores.index[-1], float(valores.iloc[-1])
        texto = f"{serie.rotulo}: {_publicado(valor)} em {serie.periodo(dia)}"
        if len(valores) > 1:
            texto += f" (anterior: {_publicado(float(valores.iloc[-2]))} em {serie.periodo(valores.index[-2])})"
        itens.append(Item(_id(serie.fonte, serie.nome, dia), serie.fonte, f"{dia:%Y-%m-%d}", MACRO, texto[:CARACTERES]))
    return itens
