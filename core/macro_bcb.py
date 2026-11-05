"""
macro_bcb.py - The Banco Central block of the market panel (the user's request of 2026-11-04): economic activity
(IBC-Br), foreign direct investment (IDP), credit volume, its cost and its defaults from the BCB's SGS, and the last
Copom minutes from the BCB site. The adviser's macro pack reads them too (``core.assessor.macro_do_painel``).

Series codes checked against the BCB open data catalog on 2026-11-04 (dadosabertos.bcb.gov.br): 24363 IBC-Br, 22885
IDP monthly net, 20631 credit granted (total), 20714 average credit rate (total), 21082 credit defaults (total). The
weekly exchange flow has no SGS code the catalog confirms, so it is not here. Kept out of ``core/bcb.py`` (a watched
file): this module only calls ``bcb.baixar_serie``.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date, timedelta

import pandas as pd
import requests

from core import bcb

SERIE_IBC_BR = 24363            # index, monthly (2002 = 100), not seasonally adjusted: compared with 12 months before
SERIE_IDP = 22885               # US$ million, monthly, net
SERIE_CONCESSOES = 20631        # R$ million, monthly
SERIE_JUROS_DO_CREDITO = 20714  # % a year
SERIE_INADIMPLENCIA = 21082     # % of the credit portfolio over 90 days late

IBC_BR = "IBC-Br: atividade em 12 meses (%)"
IDP = "Investimento direto no país em 12 meses (US$ bi)"
CONCESSOES = "Crédito concedido no mês (R$ bi)"
CONCESSOES_12M = "Crédito concedido: variação em 12 meses (%)"
JUROS_DO_CREDITO = "Juros médios do crédito (% a.a.)"
INADIMPLENCIA = "Inadimplência do crédito (%)"
ATA_DO_COPOM = "Ata do Copom"

EXPLICACOES = {
    IBC_BR: "Índice de Atividade Econômica do Banco Central, uma prévia mensal do PIB. Aqui, a variação contra o mesmo "
            "mês do ano anterior: acima de 0, a economia cresce.",
    IDP: "Dinheiro que empresas estrangeiras investem em empresas no Brasil (fábricas, participações), soma dos últimos "
         "12 meses. Alto e estável é sinal de confiança de longo prazo no país.",
    CONCESSOES: "Total de crédito novo que os bancos emprestaram no último mês, para empresas e pessoas. Crédito "
                "crescendo costuma acompanhar consumo e investimento em alta.",
    CONCESSOES_12M: "Crédito concedido no último mês contra o mesmo mês do ano anterior.",
    JUROS_DO_CREDITO: "Taxa média que os bancos cobram nos empréstimos (custo do crédito). Sobe com a Selic e com o risco "
                      "de calote; juros altos freiam consumo e investimento e pesam no lucro das empresas endividadas.",
    INADIMPLENCIA: "Parte do crédito com atraso de mais de 90 dias. Em alta, sinal de famílias e empresas apertadas e "
                   "de bancos mais cautelosos.",
    ATA_DO_COPOM: "Documento em que o Comitê de Política Monetária explica a última decisão sobre a Selic e o que pode "
                  "fazer nas próximas reuniões. Mexe com juros futuros, câmbio e bolsa.",
}

URL_ATAS = "https://www.bcb.gov.br/api/servico/sitebcb/copomminutes/ultimas?quantidade=1&filtro="
SITE_DO_BCB = "https://www.bcb.gov.br"
PRAZO_S = 20


@dataclass(frozen=True)
class Ata:
    data: pd.Timestamp
    titulo: str
    link: str


def _ultimo(serie: pd.Series) -> tuple[pd.Timestamp, float]:
    serie = serie.dropna()
    return pd.Timestamp(serie.index[-1]), float(serie.iloc[-1])


def _em_12_meses(serie: pd.Series) -> tuple[pd.Timestamp, float]:
    """The last month against the same month a year before, in %."""
    serie = serie.dropna()
    fim = pd.Timestamp(serie.index[-1])
    antes = serie.get(fim - pd.DateOffset(years=1))
    return fim, float("nan") if antes is None or antes == 0 else 100 * (float(serie.iloc[-1]) / float(antes) - 1)


def indicadores(baixar: Callable[[int, date, date], pd.Series], hoje: date) -> dict[str, tuple[pd.Timestamp, float]]:
    """Name -> (date, value) of each indicator; one whose series fails (``bcb.ErroBcb``) is left out."""
    inicio = hoje - timedelta(days=800)
    saida: dict[str, tuple[pd.Timestamp, float]] = {}

    def serie(codigo: int) -> pd.Series | None:
        try:
            return baixar(codigo, inicio, hoje)
        except bcb.ErroBcb:
            return None

    if (ibc := serie(SERIE_IBC_BR)) is not None:
        saida[IBC_BR] = _em_12_meses(ibc)
    if (idp := serie(SERIE_IDP)) is not None:
        idp = idp.dropna()
        saida[IDP] = (pd.Timestamp(idp.index[-1]), float(idp.iloc[-12:].sum()) / 1000)
    if (concessoes := serie(SERIE_CONCESSOES)) is not None:
        data, valor = _ultimo(concessoes)
        saida[CONCESSOES] = (data, valor / 1000)
        saida[CONCESSOES_12M] = _em_12_meses(concessoes)
    if (juros := serie(SERIE_JUROS_DO_CREDITO)) is not None:
        saida[JUROS_DO_CREDITO] = _ultimo(juros)
    if (inadimplencia := serie(SERIE_INADIMPLENCIA)) is not None:
        saida[INADIMPLENCIA] = _ultimo(inadimplencia)
    return saida


def _buscar_json(url: str) -> dict:
    resposta = requests.get(url, timeout=PRAZO_S)
    resposta.raise_for_status()
    return resposta.json()


def ata_do_copom(buscar: Callable[[str], dict] = _buscar_json) -> Ata | None:
    """The last Copom minutes on the BCB site (date, a Portuguese title, the PDF link); None without one."""
    conteudo = buscar(URL_ATAS).get("conteudo") or []
    if not conteudo:
        return None
    ultima = conteudo[0]
    numero = re.match(r"\s*(\d+)", ultima.get("Titulo", ""))
    titulo = f"Ata da {numero.group(1)}ª reunião do Copom" if numero else ultima.get("Titulo", "Ata do Copom")
    data = pd.Timestamp(ultima["DataReferencia"]).tz_localize(None).normalize()
    return Ata(data, titulo, SITE_DO_BCB + ultima["Url"])
