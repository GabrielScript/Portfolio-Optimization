"""
dolar_bcb.py - The dollar of the market panel: the day's last PTAX bulletin from the BCB's Olinda service (opening at
~10h, intermediate at ~11h and ~12h, the PTAX close at ~13h). The SGS series 1 (``core.bcb.baixar_ptax``) only carries
the close and is a day behind until 13h, which the user saw as an outdated dollar (2026-11-05). Before the first
bulletin, and on weekends and holidays, the last day that has one stands in. Kept out of ``core/bcb.py`` (a watched
file).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import date, timedelta

import pandas as pd
import requests

URL_DO_DIA = ("https://olinda.bcb.gov.br/olinda/servico/PTAX/versao/v1/odata/"
              "CotacaoMoedaDia(moeda=@moeda,dataCotacao=@dataCotacao)?@moeda='USD'&@dataCotacao='{dia:%m-%d-%Y}'"
              "&$format=json")
DIAS_PARA_TRAS = 7             # covers a long holiday weekend
PRAZO_S = 20
CABECALHO = {"User-Agent": "Mozilla/5.0 (Aleph Quant)"}      # the service answers 403 to a bare client


@dataclass(frozen=True)
class DolarDoDia:
    valor: float               # R$ per US$, the bulletin's selling rate
    quando: pd.Timestamp       # the bulletin's time (Brasília)
    boletim: str               # "Abertura", "Intermediário" or "Fechamento PTAX"


def url_do_dia(dia: date) -> str:
    return URL_DO_DIA.format(dia=dia)


def _baixar(dia: date) -> dict:
    resposta = requests.get(url_do_dia(dia), headers=CABECALHO, timeout=PRAZO_S)
    resposta.raise_for_status()
    return resposta.json()


def dolar_do_dia(hoje: date, baixar: Callable[[date], dict] | None = None) -> DolarDoDia | None:
    """The last bulletin of ``hoje``, or of the last day before it (up to ``DIAS_PARA_TRAS``) that has one; None when
    none has. Network errors propagate (the caller shows a dash)."""
    baixar = baixar or _baixar
    for atras in range(DIAS_PARA_TRAS + 1):
        boletins = baixar(hoje - timedelta(days=atras)).get("value") or []
        if boletins:
            ultimo = max(boletins, key=lambda boletim: boletim["dataHoraCotacao"])
            return DolarDoDia(float(ultimo["cotacaoVenda"]), pd.Timestamp(ultimo["dataHoraCotacao"]),
                              str(ultimo["tipoBoletim"]))
    return None
