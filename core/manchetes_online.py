"""
manchetes_online.py - The headlines on the deployed app (option (b) of DECISOES 126, the user's pick of 2026-10-24).
The app's repository is public and keeps no headline text (DECISOES 123), so the daily upkeep publishes the newest
scored headlines (``core.sentimento.manchetes``: title, link, topic and tone; never the summary text) as one JSON file
to a private GitHub repository, and the deployed app downloads it with a read-only token kept in Streamlit secrets.
Information only, like the rest of the market panel: nothing here reaches the optimizer or the Alepha plan.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from datetime import datetime

import pandas as pd
import requests

from core.sentimento import link_seguro

QUANTAS_MANCHETES = 50
# the private store (a repository of the app owner's account; the names are not secret, the token is)
DONO, REPOSITORIO, ARQUIVO = "GabrielScript", "alepha-manchetes", "manchetes.json"
URL_DO_ARQUIVO = f"https://api.github.com/repos/{DONO}/{REPOSITORIO}/contents/{ARQUIVO}"
PRAZO_S = 10
COLUNAS = ("quando", "fonte", "titulo", "tema", "tom", "link")


def para_json(manchetes: pd.DataFrame, gerado_em: datetime) -> str:
    """The file the upkeep publishes: the publish time and the rows of ``manchetes`` (``COLUNAS``), times in ISO."""
    linhas = manchetes.loc[:, list(COLUNAS)].assign(quando=manchetes["quando"].dt.strftime("%Y-%m-%dT%H:%M:%S"))
    registros = json.loads(linhas.to_json(orient="records", force_ascii=False))
    return json.dumps({"gerado_em": gerado_em.isoformat(timespec="seconds"), "manchetes": registros},
                      ensure_ascii=False, indent=1)


def de_json(texto: str) -> tuple[pd.DataFrame, datetime]:
    """The headlines (``COLUNAS``; ``quando`` as datetime, ``tom`` as float) and the publish time of a file made by
    ``para_json``; links that are not http(s) become empty (the file is data from outside the app)."""
    dados = json.loads(texto)
    manchetes = pd.DataFrame(dados["manchetes"], columns=list(COLUNAS))
    manchetes["quando"] = pd.to_datetime(manchetes["quando"])
    manchetes["tom"] = manchetes["tom"].astype(float)
    manchetes["link"] = manchetes["link"].map(link_seguro)
    return manchetes, datetime.fromisoformat(dados["gerado_em"])


def baixar(token: str, get: Callable[..., requests.Response] = requests.get) -> str | None:
    """The published file from the private repository (GitHub contents API, raw), with ``token`` (fine-grained,
    Contents read-only on that repository) in the ``Authorization`` header only; None on any failure (no network, bad
    or expired token, file missing): the page never depends on GitHub being up, and no message carries the token."""
    try:
        resposta = get(URL_DO_ARQUIVO, headers={"Authorization": f"Bearer {token}",
                                                "Accept": "application/vnd.github.raw+json",
                                                "X-GitHub-Api-Version": "2022-11-28"}, timeout=PRAZO_S)
    except requests.RequestException:
        return None
    return resposta.content.decode("utf-8") if resposta.status_code == 200 else None
