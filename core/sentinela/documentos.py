"""A new Fato Relevante's document text for the stage-1 scan (spec amendment 7): each one not yet in the seen store is
downloaded once from its RAD link and its PDF read with ``pypdf``; ``core.sentinela.itens.itens_cvm`` appends the text
after the subject. A failed download or an unreadable PDF keeps the subject line and is noted in the run's failures."""

from __future__ import annotations

import io
import re
from collections.abc import Callable

import pandas as pd
import pypdf
import requests

from core.sentinela.itens import itens_cvm

CATEGORIA = "Fato Relevante"
PRAZO_S = 30
TENTATIVAS = 3
AGENTE = {"User-Agent": "Mozilla/5.0 (Aleph Quant Sentinela)"}
_ESPACOS = re.compile(r"\s+")


def texto_do_pdf(conteudo: bytes) -> str:
    """Every page's text, whitespace collapsed to single spaces (the citation check normalizes the same way)."""
    leitor = pypdf.PdfReader(io.BytesIO(conteudo))
    return _ESPACOS.sub(" ", " ".join(pagina.extract_text() or "" for pagina in leitor.pages)).strip()


def _pdf(link: str, get: Callable) -> bytes:
    """The PDF's bytes; the RAD sometimes answers an HTML page first and the PDF on the next request."""
    for _ in range(TENTATIVAS):
        resposta = get(link, timeout=PRAZO_S, headers=AGENTE)
        if not resposta.ok:
            raise ValueError(f"HTTP {resposta.status_code}")
        if resposta.content.startswith(b"%PDF"):
            return resposta.content
    raise ValueError("a resposta não é um PDF")


def _corpo(link: str, get: Callable) -> str:
    conteudo = _pdf(link, get)
    try:
        texto = texto_do_pdf(conteudo)
    except Exception as erro:       # noqa: BLE001 - a malformed PDF raises many types; the filing keeps its subject
        raise ValueError(f"PDF ilegível: {type(erro).__name__}") from erro
    if not texto:
        raise ValueError("PDF sem texto")
    return texto


def com_corpos(fatos: pd.DataFrame, vistos: set[str], get: Callable = requests.get) -> tuple[pd.DataFrame, list[str]]:
    """``fatos`` with a ``corpo`` column: the document text of each Fato Relevante whose item id is not in ``vistos``
    (any name's seen ids), None for the rest; and the failures, one line per filing."""
    fatos = fatos.copy()
    fatos["corpo"] = None
    falhas = []
    for indice, fato in fatos[fatos["categoria"] == CATEGORIA].iterrows():
        link = fato.get("link")
        if not isinstance(link, str) or not link or itens_cvm(fatos.loc[[indice]], [fato["ticker"]])[0].id in vistos:
            continue
        try:
            fatos.at[indice, "corpo"] = _corpo(link, get)
        except (requests.RequestException, ValueError) as erro:
            falhas.append(f"CVM documento {fato['ticker']} {pd.Timestamp(fato['data']):%d/%m}: {type(erro).__name__}"
                          f"{f' ({erro})' if isinstance(erro, ValueError) else ''}")
    return fatos, falhas
