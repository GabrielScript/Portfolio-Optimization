"""Daily CVM filings (RAD/ENET), so Fato Relevante and Comunicado ao Mercado reach the Sentinela the day they are
filed instead of with the weekly open data's delay of up to 6 days (spec, "Build order" item 3).

Source: CVM's public document search (``ENETWeb/frmConsultaExternaCVM.aspx``; the old ``ENET`` page was
discontinued on 06/07/2026). Its ``ListarDocumentos`` method answers JSON with the rows in one string (rows split by
``&*``, fields by ``$&``); the page's captcha flag was off (``hdnHabilitaCaptcha = N``) when checked on 05/11/2026.
If CVM turns the captcha on, the request fails and the run falls back to the weekly table (reported, never guessed).
Each filing's link is written exactly as in the weekly open data, so a filing seen in both keeps one item id.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from datetime import date

import pandas as pd
import requests

URL_PAGINA = "https://www.rad.cvm.gov.br/ENETWeb/frmConsultaExternaCVM.aspx"
URL_LISTA = URL_PAGINA + "/ListarDocumentos"
URL_DOCUMENTO = ("https://www.rad.cvm.gov.br/ENET/frmDownloadDocumento.aspx?Tela=ext&descTipo=IPE&"
                 "CodigoInstituicao=1&numProtocolo={protocolo}&numSequencia={sequencia}&numVersao={versao}")
CATEGORIAS = ("IPE_4_-1_-1", "IPE_6_-1_-1")          # Fato Relevante, Comunicado ao Mercado (all types and kinds)
PRAZO_S = 90
AGENTE = {"User-Agent": "Mozilla/5.0 (Aleph Quant Sentinela)"}
_TAG = re.compile(r"<[^>]+>")
_DOWNLOAD = re.compile(r"OpenDownloadDocumentos\('(\d+)','(\d+)','(\d+)','IPE'\)")
_ENTREGA = re.compile(r"(\d{2}/\d{2}/\d{4})(?:\s+(\d{2}:\d{2}))?")


class ErroDaCvm(Exception):
    """The search failed (session, error flag, captcha); the message is safe to show."""


def _limpo(texto: str) -> str:
    return re.sub(r"\s+", " ", _TAG.sub("", texto)).strip()


def ler_lista(dados: str) -> pd.DataFrame:
    """One row per filing: codigo_cvm, empresa, categoria, assunto, entregue_em, link."""
    linhas = []
    for bruta in dados.split("&*"):
        campos = bruta.split("$&")
        if len(campos) < 12:
            continue
        download = _DOWNLOAD.search(campos[10])
        entrega = _ENTREGA.search(_limpo(campos[6]))
        if not download or not entrega:
            continue
        sequencia, versao, protocolo = download.groups()
        quando = pd.to_datetime(f"{entrega.group(1)} {entrega.group(2) or '00:00'}", format="%d/%m/%Y %H:%M")
        assunto = _limpo(campos[11]) or _limpo(campos[4]).removesuffix("-").strip()
        linhas.append({"codigo_cvm": int(campos[0].replace("-", "")), "empresa": _limpo(campos[1]),
                       "categoria": _limpo(campos[2]), "assunto": assunto, "entregue_em": quando,
                       "link": URL_DOCUMENTO.format(protocolo=protocolo, sequencia=sequencia, versao=versao)})
    return pd.DataFrame(linhas, columns=["codigo_cvm", "empresa", "categoria", "assunto", "entregue_em", "link"])


def codigos(setores: pd.DataFrame, tickers: Iterable[str]) -> dict[int, list[str]]:
    """CVM code -> the watched tickers of that issuer (the lake's ``sectors``: issuer, code_cvm)."""
    por_emissor = dict(zip(setores["issuer"], setores["code_cvm"]))
    saida: dict[int, list[str]] = {}
    for ticker in tickers:
        codigo = por_emissor.get(ticker[:4])
        if codigo is not None and pd.notna(codigo):
            saida.setdefault(int(codigo), []).append(ticker)
    return saida


def fatos_dos_nomes(lista: pd.DataFrame, codigos_dos_nomes: dict[int, list[str]]) -> pd.DataFrame:
    """The filings of the watched issuers in the weekly table's columns (one row per ticker; ``data`` = delivery day)."""
    linhas = []
    for _, fato in lista[lista["codigo_cvm"].isin(codigos_dos_nomes)].iterrows():
        for ticker in codigos_dos_nomes[int(fato["codigo_cvm"])]:
            linhas.append({"ticker": ticker, "empresa": fato["empresa"], "categoria": fato["categoria"],
                           "assunto": fato["assunto"], "data": fato["entregue_em"].normalize(), "link": fato["link"],
                           "entregue_em": fato["entregue_em"]})
    return pd.DataFrame(linhas, columns=["ticker", "empresa", "categoria", "assunto", "data", "link", "entregue_em"])


def corpo(de: date, ate: date) -> str:
    """The request body, written as the page's own script writes it."""
    return (f"{{ dataDe: '{de:%d/%m/%Y}', dataAte: '{ate:%d/%m/%Y}' , empresa: '', setorAtividade: '-1', "
            f"categoriaEmissor: '-1', situacaoEmissor: '-1', tipoParticipante: '-1', dataReferencia: '', "
            f"categoria: '{','.join(CATEGORIAS)}', periodo: '2', horaIni: '', horaFim: '', palavraChave:'',"
            f"ultimaDtRef:'false', tipoEmpresa:'0', token: '', versaoCaptcha: ''}}")


def consultar(sessao: requests.Session, de: date, ate: date) -> pd.DataFrame:
    """Every Fato Relevante and Comunicado ao Mercado delivered from ``de`` to ``ate`` (one page visit for the
    session cookie, then one request)."""
    try:
        sessao.get(URL_PAGINA, timeout=PRAZO_S, headers=AGENTE)
        resposta = sessao.post(URL_LISTA, data=corpo(de, ate).encode("utf-8"), timeout=PRAZO_S,
                               headers=AGENTE | {"Content-Type": "application/json; charset=utf-8",
                                                 "X-Requested-With": "XMLHttpRequest", "Referer": URL_PAGINA})
        retorno = (resposta.json() or {}).get("d") or {}
    except (requests.RequestException, ValueError) as erro:
        raise ErroDaCvm(f"CVM RAD indisponível ({type(erro).__name__})") from erro
    if retorno.get("expirouSessao") or retorno.get("temErro") or "dados" not in retorno:
        raise ErroDaCvm(f"CVM RAD recusou a consulta: {retorno.get('msgErro') or 'sessão expirada'}")
    return ler_lista(retorno["dados"] or "")
