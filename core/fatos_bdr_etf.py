"""
fatos_bdr_etf.py - CVM material facts of the market panel's BDRs and ETFs (the user's UI list of 2026-11-03), from
official sources only:

- Sponsored BDRs: their foreign issuers file with the CVM like the B3 companies (``ipe.parquet``, CVM codes 80000 and
  up). A BDR maps by the lake's issuer->CVM code table when that code files there, else by its B3 name being the start
  of exactly one foreign issuer's CVM name. Unsponsored BDRs (Apple, Amazon...) file nothing with the CVM.
- ETFs: the CVM's fund eventual documents (dados.cvm.gov.br, ``FI/DOC/EVENTUAL``, type "FATO RELEV"); a ticker maps
  to the one fund CNPJ whose CVM name starts with the fund's full name in the B3 instruments register (BVBG.028,
  ``CrpnNm``).
"""

from __future__ import annotations

import unicodedata
import xml.etree.ElementTree as ET
from collections.abc import Iterable, Mapping
from datetime import date
from pathlib import Path

import pandas as pd

from core.sentimento import link_seguro

CODIGO_ESTRANGEIRO = 80_000          # the CVM's codes of foreign issuers start here
MINIMO_DO_NOME = 4                   # a shorter B3 name is no match
TIPO_FATO_DE_FUNDO = "FATO RELEV"
COLUNAS = ["ticker", "empresa", "categoria", "assunto", "data", "link"]


def normalizar(nome: str) -> str:
    """Upper case, no accents, letters and digits only."""
    sem_acento = unicodedata.normalize("NFKD", str(nome)).encode("ascii", "ignore").decode("ascii")
    return "".join(caractere for caractere in sem_acento.upper() if caractere.isalnum())


def _unico_que_comeca(prefixo: str, nomes: Mapping[object, str]) -> object | None:
    """The one key whose normalized name starts with ``prefixo``; None for none or several."""
    if len(prefixo) < MINIMO_DO_NOME:
        return None
    achados = {chave for chave, nome in nomes.items() if nome.startswith(prefixo)}
    return achados.pop() if len(achados) == 1 else None


def codigos_dos_bdrs(bdrs: pd.DataFrame, setores: pd.DataFrame, ipe: pd.DataFrame) -> dict[int, str]:
    """CVM code -> BDR ticker for the BDRs (``security``, ``name``) whose issuer files with the CVM, one ticker per
    issuer (the first in ``bdrs``)."""
    no_ipe = set(pd.to_numeric(ipe["codigo_cvm"], errors="coerce").dropna().astype(int))
    estrangeiros = (ipe.assign(codigo=pd.to_numeric(ipe["codigo_cvm"], errors="coerce"))
                    .dropna(subset=["codigo"]).query("codigo >= @CODIGO_ESTRANGEIRO")
                    .drop_duplicates("codigo").set_index("codigo")["nome_companhia"].map(normalizar).to_dict())
    do_lago = setores.dropna(subset=["code_cvm"]).drop_duplicates("issuer").set_index("issuer")["code_cvm"]
    codigos: dict[int, str] = {}
    for linha in bdrs.itertuples(index=False):
        codigo = do_lago.get(linha.security[:4])
        codigo = int(codigo) if codigo is not None and int(codigo) in no_ipe else _unico_que_comeca(
            normalizar(linha.name), estrangeiros)
        if codigo is not None:
            codigos.setdefault(int(codigo), linha.security)
    return codigos


def nomes_completos_do_cadastro(arquivo: Path, tickers: Iterable[str]) -> dict[str, str]:
    """Ticker -> full name (``CrpnNm``) of ``tickers`` in a B3 instruments register (BVBG.028 XML), read as a
    stream: the file has hundreds of megabytes."""
    procurados, nomes = set(tickers), {}
    nome = None
    for _, elemento in ET.iterparse(arquivo, events=("end",)):
        etiqueta = elemento.tag.rsplit("}", 1)[-1]
        if etiqueta == "CrpnNm":
            nome = (elemento.text or "").strip()
        elif etiqueta == "TckrSymb":
            ticker = (elemento.text or "").strip()
            if ticker in procurados and nome:
                nomes.setdefault(ticker, nome)
        elif etiqueta == "Instrm":
            nome = None
            elemento.clear()
    return nomes


def cnpjs_dos_etfs(nomes: Mapping[str, str], eventuais: pd.DataFrame) -> dict[str, str]:
    """Fund CNPJ -> ETF ticker: the single CNPJ of ``eventuais`` whose normalized CVM name starts with the ETF's
    normalized full name."""
    da_cvm = (eventuais.drop_duplicates("CNPJ_FUNDO_CLASSE").set_index("CNPJ_FUNDO_CLASSE")["DENOM_SOCIAL"]
              .map(normalizar).to_dict())
    cnpjs: dict[str, str] = {}
    for ticker, nome in nomes.items():
        cnpj = _unico_que_comeca(normalizar(nome), da_cvm)
        if cnpj is not None:
            cnpjs.setdefault(str(cnpj), ticker)
    return cnpjs


def fatos_dos_fundos(eventuais: pd.DataFrame, cnpjs: Mapping[str, str], desde: date) -> pd.DataFrame:
    """The ETFs' "FATO RELEV" documents received on or after ``desde``, in ``core.sentimento.fatos_recentes``'s
    columns (the CVM gives no subject: "Fato relevante do fundo"), newest first."""
    recebido = pd.to_datetime(eventuais["DT_RECEB"], errors="coerce")
    dentro = eventuais[(eventuais["TP_DOC"] == TIPO_FATO_DE_FUNDO) & (recebido >= pd.Timestamp(desde))
                       & eventuais["CNPJ_FUNDO_CLASSE"].isin(list(cnpjs))]
    fatos = pd.DataFrame({"ticker": dentro["CNPJ_FUNDO_CLASSE"].map(cnpjs), "empresa": dentro["DENOM_SOCIAL"],
                          "categoria": "Fato Relevante", "assunto": "Fato relevante do fundo",
                          "data": recebido[dentro.index], "link": dentro["LINK_ARQ"].map(link_seguro)}, columns=COLUNAS)
    return fatos.sort_values(["data", "ticker"], ascending=[False, True], kind="stable").reset_index(drop=True)
