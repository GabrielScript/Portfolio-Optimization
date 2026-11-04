"""
fatos_bdr_etf.py - CVM material facts of the market panel's BDRs and ETFs (the user's UI list of 2026-11-03), from
official sources only:

- Sponsored BDRs: their foreign issuers file with the CVM like the B3 companies (``ipe.parquet``, CVM codes 80000 and
  up). A BDR maps by the lake's issuer->CVM code table when that code files there, else by its B3 name being the start
  of exactly one foreign issuer's CVM name. Unsponsored BDRs (Apple, Amazon...) file nothing with the CVM.
- ETFs (and the other B3-listed funds the panel lists as ETFs): Fundos.NET, B3's system where listed funds deliver
  their CVM filings, category "Fato Relevante", matched by the B3 trading name (``nomePregao``) the app already has.
  The CVM's open fund eventual-documents dataset was tried first and has no index ETF (2026-11-04).
"""

from __future__ import annotations

import unicodedata
from collections.abc import Mapping
from datetime import date

import pandas as pd

CODIGO_ESTRANGEIRO = 80_000          # the CVM's codes of foreign issuers start here
MINIMO_DO_NOME = 4                   # a shorter B3 name is no match
LINK_DO_FNET = "https://fnet.bmfbovespa.com.br/fnet/publico/exibirDocumento?id={id}"
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


def fatos_do_fnet(registros: pd.DataFrame, etfs: pd.DataFrame, desde: date) -> pd.DataFrame:
    """The listed funds' material facts (``registros``: Fundos.NET's "Fato Relevante" search rows) of the ETFs in
    ``etfs`` (``security``, ``name``: the B3 trading name), delivered on or after ``desde`` and not cancelled, in
    ``core.sentimento.fatos_recentes``'s columns (Fundos.NET gives no subject), newest first."""
    por_nome = dict(zip(etfs["name"].str.strip(), etfs["security"]))
    entrega = pd.to_datetime(registros["dataEntrega"], format="%d/%m/%Y %H:%M", errors="coerce")
    ticker = registros["nomePregao"].fillna("").str.strip().map(por_nome)
    dentro = registros[ticker.notna() & (entrega >= pd.Timestamp(desde)) & (registros["situacaoDocumento"] != "C")]
    fatos = pd.DataFrame({"ticker": ticker[dentro.index], "empresa": dentro["descricaoFundo"],
                          "categoria": dentro["categoriaDocumento"], "assunto": "Fato relevante do fundo",
                          "data": entrega[dentro.index],
                          "link": [LINK_DO_FNET.format(id=int(numero)) for numero in dentro["id"]]}, columns=COLUNAS)
    return fatos.sort_values(["data", "ticker"], ascending=[False, True], kind="stable").reset_index(drop=True)
