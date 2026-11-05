"""
titulos_publicos.py - The debt reference of the strategies comparison (the user, 2026-11-05: "índices de dívida"): a
Tesouro IPCA+ index from Tesouro Transparente's open daily prices of Tesouro Direto (``PrecoTaxaTesouroDireto.csv``,
public, through the last week). Each day, the equal-weight average of the daily change of the morning base unit price
of every Tesouro IPCA+ (NTN-B Principal, zero coupon: its price accrues with no cash flow) with at least a year left;
close to ANBIMA's IMA-B 5+ family in spirit, built only from public data (ANBIMA's own files may not be republished,
docs/dados/2026-10-21-anbima/LEIAME.md). The lake's Tesouro workbooks (cdn.tesouro.gov.br) stop at 21/08/2026.
"""

from __future__ import annotations

import io

import pandas as pd
import requests

URL = ("https://www.tesourotransparente.gov.br/ckan/dataset/df56aa42-484a-4a59-8184-7676580c81e3/resource/"
       "796d2059-14e9-44e3-80c9-2d9e30b405c1/download/PrecoTaxaTesouroDireto.csv")
TIPO_IPCA = "Tesouro IPCA+"
NOME = "Tesouro IPCA+"
PRAZO_MINIMO = pd.Timedelta(days=365)      # near maturity the bond is cash, not duration
PRAZO_S = 180
CABECALHO = {"User-Agent": "Mozilla/5.0 (Aleph Quant)"}
COLUNAS = {"Tipo Titulo": "tipo", "Data Vencimento": "vencimento", "Data Base": "data", "PU Base Manha": "pu_base"}


def baixar() -> bytes:
    resposta = requests.get(URL, headers=CABECALHO, timeout=PRAZO_S)
    resposta.raise_for_status()
    return resposta.content


def ler_precos_e_taxas(conteudo: bytes) -> pd.DataFrame:
    """The CSV (``;`` separated, dd/mm/yyyy dates, decimal commas) -> tipo, vencimento, data, pu_base."""
    tabela = pd.read_csv(io.BytesIO(conteudo), sep=";", decimal=",", encoding="latin-1", usecols=list(COLUNAS))
    tabela = tabela.rename(columns=COLUNAS)
    for coluna in ("vencimento", "data"):
        tabela[coluna] = pd.to_datetime(tabela[coluna], format="%d/%m/%Y")
    return tabela


def indice_tesouro_ipca(tabela: pd.DataFrame) -> pd.Series:
    """Daily level (1.0 on the first day) of the equal-weight Tesouro IPCA+ index described above."""
    ipca = tabela[tabela["tipo"] == TIPO_IPCA]
    pus = ipca.pivot_table(index="data", columns="vencimento", values="pu_base", aggfunc="last").sort_index()
    retornos = pus.pct_change(fill_method=None)
    vencimentos = pd.DataFrame({vencimento: vencimento for vencimento in pus.columns}, index=pus.index)
    longe = vencimentos.sub(pd.Series(pus.index, index=pus.index), axis=0) >= PRAZO_MINIMO
    medio = retornos.where(longe.shift().fillna(False).astype(bool)).mean(axis=1).fillna(0.0)
    return (1.0 + medio).cumprod().rename(NOME)
