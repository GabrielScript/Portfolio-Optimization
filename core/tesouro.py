"""
tesouro.py - Taxas prefixadas do Tesouro Direto (Tesouro Transparente), sem Streamlit.

Fonte: CSV ``PrecoTaxaTesouroDireto.csv`` (~14,6 MB; ~27 s para baixar), com
``;`` de separador, vírgula decimal e latin-1. Data Base de 2004-12-31 em diante.

Achados (sondados ao vivo em 2026-10-12 e 2026-10-13):
- Prefixados: "Tesouro Prefixado" (LTN, sem cupom) e "Tesouro Prefixado com
  Juros Semestrais" (NTN-F, com cupom). Até ~2010 os prazos longos só existem
  via NTN-F. O primeiro mês com prazo >= 5 anos é 2005-11; antes dele o
  prazo máximo não passa de 4,43 anos, e nenhum mês tem prazo máximo entre
  4,5 e 5 anos. Por isso "sem extrapolação" em 5 anos equivale ao critério
  "título com prazo >= 4,5a" do spec.
- ``Taxa Compra Manha`` vale 0 quando o título não está à venda (NTN-F 2021,
  jan-mar/2010, cuja ``Taxa Venda Manha`` de 0,06 também é lixo). Linhas com
  taxa <= 0 são descartadas.
- LTN e NTN-F podem vencer na mesma data. Fica a LTN, que é zero-cupom; a
  taxa da NTN-F é o yield de um título com cupom.

``hoje`` é sempre explícito: linhas com ``Data Base > hoje`` são descartadas.
"""

from __future__ import annotations

import io
from datetime import date
from typing import NamedTuple

import numpy as np
import pandas as pd
import requests

URL_TESOURO = ("https://www.tesourotransparente.gov.br/ckan/dataset/"
               "df56aa42-484a-4a59-8184-7676580c81e3/resource/"
               "796d2059-14e9-44e3-80c9-2d9e30b405c1/download/PrecoTaxaTesouroDireto.csv")
TIPO_LTN = "Tesouro Prefixado"
TIPO_NTNF = "Tesouro Prefixado com Juros Semestrais"
DIAS_POR_ANO = 365.25
PRAZO_CURTO_ANOS = 1.0
PRAZO_LONGO_ANOS = 5.0
_TIMEOUT_HTTP = 180
_COLUNAS = {"Tipo Titulo": "tipo", "Data Vencimento": "vencimento",
            "Data Base": "data_base", "Taxa Compra Manha": "taxa"}


class ErroTesouro(RuntimeError):
    """Falha ao obter ou interpretar o CSV do Tesouro Direto."""


class Curva(NamedTuple):
    """Taxas prefixadas (% a.a.) interpoladas em 1 e 5 anos num dia; NaN fora dos prazos."""

    data_base: pd.Timestamp
    juros_1a: float
    juros_5a: float


def ler_csv_tesouro(conteudo: bytes, hoje: date) -> pd.DataFrame:
    """
    Prefixados com ``data_base <= hoje``, um por (data_base, vencimento).

    Colunas: ``data_base``, ``vencimento`` (Timestamp), ``tipo`` (str),
    ``taxa`` (% a.a.) e ``prazo_anos`` (dias corridos / 365,25).
    """
    try:
        bruto = pd.read_csv(io.BytesIO(conteudo), sep=";", decimal=",", encoding="latin-1")
    except ValueError as erro:
        raise ErroTesouro(f"CSV do Tesouro ilegível: {erro}") from erro
    faltando = sorted(set(_COLUNAS) - set(bruto.columns))
    if faltando:
        raise ErroTesouro(f"CSV do Tesouro sem as colunas {faltando}.")

    tabela = bruto[list(_COLUNAS)].rename(columns=_COLUNAS)
    tabela = tabela[tabela["tipo"].isin((TIPO_LTN, TIPO_NTNF))].copy()
    for coluna in ("data_base", "vencimento"):
        tabela[coluna] = pd.to_datetime(tabela[coluna], format="%d/%m/%Y", errors="coerce")
    tabela["taxa"] = pd.to_numeric(tabela["taxa"], errors="coerce")
    tabela = tabela.dropna(subset=["data_base", "vencimento", "taxa"])
    tabela = tabela[(tabela["taxa"] > 0)
                    & (tabela["data_base"] <= pd.Timestamp(hoje))
                    & (tabela["vencimento"] > tabela["data_base"])]
    if tabela.empty:
        raise ErroTesouro(f"Nenhum prefixado com Data Base até {hoje}.")

    tabela["ltn_primeiro"] = tabela["tipo"] != TIPO_LTN
    tabela = (tabela.sort_values(["data_base", "vencimento", "ltn_primeiro"])
              .drop_duplicates(["data_base", "vencimento"], keep="first")
              .drop(columns="ltn_primeiro"))
    tabela["prazo_anos"] = (tabela["vencimento"] - tabela["data_base"]).dt.days / DIAS_POR_ANO
    return tabela.reset_index(drop=True)


def baixar_prefixados(hoje: date) -> pd.DataFrame:
    """Baixa o CSV inteiro (~27 s) e devolve ``ler_csv_tesouro``. Levanta ErroTesouro."""
    try:
        resposta = requests.get(URL_TESOURO, timeout=_TIMEOUT_HTTP)
        resposta.raise_for_status()
    except requests.RequestException as erro:
        raise ErroTesouro(f"Download do CSV do Tesouro: {erro}") from erro
    return ler_csv_tesouro(resposta.content, hoje)


def interpolar_taxa(prazos: np.ndarray, taxas: np.ndarray, alvo_anos: float) -> float:
    """Interpolação linear da taxa em ``alvo_anos``. NaN fora do intervalo de prazos (sem extrapolação)."""
    if len(prazos) == 0:
        return float("nan")
    ordem = np.argsort(prazos)
    prazos, taxas = np.asarray(prazos)[ordem], np.asarray(taxas)[ordem]
    if alvo_anos < prazos[0] or alvo_anos > prazos[-1]:
        return float("nan")
    return float(np.interp(alvo_anos, prazos, taxas))


def _curva_do_dia(data_base: pd.Timestamp, titulos_do_dia: pd.DataFrame) -> Curva:
    prazos = titulos_do_dia["prazo_anos"].to_numpy()
    taxas = titulos_do_dia["taxa"].to_numpy()
    return Curva(data_base, interpolar_taxa(prazos, taxas, PRAZO_CURTO_ANOS),
                 interpolar_taxa(prazos, taxas, PRAZO_LONGO_ANOS))


def curva_mais_recente(titulos: pd.DataFrame) -> Curva:
    """Curva na última Data Base disponível."""
    ultima = titulos["data_base"].max()
    return _curva_do_dia(ultima, titulos[titulos["data_base"] == ultima])


def curva_fim_de_mes(titulos: pd.DataFrame) -> pd.DataFrame:
    """
    Taxas de 1 e 5 anos (% a.a.) na última Data Base de cada mês.
    Índice: PeriodIndex mensal ``mes``. Colunas: ``juros_1a``, ``juros_5a``.
    Meses sem cotação não aparecem; quem consome reindexa.
    """
    mes = titulos["data_base"].dt.to_period("M")
    ultimo_dia = titulos["data_base"] == titulos.groupby(mes)["data_base"].transform("max")
    curvas = [_curva_do_dia(data_base, grupo)
              for data_base, grupo in titulos[ultimo_dia].groupby("data_base")]
    return pd.DataFrame(
        {"juros_1a": [c.juros_1a for c in curvas], "juros_5a": [c.juros_5a for c in curvas]},
        index=pd.PeriodIndex([c.data_base.to_period("M") for c in curvas], name="mes"),
    )
