"""
Ibovespa rebalances from the CVM holdings of the funds that copy it (spec
``docs/superpowers/specs/2026-11-08-ibov-recomposicao-design.md``): monthly compositions, then each rebalance's
entrants and leavers. Pure functions; the script reads the lake and freezes the tables.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping

import pandas as pd

SUFIXO = re.compile(r"^[A-Z0-9]{4}(\d{1,2})$")
SUFIXOS_DE_ACAO = {"3", "4", "5", "6", "7", "8", "11"}       # ON, PN, PNA-PND and units; 1, 2, 9, 10, 12-15 are not


def normalizar(tickers: Iterable[str], aliases: Mapping[str, str]) -> set[str]:
    """The lake's security of each share or unit ticker (a rename maps to one security); rights and receipts out."""
    saida = set()
    for ticker in tickers:
        achado = SUFIXO.match(str(ticker).strip())
        if achado and achado.group(1) in SUFIXOS_DE_ACAO:
            saida.add(aliases.get(ticker, ticker))
    return saida


def composicoes(posicoes: pd.DataFrame, principais: list[tuple[str, str, str]], outros: list[str],
                aliases: Mapping[str, str], minimo: int = 2) -> dict[pd.Timestamp, set[str]]:
    """Month-end -> the index's securities: the main fund of the month's period (``principais`` = cnpj, first and
    last month), else the names at least ``minimo`` of the ``outros`` index funds hold that month."""
    por_fundo = {(cnpj, mes): normalizar(grupo["ticker"].dropna(), aliases)
                 for (cnpj, mes), grupo in posicoes.groupby(["cnpj", "month"])}
    saida = {}
    for mes in sorted(posicoes["month"].unique()):
        mes = pd.Timestamp(mes)
        principal = next((cnpj for cnpj, inicio, fim in principais
                          if pd.Timestamp(inicio) <= mes <= pd.Timestamp(fim) and (cnpj, mes) in por_fundo), None)
        if principal is not None:
            saida[mes] = por_fundo[(principal, mes)]
            continue
        contagem: dict[str, int] = {}
        for cnpj in outros:
            for nome in por_fundo.get((cnpj, mes), set()):
                contagem[nome] = contagem.get(nome, 0) + 1
        nomes = {nome for nome, vezes in contagem.items() if vezes >= minimo}
        if nomes:
            saida[mes] = nomes
    return saida


MESES_DE_REBALANCEAMENTO = (1, 5, 9)


def eventos(meses: Mapping[pd.Timestamp, set[str]]) -> pd.DataFrame:
    """Each rebalance's entrants (absent two month-ends before, present at both month-ends after) and leavers
    (present before, absent at both after): one row per (rebalance month's first day, security, kind). The month-end
    just before is skipped: the funds trade on the old portfolio's last session, often that month's last day."""
    linhas = []
    for mes in sorted(meses):
        antes, depois = mes - pd.offsets.MonthEnd(2), mes + pd.offsets.MonthEnd(1)
        if mes.month not in MESES_DE_REBALANCEAMENTO or antes not in meses or depois not in meses:
            continue
        dia = mes.replace(day=1)
        entradas = (meses[mes] & meses[depois]) - meses[antes]
        saidas = meses[antes] - (meses[mes] | meses[depois])
        linhas += [{"rebalanceamento": dia, "papel": papel, "tipo": "entrada"} for papel in entradas]
        linhas += [{"rebalanceamento": dia, "papel": papel, "tipo": "saida"} for papel in saidas]
    tabela = pd.DataFrame(linhas, columns=["rebalanceamento", "papel", "tipo"])
    return tabela.sort_values(["rebalanceamento", "papel"], ignore_index=True)


def previas(paginas: Iterable[Mapping], aliases: Mapping[str, str]) -> pd.DataFrame:
    """B3's preview pages (Firecrawl extractions) -> one row per (rebalance, preview number): the first page found
    for it, its publication date and its entrants and leavers as the lake's securities. Pages about other indices,
    the final portfolio (number 0) or failed extractions are dropped."""
    linhas, vistos = [], set()
    for pagina in paginas:
        dados = pagina.get("json") or {}
        numero, mes = dados.get("numero_da_previa"), str(dados.get("mes_de_vigencia") or "")
        if not dados.get("e_previa_do_ibovespa") or numero not in (1, 2, 3) or not re.fullmatch(r"\d{4}-\d{2}", mes):
            continue
        chave = (mes, numero)
        if chave in vistos:
            continue
        vistos.add(chave)
        divulgacao = pd.to_datetime(dados.get("data_de_divulgacao"), errors="coerce")
        linhas.append({"rebalanceamento": pd.Timestamp(f"{mes}-01"), "previa": int(numero), "divulgacao": divulgacao,
                       "entradas": tuple(sorted(normalizar(dados.get("entradas") or [], aliases))),
                       "saidas": tuple(sorted(normalizar(dados.get("saidas") or [], aliases))),
                       "fonte": pagina.get("url")})
    tabela = pd.DataFrame(linhas, columns=["rebalanceamento", "previa", "divulgacao", "entradas", "saidas", "fonte"])
    return tabela.sort_values(["rebalanceamento", "previa"], ignore_index=True)


def concordancia(previas_: pd.DataFrame, eventos_: pd.DataFrame, previa: int = 3) -> pd.DataFrame:
    """Per rebalance with that preview: the entrants both the preview and the CVM holdings show, and each one's own."""
    linhas = []
    for linha in previas_[previas_["previa"] == previa].itertuples():
        cvm = set(eventos_.loc[(eventos_["rebalanceamento"] == linha.rebalanceamento)
                               & (eventos_["tipo"] == "entrada"), "papel"])
        da_previa = set(linha.entradas)
        linhas.append({"rebalanceamento": linha.rebalanceamento, "ambas": tuple(sorted(da_previa & cvm)),
                       "so_previa": tuple(sorted(da_previa - cvm)), "so_cvm": tuple(sorted(cvm - da_previa))})
    return pd.DataFrame(linhas, columns=["rebalanceamento", "ambas", "so_previa", "so_cvm"])
