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
    """Each rebalance's entrants (absent at the month-end before, present at both month-ends after) and leavers
    (present before, absent at both after): one row per (rebalance month's first day, security, kind)."""
    linhas = []
    for mes in sorted(meses):
        antes, depois = mes - pd.offsets.MonthEnd(1), mes + pd.offsets.MonthEnd(1)
        if mes.month not in MESES_DE_REBALANCEAMENTO or antes not in meses or depois not in meses:
            continue
        dia = mes.replace(day=1)
        entradas = (meses[mes] & meses[depois]) - meses[antes]
        saidas = meses[antes] - (meses[mes] | meses[depois])
        linhas += [{"rebalanceamento": dia, "papel": papel, "tipo": "entrada"} for papel in entradas]
        linhas += [{"rebalanceamento": dia, "papel": papel, "tipo": "saida"} for papel in saidas]
    tabela = pd.DataFrame(linhas, columns=["rebalanceamento", "papel", "tipo"])
    return tabela.sort_values(["rebalanceamento", "papel"], ignore_index=True)
