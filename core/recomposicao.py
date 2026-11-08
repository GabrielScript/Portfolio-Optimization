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


def inicio_da_carteira(mes: pd.Timestamp, pregoes: pd.DatetimeIndex) -> pd.Timestamp:
    """The new portfolio's first session: the first B3 session on or after the month's first Monday."""
    primeiro = mes.replace(day=1)
    primeira_segunda = primeiro + pd.Timedelta(days=(7 - primeiro.weekday()) % 7)
    return pregoes[pregoes >= primeira_segunda][0]


def ultimo_pregao_antigo(mes: pd.Timestamp, pregoes: pd.DatetimeIndex) -> pd.Timestamp:
    """The old portfolio's last session, whose closing call the index funds trade in."""
    return pregoes[pregoes < inicio_da_carteira(mes, pregoes)][-1]


def data_da_terceira_previa(mes: pd.Timestamp, pregoes: pd.DatetimeIndex) -> pd.Timestamp:
    """B3's rule date of the 3rd preview: the old portfolio's penultimate session."""
    return pregoes[pregoes < ultimo_pregao_antigo(mes, pregoes)][-1]


def resultado_do_evento(precos: pd.Series, bova: pd.Series, entrada: pd.Timestamp, saida: pd.Timestamp,
                        custo_por_lado: float) -> dict[str, float] | None:
    """Total return from the entry close to the exit close, net of the cost of buying and of selling, BOVA11's over the
    same closes and their difference; None without both prices."""
    try:
        compra, venda = float(precos.loc[entrada]), float(precos.loc[saida])
        base, fim = float(bova.loc[entrada]), float(bova.loc[saida])
    except KeyError:
        return None
    if not all(pd.notna(valor) and valor > 0 for valor in (compra, venda, base, fim)):
        return None
    retorno = venda / compra * (1.0 - custo_por_lado) / (1.0 + custo_por_lado) - 1.0
    retorno_bova = fim / base - 1.0
    return {"retorno": retorno, "retorno_bova": retorno_bova, "anormal": retorno - retorno_bova}


T_MINIMO = 2.24                       # two rules, Bonferroni at 5% (spec, gate)
BLOCOS = {"2009-2014": (2009, 2014), "2015-2020": (2015, 2020), "2021-2026": (2021, 2026)}


def resumo(eventos_: pd.DataFrame, exige_blocos: bool = True) -> dict:
    """The pre-registered reading: one equal-weight abnormal return per rebalance date, their mean and t across
    dates (clustered by date), the mean of each block of years, and whether the gate passes (mean > 0, t >= 2.24 and,
    for R1, a positive mean in at least two of the three blocks)."""
    por_data = eventos_.groupby("rebalanceamento")["anormal"].mean()
    media = float(por_data.mean()) if len(por_data) else float("nan")
    desvio = float(por_data.std(ddof=1)) if len(por_data) > 1 else float("nan")
    t = media / (desvio / len(por_data) ** 0.5) if desvio and desvio == desvio else float("nan")
    anos = por_data.index.year
    blocos = {nome: float(por_data[(anos >= ini) & (anos <= fim)].mean())
              for nome, (ini, fim) in BLOCOS.items() if ((anos >= ini) & (anos <= fim)).any()}
    positivos = sum(valor > 0 for valor in blocos.values())
    passa = bool(media > 0 and t >= T_MINIMO and (positivos >= 2 or not exige_blocos))
    return {"datas": len(por_data), "eventos": len(eventos_), "media": media, "t": t,
            "mediana_eventos": float(eventos_["anormal"].median()) if len(eventos_) else float("nan"),
            "acerto_eventos": float((eventos_["anormal"] > 0).mean()) if len(eventos_) else float("nan"),
            "pior_evento": float(eventos_["anormal"].min()) if len(eventos_) else float("nan"),
            "blocos": blocos, "passa": passa}


def data_da_primeira_previa(mes: pd.Timestamp, pregoes: pd.DatetimeIndex) -> pd.Timestamp:
    """B3's rule date of the 1st preview: the first session of the month before the rebalance month."""
    return pregoes[pregoes >= mes.replace(day=1) - pd.offsets.MonthBegin(1)][0]


def data_de_entrada(regra: pd.Timestamp, divulgacao: pd.Timestamp, saida: pd.Timestamp,
                    pregoes: pd.DatetimeIndex) -> pd.Timestamp:
    """The entry session: B3's rule date, or the found publication's session when it is later and still before the
    exit (never earlier than the rule: no buying before the information exists)."""
    if pd.isna(divulgacao):
        return regra
    sessao = pregoes[pregoes >= divulgacao]
    if sessao.empty or sessao[0] >= saida or sessao[0] <= regra:
        return regra
    return sessao[0]
