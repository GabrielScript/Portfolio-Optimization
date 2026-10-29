"""
melhores_ideias.py - Fund managers' best ideas from the CVM CDA (docs/superpowers/specs/2026-10-29-cda-best-ideas-design.md).

Cohen, Polk and Silli (2010): a manager's most overweighted stocks against the market. For each fund portfolio t: the
fund's weights in the issuers of the benchmark (the universe of month t+1, cap-weighted at the last close of t), the
qualifying rules, and its three largest active weights. For each signal month m: each fund's latest portfolio that is
entirely public by m (``publico_a_partir``), and the share of qualifying funds naming each issuer.
"""

from __future__ import annotations

from collections.abc import Iterable

import pandas as pd

from core.risco import emissor

NAV_MINIMO = 10e6              # R$
PARTE_EM_ACOES = 0.67          # stock value / nav: an equity fund by the CVM rule
EMISSORES_MINIMOS = 10
ACTIVE_SHARE_MINIMO = 0.5
IDEIAS = 3
DIAS_PARA_ENTREGAR = 10        # funds file the CDA within 10 days
MESES_DE_VALIDADE = 12         # a portfolio older than this at m is not used


def publico_a_partir(posicoes: pd.DataFrame) -> pd.Series:
    """(cnpj, month-end) -> first day of the month in which every position of that portfolio is public: a public
    position from month t+2, a confidential one from the month after ``confid_until`` + 10 days."""
    publica = (posicoes["month"].dt.to_period("M") + 2).dt.to_timestamp()
    liberada = ((posicoes["confid_until"] + pd.Timedelta(days=DIAS_PARA_ENTREGAR)).dt.to_period("M") + 1)
    conhecida = publica.where(posicoes["confid_until"].isna(), liberada.dt.to_timestamp())
    return conhecida.groupby([posicoes["cnpj"], posicoes["month"]]).max()


def carteira_usada(publico: pd.Series, mes: pd.Timestamp, corte: pd.Timestamp) -> pd.Series:
    """cnpj -> the latest portfolio month-end public by ``mes``, no older than 12 months and not after ``corte``."""
    tabela = publico.rename("publico").reset_index()
    tabela.columns = ["cnpj", "month", "publico"]
    inicio = (mes.to_period("M") - MESES_DE_VALIDADE).to_timestamp()
    validas = tabela[(tabela["publico"] <= mes) & (tabela["month"] >= inicio) & (tabela["month"] <= corte)]
    return validas.groupby("cnpj")["month"].max()


def referencia(caps: pd.DataFrame, universo: pd.DataFrame) -> pd.DataFrame:
    """(month = portfolio month-end, issuer, w_b): the cap weights of the issuers of each universe month u, dated u - 1
    day (the portfolio the benchmark serves). ``caps``: issuer, month (1st day), market_cap."""
    emissores = universo.assign(month=pd.to_datetime(universo["month"]),
                                issuer=universo["security"].map(emissor))[["month", "issuer"]].drop_duplicates()
    pesos = emissores.merge(caps.assign(month=pd.to_datetime(caps["month"]))[["issuer", "month", "market_cap"]],
                            on=["month", "issuer"], how="inner")
    pesos = pesos[pesos["market_cap"] > 0]
    pesos["w_b"] = pesos["market_cap"] / pesos.groupby("month")["market_cap"].transform("sum")
    return pesos.assign(month=pesos["month"] - pd.Timedelta(days=1))[["month", "issuer", "w_b"]]


def ideias_por_carteira(posicoes: pd.DataFrame, navs: pd.DataFrame, caps: pd.DataFrame,
                        universo: pd.DataFrame) -> pd.DataFrame:
    """(cnpj, month-end) -> qualifica, ideias (the 3 largest active weights), pesos (issuer -> w_f), active share."""
    total = posicoes.groupby(["cnpj", "month"])["value"].sum().rename("acoes")
    bench = referencia(caps, universo)
    por_emissor = (posicoes.assign(issuer=posicoes["ticker"].str[:4])
                   .groupby(["cnpj", "month", "issuer"], as_index=False)["value"].sum()
                   .merge(bench, on=["month", "issuer"], how="inner"))
    por_emissor = por_emissor[por_emissor["value"] > 0]
    por_emissor["w_f"] = por_emissor["value"] / por_emissor.groupby(["cnpj", "month"])["value"].transform("sum")
    por_emissor["ativo"] = por_emissor["w_f"] - por_emissor["w_b"]
    chave = [por_emissor["cnpj"], por_emissor["month"]]
    resumo = pd.DataFrame({
        "nomes": por_emissor.groupby(chave)["issuer"].size(),
        "fora": 1.0 - por_emissor.groupby(chave)["w_b"].sum(),         # benchmark weight the fund does not hold
        "desvio": por_emissor["ativo"].abs().groupby(chave).sum()})
    resumo["active_share"] = 0.5 * (resumo["desvio"] + resumo["fora"])
    ordem = por_emissor.sort_values(["cnpj", "month", "ativo", "w_f", "issuer"],
                                    ascending=[True, True, False, False, True])
    resumo["ideias"] = ordem.groupby(["cnpj", "month"])["issuer"].agg(lambda s: tuple(s.iloc[:IDEIAS]))
    resumo["pesos"] = por_emissor.groupby(chave).apply(lambda g: dict(zip(g["issuer"], g["w_f"])),
                                                       include_groups=False)
    resumo = resumo.join(total).join(navs.set_index(["cnpj", "month"])["nav"])
    resumo["qualifica"] = ((resumo["nav"] >= NAV_MINIMO) & (resumo["acoes"] >= PARTE_EM_ACOES * resumo["nav"])
                           & (resumo["nomes"] >= EMISSORES_MINIMOS)
                           & (resumo["active_share"] >= ACTIVE_SHARE_MINIMO))
    resumo.index.names = ["cnpj", "month"]
    return resumo[["qualifica", "ideias", "pesos", "active_share", "nomes"]]


def painel_de_ideias(ideias: pd.DataFrame, publico: pd.Series, bench: pd.DataFrame, universo: pd.DataFrame,
                     meses: Iterable[pd.Timestamp], corte: pd.Timestamp) -> pd.DataFrame:
    """(month, issuer, score, consensus, qualifying) for the issuers of each month's universe: score = share of the
    qualifying funds (each at its latest portfolio public by m) naming the issuer among their best ideas; consensus =
    mean over those funds of w_f - w_b (0 where not held or not in the benchmark)."""
    emissores_do_mes = (universo.assign(month=pd.to_datetime(universo["month"]),
                                        issuer=universo["security"].map(emissor))
                        .groupby("month")["issuer"].apply(lambda s: sorted(set(s))))
    por_mes = {t: grupo.set_index("issuer")["w_b"] for t, grupo in bench.groupby("month")}
    vazio = pd.Series(dtype=float)
    partes = []
    for mes in meses:
        emissores = emissores_do_mes.get(mes, [])
        usadas = carteira_usada(publico, mes, corte)
        chaves = [(cnpj, t) for cnpj, t in usadas.items() if (cnpj, t) in ideias.index]
        escolhidas = ideias.loc[chaves] if chaves else ideias.iloc[:0]
        escolhidas = escolhidas[escolhidas["qualifica"]]
        fundos = len(escolhidas)
        contagem = pd.Series([e for tupla in escolhidas["ideias"] for e in tupla], dtype=object).value_counts()
        soma_wf: dict[str, float] = {}
        for pesos in escolhidas["pesos"]:
            for issuer, peso in pesos.items():
                soma_wf[issuer] = soma_wf.get(issuer, 0.0) + peso
        meses_das_carteiras = escolhidas.index.get_level_values(1).value_counts()
        soma_wb = vazio
        for t, n in meses_das_carteiras.items():          # funds sharing a portfolio month share its benchmark
            soma_wb = soma_wb.add(por_mes.get(t, vazio) * n, fill_value=0.0)
        soma_wf = pd.Series(soma_wf, dtype=float)
        indice = pd.Index(emissores, name="issuer")
        divisor = fundos if fundos else float("nan")
        partes.append(pd.DataFrame({
            "month": mes, "issuer": indice,
            "score": contagem.reindex(indice).fillna(0.0).to_numpy() / divisor,
            "consensus": (soma_wf.reindex(indice).fillna(0.0) - soma_wb.reindex(indice).fillna(0.0)).to_numpy()
            / divisor,
            "qualifying": fundos}))
    return pd.concat(partes, ignore_index=True) if partes else pd.DataFrame(
        columns=["month", "issuer", "score", "consensus", "qualifying"])
