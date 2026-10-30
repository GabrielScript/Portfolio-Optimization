"""
tres_fundos.py - The three funds the user named on 2026-10-30 (DECISOES 189): Capyvara FIA RL, AMI FIF Ações and
Trend Bolsa Americana Alavancado FIF Ações RL, in their 7 combinations (the three, each pair, each alone), two ways:
investing in the funds (their CVM month-end quotas, equal parts rebalanced monthly) and copying their public CDA stock
and BDR holdings (attempt 52's publication rule; a member with no B3 holdings to copy, Trend, enters with its own
quota). Each against the Conservadora, the composite and BOVA11 on the same months (the live rerun's phase 0,
``scripts/curvas_ao_vivo.py``), with the CDI. Descriptive only: picked by name after their returns were seen, so no
trial, no gate, N stays 53.

    python -X utf8 scripts/tres_fundos.py carteiras    (once: the inputs into docs/dados/2026-10-30-tres-fundos/)
    python -X utf8 scripts/tres_fundos.py > docs/tres-fundos-2026-10-30.txt
"""

from __future__ import annotations

import hashlib
import io
import sys
import zipfile
from pathlib import Path

import pandas as pd

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))

from core.poder import ic_por_mes  # noqa: E402
from core.robustez import psr_da_diferenca  # noqa: E402
from core.top_fundos import retornos_mensais  # noqa: E402
from core.tres_fundos import (combinacoes, linhas_do_blc4, medidas_mensais, pesos_copiados,  # noqa: E402
                              retorno_da_combinacao, retorno_da_copia)
from scripts.avaliar_conservadora import resumo_do_ic  # noqa: E402
from scripts.avaliar_robustez import conferir_hashes  # noqa: E402

FUNDOS = {"53.147.029/0001-67": "Capyvara", "45.295.132/0001-26": "AMI", "36.327.455/0001-63": "Trend EUA 2x"}
CDA_BRUTO = Path("D:/b3-datalake/raw/cvm_cda")
LAGO = RAIZ.parent / "Portfolio-Optimization" / "data" / "b3lake"       # the main checkout's copy of the lake exports
PASTA = RAIZ / "docs" / "dados" / "2026-10-30-tres-fundos"
CURVAS = RAIZ / "docs" / "dados" / "curvas-ao-vivo.parquet"
ARQUIVOS = ("carteiras.parquet", "cotas.parquet", "retornos_acoes.parquet", "universo.parquet")
REFERENCIAS = ("Conservadora", "Composto", "BOVA11")
PRIMEIRO_ANO = 2020


def _somas(pasta: Path) -> None:
    linhas = [f"{hashlib.sha256((pasta / nome).read_bytes()).hexdigest()} *{nome}\n" for nome in ARQUIVOS]
    (pasta / "SHA256SUMS").write_text("".join(linhas), encoding="ascii", newline="\n")


def congelar_entradas() -> None:
    """The 3 funds' BLC_4 rows of every raw CDA zip since 2020, their quotas, and the lake's monthly total returns
    and universe of the names they ever held, frozen with SHA256SUMS."""
    partes = []
    for arquivo in sorted(CDA_BRUTO.glob("cda_fi_20[2-9]*.zip")):
        with zipfile.ZipFile(arquivo) as zf:
            for nome in (membro for membro in zf.namelist() if "BLC_4" in membro):
                with zf.open(nome) as bruto:
                    partes.append(linhas_do_blc4(io.TextIOWrapper(bruto, encoding="latin-1"), set(FUNDOS)))
    carteiras = pd.concat(partes, ignore_index=True).drop_duplicates(ignore_index=True)
    PASTA.mkdir(parents=True, exist_ok=True)
    carteiras.to_parquet(PASTA / "carteiras.parquet", index=False)
    pd.read_parquet(LAGO / "fund_monthly_quota.parquet", filters=[("cnpj", "in", list(FUNDOS))]).to_parquet(
        PASTA / "cotas.parquet", index=False)
    universo = pd.read_parquet(LAGO / "universe_monthly.parquet", columns=["month", "security"])
    universo = universo[pd.to_datetime(universo["month"]).dt.year >= PRIMEIRO_ANO - 1]
    nomes = sorted(set(carteiras["ticker"]) | set(universo["security"]))
    retornos = pd.read_parquet(LAGO / "monthly_total_returns.parquet")
    retornos = retornos[pd.to_datetime(retornos["month"]).dt.year >= PRIMEIRO_ANO - 1]
    retornos[["month", *[nome for nome in nomes if nome in retornos.columns]]].to_parquet(
        PASTA / "retornos_acoes.parquet", index=False)
    universo.to_parquet(PASTA / "universo.parquet", index=False)
    _somas(PASTA)
    print(f"{len(carteiras)} CDA rows: " + "; ".join(
        f"{FUNDOS[cnpj]} {grupo['month'].min():%m/%Y}-{grupo['month'].max():%m/%Y}, {grupo['month'].nunique()} books"
        for cnpj, grupo in carteiras.groupby("cnpj")))


def _mensal(valores: pd.Series) -> pd.Series:
    """Month (1st day) -> the month's return of a daily value curve (month-end over the previous month-end)."""
    fechamentos = valores.dropna().groupby(valores.dropna().index.to_period("M")).last()
    return fechamentos.pct_change().dropna().set_axis(fechamentos.index[1:].to_timestamp())


def _ler_entradas() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, str]]:
    sha256 = conferir_hashes(PASTA, list(ARQUIVOS))
    carteiras = pd.read_parquet(PASTA / "carteiras.parquet")
    cotas = pd.read_parquet(PASTA / "cotas.parquet")
    for coluna in ("month", "date"):
        cotas[coluna] = pd.to_datetime(cotas[coluna])
    fundos = retornos_mensais(cotas).pivot(index="month", columns="cnpj", values="r")
    acoes = pd.read_parquet(PASTA / "retornos_acoes.parquet")
    acoes = acoes.set_index(pd.DatetimeIndex(pd.to_datetime(acoes.pop("month")), name=None))
    universo = pd.read_parquet(PASTA / "universo.parquet")
    universo["month"] = pd.to_datetime(universo["month"])
    return carteiras, fundos, acoes, universo, sha256


def _copia(carteiras: pd.DataFrame, fundos: pd.DataFrame, acoes: pd.DataFrame, cnpj: str) -> pd.Series:
    """One member's copy, month by month: its latest public book on the names' total returns; the fund's own quota
    when it never had stocks or BDRs to copy."""
    if cnpj not in set(carteiras["cnpj"]):
        return fundos[cnpj]
    proprias = carteiras[carteiras["cnpj"] == cnpj]
    return pd.Series({mes: retorno_da_copia(pesos_copiados(proprias, mes, (cnpj,)), acoes.loc[mes])
                      for mes in acoes.index}, dtype=float)


def _ic(carteiras: pd.DataFrame, acoes: pd.DataFrame, universo: pd.DataFrame, membros: tuple[str, ...]) -> pd.Series:
    """Monthly Spearman IC of the copy's weight (0 for a universe name not held) against the month's total return,
    in the month's B3 stock universe; empty when no member holds a universe stock."""
    meses, sinais = [], []
    for mes in acoes.index:
        no_universo = universo.loc[universo["month"] == mes, "security"]
        pesos = pesos_copiados(carteiras, mes, membros)
        if pesos.empty or no_universo.empty or not pesos.index.isin(no_universo).any():
            continue
        meses.append(mes)
        sinais.append(pesos.reindex(no_universo).fillna(0.0).rename(None))
    if not meses:
        return pd.Series(dtype=float)
    sinal = pd.DataFrame(sinais, index=meses)
    return ic_por_mes(sinal, acoes.reindex(index=meses, columns=sinal.columns))


def main() -> int:
    carteiras, fundos, acoes, universo, sha256 = _ler_entradas()
    curvas = pd.read_parquet(CURVAS)
    fase0 = curvas[curvas["fase"] == 0].set_index("date")
    refs = pd.DataFrame({nome: _mensal(fase0[nome]) for nome in REFERENCIAS})
    cdi = (1 + fase0["CDI"].fillna(0.0)).groupby(fase0.index.to_period("M")).prod() - 1
    cdi.index = cdi.index.to_timestamp()
    ultimo = min(fundos.index.max(), refs.index.max() - pd.offsets.MonthBegin(1))   # full months only
    fundos, refs, acoes = fundos.loc[:ultimo], refs.loc[:ultimo], acoes.loc[:ultimo]
    copias = pd.DataFrame({cnpj: _copia(carteiras, fundos, acoes, cnpj) for cnpj in FUNDOS}).loc[:ultimo]
    print(f"DESCRIPTIVE (no trial, no gate, N stays 53): the 3 funds the user named, picked after their returns were "
          f"seen. Inputs {PASTA.relative_to(RAIZ).as_posix()} (sha256 {sha256}); references from "
          f"{CURVAS.relative_to(RAIZ).as_posix()} phase 0 to {fase0.index[-1]:%d/%m/%Y}; months to {ultimo:%m/%Y}.")
    print("Copy: each member's latest CDA book public at the month (t+2, or after confid_until + 10 days), stocks "
          "and BDRs by value (lent shares included), members in equal parts; Trend has no B3 stock or BDR (an S&P "
          "500 futures ETF, futures and Treasuries), so its sleeve is its own quota in both views.")
    for cnpj, nome in FUNDOS.items():
        proprias = carteiras[carteiras["cnpj"] == cnpj]
        if proprias.empty:
            inicio = fundos[cnpj].first_valid_index()
            print(f"  {nome}: no stock or BDR position in the CDA (quota from {inicio:%m/%Y})")
            continue
        ultima = proprias[proprias["month"] == proprias["month"].max()].groupby("ticker")["value"].sum()
        topo = ", ".join(f"{papel} {peso:.0%}" for papel, peso in (ultima / ultima.sum()).nlargest(5).items())
        print(f"  {nome}: {proprias['month'].nunique()} books {proprias['month'].min():%m/%Y}-"
              f"{proprias['month'].max():%m/%Y}, {proprias.groupby('month')['ticker'].nunique().median():.0f} names "
              f"(median); latest {proprias['month'].max():%m/%Y}: {topo}")

    linhas, anos = [], {}
    for membros in combinacoes(list(FUNDOS)):
        rotulo = " + ".join(FUNDOS[cnpj] for cnpj in membros)
        for modo, tabela in (("investir", fundos), ("copiar", copias)):
            serie = retorno_da_combinacao(tabela, membros)
            if serie.empty:
                continue
            janela = serie.index
            linha = {"Combinação": rotulo, "Modo": modo, "De": f"{janela[0]:%m/%Y}",
                     **medidas_mensais(serie, refs["BOVA11"], cdi)}
            valores = (1 + serie).cumprod()
            for ref in ("Conservadora", "Composto"):
                linha[f"CAGR {ref}"] = medidas_mensais(refs.loc[janela, ref], refs["BOVA11"], cdi)["CAGR"]
                linha[f"P[> {ref}]"] = psr_da_diferenca(valores, (1 + refs.loc[janela, ref]).cumprod(), log=True)
            linha["CAGR BOVA11"] = medidas_mensais(refs.loc[janela, "BOVA11"], refs["BOVA11"], cdi)["CAGR"]
            ic = _ic(carteiras, acoes, universo, membros) if modo == "copiar" else pd.Series(dtype=float)
            resumo = resumo_do_ic(ic) if len(ic) > 1 else None
            linha["IC médio"] = float(resumo["IC médio"]) if resumo is not None else float("nan")
            linha["t do IC"] = float(resumo["t"]) if resumo is not None else float("nan")
            linha["Meses de IC"] = int(resumo["meses"]) if resumo is not None else 0
            linhas.append(linha)
            anos[f"{rotulo} ({modo})"] = (1 + serie).groupby(serie.index.year).prod() - 1
    tabela = pd.DataFrame(linhas)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 40)
    porcento = ["Retorno total", "CAGR", "Vol", "Queda máx.", "Alfa a.a. (BOVA11)", "CAGR Conservadora",
                "CAGR Composto", "CAGR BOVA11"]
    decimais = ("Sharpe", "Beta (BOVA11)", "t do alfa", "P[> Conservadora]", "P[> Composto]", "IC médio", "t do IC")
    print("\n" + tabela.to_string(index=False, formatters={**{c: "{:.1%}".format for c in porcento},
                                                           **{c: "{:.3f}".format for c in decimais}}))
    for ref in REFERENCIAS:
        anos[ref] = (1 + refs[ref]).groupby(refs.index.year).prod() - 1
    anos["CDI"] = (1 + cdi.loc[:ultimo]).groupby(cdi.loc[:ultimo].index.year).prod() - 1
    por_ano = pd.DataFrame(anos)
    print("\nCalendar-year returns (a fund's first year starts at its first month):\n"
          + por_ano.loc[por_ano.index >= PRIMEIRO_ANO].T.to_string(float_format=lambda x: f"{x:.1%}"))
    return 0


if __name__ == "__main__":
    if sys.argv[1:] == ["carteiras"]:
        congelar_entradas()
        raise SystemExit(0)
    raise SystemExit(main())
