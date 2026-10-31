"""
us_avaliar.py - The Alepha strategies on US stocks (pre-registered in ``docs/superpowers/specs/2026-10-31-us-estrategias-design.md``,
commit ``5e4596e``, before any result). No network: the US lake's exports, the B3 live curves and the funds' CVM quotas.

    python -X utf8 scripts/us_avaliar.py > docs/us-estrategias-2026-10-31.txt

Also writes docs/dados/us-*: phase 0's curves, the tables and today's picks. Derived numbers only, what the app shows.
The raw provider files never leave D:.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core import us_teste as ut  # noqa: E402
from core.backtest import estrategia_equiponderada, walk_forward_dinamico  # noqa: E402
from core.classicas import multiplo_do_adquirente_no_mes, piotroski_no_mes  # noqa: E402
from core.composto import estrategia_por_nota  # noqa: E402
from core.melhores_ideias import NAV_MINIMO, PARTE_EM_ACOES  # noqa: E402
from core.metrics import momentos_sharpe, sharpe_probabilistico, sharpe_referencia_deflacionado  # noqa: E402
from core.robustez import psr_da_diferenca  # noqa: E402
from core.sinais import NOMES, POR_SETOR, carteira_da_nota, conservadora, estrategia_conservadora  # noqa: E402
from core.top_fundos import retornos_mensais  # noqa: E402
from core.valor import Balancos, NotaDoMes, formula_magica_no_mes, trending_value  # noqa: E402
from scripts.avaliar_robustez import contar_tentativas  # noqa: E402
from scripts.paper_v3 import nota_do_composto  # noqa: E402

RAIZ = Path(__file__).resolve().parent.parent
LAGO_US = Path("D:/b3-datalake/exports/us_stocks")
LAGO_US_PISO_5 = Path("D:/b3-datalake/exports/us_stocks_piso5")
EXPORTS = Path("D:/b3-datalake/exports")
DOIS_ANOS = 24                  # the funds' history read before the first decision, enough for a qualified 12-month rank
DEFASAGEM_DA_CDA = 3            # a month's book is public about 3 months later (t+2, plus the month that closes)
FUNDOS_NOMEADOS = {"53.147.029/0001-67": "Capyvara", "45.295.132/0001-26": "AMI", "36.327.455/0001-63": "Trend EUA 2x"}
CURVAS_B3 = RAIZ / "docs" / "dados" / "curvas-ao-vivo.parquet"
PRECOS_B3 = RAIZ / "lago" / "precos_diarios.parquet"
SAIDA = RAIZ / "docs" / "dados"     # what the app shows: us-curvas, us-resumo, us-portao, us-carteiras, us-cobertura
PERDA_TERMINAL = -0.30          # Shumway (1997): the average performance-delisting return the data lacks
FASES = (0, 1, 2)               # months, the analogue of the B3 test's 0, 21 and 42 sessions
TOP_N = 15
SPY = "SPY"
ETFS_DE_REFERENCIA = (SPY, "QQQ", "IWM", "VTI")
COMPOSTO, CONSERVADORA = "Composto", "Conservadora"
BRACOS_COM_PORTAO = (COMPOSTO, CONSERVADORA, "Fórmula Mágica", "Trending Value", "Piotroski", "Múltiplo do Adquirente")
UM_PARA_N = "Universo inteiro, pesos iguais"
CANDIDATOS_MORTOS = 1_505       # the dead names on Tiingo's list that stockanalysis lacks (raw/us/tiingo/candidates.parquet)


@dataclass(frozen=True)
class Dados:
    retornos: pd.DataFrame          # month (1st day) x security
    proventos: pd.DataFrame
    universo: pd.DataFrame
    setores: dict[str, str]         # security -> sector
    balancos: Balancos
    papeis: pd.DataFrame            # securities.parquet
    paradas: pd.DataFrame           # delistings.parquet
    etfs: pd.DataFrame              # month (1st day) x ETF
    fatores: pd.DataFrame           # month ends x factor
    ptax: pd.Series                 # month ends
    cobertura: dict


def _meses(tabela: pd.DataFrame) -> pd.DataFrame:
    return tabela.set_index(pd.DatetimeIndex(pd.to_datetime(tabela.pop("month")), name=None)).astype("float64")


def carregar(pasta: Path) -> Dados:
    papeis = pd.read_parquet(pasta / "securities.parquet")
    proventos = pd.read_parquet(pasta / "dividends.parquet")
    proventos["ex_date"] = pd.to_datetime(proventos["ex_date"])
    universo = pd.read_parquet(pasta / "universe_monthly.parquet")
    universo["month"] = pd.to_datetime(universo["month"])
    fundamentos = pd.read_parquet(pasta / "fundamentals.parquet")
    for coluna in ("dt_refer", "available_from"):
        fundamentos[coluna] = pd.to_datetime(fundamentos[coluna])
    valor_mercado = pd.read_parquet(pasta / "market_caps.parquet")
    valor_mercado["month"] = pd.to_datetime(valor_mercado["month"])
    por_emissor = pd.read_parquet(pasta / "sectors.parquet").set_index("issuer")["sector"].to_dict()
    fatores = _meses(pd.read_parquet(pasta / "factors_monthly.parquet"))
    fatores.index = fatores.index + pd.offsets.MonthEnd(0)
    ptax = pd.read_parquet(pasta / "ptax_daily.parquet")
    ptax = ptax.set_index(pd.DatetimeIndex(pd.to_datetime(ptax["date"])))["ptax"].sort_index().astype(float)
    return Dados(_meses(pd.read_parquet(pasta / "monthly_total_returns.parquet")), proventos, universo,
                 papeis.set_index("security")["sector"].dropna().to_dict(),
                 Balancos(fundamentos, valor_mercado, por_emissor), papeis, pd.read_parquet(pasta / "delistings.parquet"),
                 _meses(pd.read_parquet(pasta / "etf_monthly_returns.parquet")), fatores,
                 ptax.resample("ME").last(), json.loads((pasta / "manifest.json").read_text(encoding="utf-8"))["coverage"])


def notas_das_classicas(dados: Dados) -> dict[str, NotaDoMes]:
    balancos = dados.balancos
    return {"Fórmula Mágica": lambda mes, candidatos: formula_magica_no_mes(balancos, mes, candidatos),
            "Trending Value": lambda mes, candidatos: trending_value(balancos, dados.retornos, dados.proventos, mes,
                                                                     candidatos),
            "Piotroski": lambda mes, candidatos: piotroski_no_mes(balancos, mes, candidatos),
            "Múltiplo do Adquirente": lambda mes, candidatos: multiplo_do_adquirente_no_mes(balancos, mes, candidatos)}


def estrategias(dados: Dados) -> dict[str, Callable]:
    """Each strategy as the engine runs it on the B3, on the monthly grid."""
    bracos = {COMPOSTO: estrategia_por_nota(nota_do_composto(dados.retornos, dados.proventos, dados.balancos), dados.setores),
              CONSERVADORA: estrategia_conservadora(dados.retornos, dados.proventos, dados.setores),
              **{nome: estrategia_por_nota(nota, dados.setores) for nome, nota in notas_das_classicas(dados).items()},
              UM_PARA_N: estrategia_equiponderada()}
    return {nome: ut.no_grid_mensal(estrategia) for nome, estrategia in bracos.items()}


def curvas_da_fase(dados: Dados, fase: int, terminais: pd.Series | None = None,
                   sem: frozenset[str] = frozenset()) -> dict[str, object]:
    """Every arm's walk-forward in one phase; ``sem``: securities taken out of the run."""
    retornos = dados.retornos.drop(columns=list(sem))
    universo = dados.universo[~dados.universo["security"].isin(sem)]
    precos = ut.indice_total(retornos, terminais).iloc[fase:]
    return {nome: walk_forward_dinamico(precos, universo, estrategia, ut.CONFIG, nome)
            for nome, estrategia in estrategias(dados).items()}


def medidas_da_fase(curvas: Mapping[str, pd.Series], spy: pd.Series, rf: pd.Series, n: int,
                    fatores: pd.DataFrame) -> pd.DataFrame:
    momentos = {nome: momentos_sharpe(valores.pct_change().dropna() - spy.pct_change().dropna())
                for nome, valores in curvas.items()}
    sr0 = sharpe_referencia_deflacionado(float(np.var([m.sharpe for m in momentos.values()], ddof=1)), n)
    linhas = {}
    for nome, valores in curvas.items():
        excesso = valores.pct_change().dropna() - rf.reindex(valores.index[1:]).fillna(0.0)
        alfa, t_alfa, _ = ut.alfa_fatorial(excesso, fatores)
        linhas[nome] = {**ut.medidas(valores, rf), "Alfa a.a.": alfa, "t": t_alfa,
                        "P[> SPY]": psr_da_diferenca(valores, spy), "DSR": sharpe_probabilistico(momentos[nome], sr0)}
    return pd.DataFrame(linhas).T


def anos(curvas: Mapping[str, pd.Series]) -> pd.DataFrame:
    """Each calendar year's return of each curve."""
    return pd.DataFrame({nome: valores.resample("YE").last().pct_change().dropna().rename(lambda data: data.year)
                         for nome, valores in curvas.items()})


def _texto(tabela: pd.DataFrame, casas: int = 3) -> str:
    return tabela.to_string(float_format=lambda x: f"{x:.{casas}f}")


def primeiras_carteiras_de_acoes(desde: pd.Timestamp) -> pd.Series:
    """CNPJ -> the first month its CDA book was an equity fund's: stocks at 67% of a NAV of R$ 10 million or more, the rule
    of the earlier fund tests (``core.melhores_ideias``), on stock value alone."""
    filtro = [("month", ">=", desde.date())]
    acoes = (pd.read_parquet(EXPORTS / "fund_stock_holdings.parquet", columns=["cnpj", "month", "value"], filters=filtro)
             .groupby(["cnpj", "month"], as_index=False)["value"].sum())
    navs = pd.read_parquet(EXPORTS / "fund_nav.parquet", columns=["cnpj", "month", "nav"], filters=filtro)
    juntos = acoes.merge(navs, on=["cnpj", "month"])
    boas = juntos[(juntos["nav"] >= NAV_MINIMO) & (juntos["value"] >= PARTE_EM_ACOES * juntos["nav"])]
    return pd.to_datetime(boas.groupby("cnpj")["month"].min())


def retornos_de_fundos(cnpjs: list[str], desde: pd.Timestamp) -> pd.DataFrame:
    """Month (1st day) x CNPJ of the CVM's month-end quotas."""
    colunas = ["cnpj", "subclass", "month", "date", "quota", "nav"]
    cotas = pd.read_parquet(EXPORTS / "fund_monthly_quota.parquet", columns=colunas,
                            filters=[("cnpj", "in", cnpjs), ("month", ">=", desde.date())])
    cotas["month"], cotas["date"] = pd.to_datetime(cotas["month"]), pd.to_datetime(cotas["date"])
    return retornos_mensais(cotas).pivot(index="month", columns="cnpj", values="r").sort_index()


def fundos_de_acoes(inicio: pd.Timestamp) -> pd.DataFrame:
    """Each equity fund's returns once its book qualified and was public, from 14 months before ``inicio``."""
    desde = inicio - pd.DateOffset(months=DOIS_ANOS)
    primeiras = primeiras_carteiras_de_acoes(desde)
    return ut.so_depois_de_qualificar(retornos_de_fundos(list(primeiras.index), desde), primeiras, DEFASAGEM_DA_CDA)


def tres_fundos(datas: pd.DatetimeIndex) -> pd.DataFrame:
    """The three funds the user named, on ``datas``, each from 1 at its first month; empty before it existed."""
    mensais = retornos_de_fundos(list(FUNDOS_NOMEADOS), datas[0] - pd.DateOffset(months=1)).rename(columns=FUNDOS_NOMEADOS)
    mensais.index = mensais.index + pd.offsets.MonthEnd(0)
    return (1.0 + mensais.reindex(datas)).cumprod()


def curvas_b3(datas: pd.DatetimeIndex) -> pd.DataFrame:
    """The live rerun's phase 0 and IVVB11 at the month ends of ``datas``, in R$."""
    vivas = pd.read_parquet(CURVAS_B3)
    vivas = vivas[vivas["fase"] == 0].set_index(pd.DatetimeIndex(pd.to_datetime(vivas.loc[vivas["fase"] == 0, "date"])))
    ivvb = pd.read_parquet(PRECOS_B3, columns=["date", "IVVB11"])
    ivvb = ivvb.set_index(pd.DatetimeIndex(pd.to_datetime(ivvb["date"])))["IVVB11"]
    juntas = pd.concat([vivas[["Composto", "Conservadora", "BOVA11", "CDI"]].rename(
        columns=lambda nome: f"B3 {nome}" if nome in ("Composto", "Conservadora") else nome),
        ivvb.rename("IVVB11")], axis=1).sort_index().ffill()
    return juntas.reindex(datas, method="ffill")


def comparacao_em_reais(dados: Dados, curvas: Mapping[str, pd.Series]) -> pd.DataFrame:
    """The pre-registered R$ table on phase 0's dates: the US arms and ETFs through PTAX, then funds and the B3."""
    datas = curvas[COMPOSTO].index
    inicio = (datas[0] + pd.Timedelta(days=1)).to_period("M").to_timestamp()
    fim = datas[-1].to_period("M").to_timestamp()
    etfs = dados.etfs.loc[:fim]
    em_dolar = {**curvas, **{nome: ut.referencia_nas_datas(dados.etfs[nome], datas) for nome in ETFS_DE_REFERENCIA},
                "15 ETFs, melhores 12 meses": ut.rotacao_trimestral(etfs, inicio, TOP_N, ut.CONFIG.custo_bps),
                "15 ETFs, melhores do período (retrospecto)": ut.melhores_em_retrospecto(etfs, inicio, TOP_N)}
    linhas = {nome: ut.em_reais(valores.reindex(datas), dados.ptax) for nome, valores in em_dolar.items()}
    fundos = fundos_de_acoes(inicio).loc[:fim]
    linhas["15 fundos BR, melhores 12 meses"] = ut.rotacao_trimestral(fundos, inicio, TOP_N, 0.0).reindex(datas)
    linhas["15 fundos BR, melhores do período (retrospecto)"] = ut.melhores_em_retrospecto(fundos, inicio, TOP_N).reindex(datas)
    linhas |= {nome: serie for nome, serie in curvas_b3(datas).items()}
    linhas |= {nome: serie for nome, serie in tres_fundos(datas).items()}
    return pd.DataFrame(linhas)


def carteiras_de_hoje(dados: Dados) -> pd.DataFrame:
    """The 15 names each main strategy would hold at the decision after the last closed month. Information, not an order."""
    mes = dados.universo["month"].max()
    historia = dados.retornos.loc[dados.retornos.index < mes].tail(ut.CONFIG.janela_treino)
    com_historia = set(historia.columns[historia.notna().all()])
    candidatos = [papel for papel in dados.universo.loc[dados.universo["month"] == mes, "security"] if papel in com_historia]
    notas: dict[str, NotaDoMes] = {
        COMPOSTO: nota_do_composto(dados.retornos, dados.proventos, dados.balancos),
        CONSERVADORA: lambda dia, nomes: conservadora(dados.retornos, dados.proventos, dia, nomes)}
    papeis = dados.papeis.set_index("security")
    partes = []
    for carteira, nota in notas.items():
        pesos = carteira_da_nota(nota(mes, candidatos).dropna(), dados.setores, NOMES, POR_SETOR)
        partes.append(pd.DataFrame({"carteira": carteira, "mes": mes, "ticker": papeis.loc[pesos.index, "symbol"].to_numpy(),
                                    "empresa": papeis.loc[pesos.index, "name"].to_numpy(),
                                    "setor": papeis.loc[pesos.index, "sector"].to_numpy(), "peso": pesos.to_numpy()}))
    return pd.concat(partes, ignore_index=True)


def gravar(reais: pd.DataFrame, resumo: pd.DataFrame, portao: pd.DataFrame, carteiras: pd.DataFrame, dados: Dados,
           n: int) -> None:
    """The derived files the app reads, which the deployed app gets through git."""
    SAIDA.mkdir(parents=True, exist_ok=True)
    (reais / reais.bfill().iloc[0]).rename_axis("date").reset_index().to_parquet(SAIDA / "us-curvas.parquet", index=False)
    resumo.rename_axis("nome").reset_index().to_parquet(SAIDA / "us-resumo.parquet", index=False)
    portao.astype({"P[> SPY] mínimo": float, "DSR mínimo": float, "Passou": bool}).rename_axis("nome").reset_index().to_parquet(
        SAIDA / "us-portao.parquet", index=False)
    carteiras.to_parquet(SAIDA / "us-carteiras.parquet", index=False)
    cobertura = {**dados.cobertura, "dead_from_tiingo": int((dados.papeis["source"] == "tiingo").sum()),
                 "n_tentativas": n, "mortos_candidatos": CANDIDATOS_MORTOS}
    (SAIDA / "us-cobertura.json").write_bytes(json.dumps(cobertura, indent=1, sort_keys=True, default=str).encode("utf-8"))


def main() -> int:
    argumentos = argparse.ArgumentParser()
    argumentos.add_argument("--sem-sensibilidades", action="store_true")
    opcoes = argumentos.parse_args()
    inicio = time.perf_counter()
    dados = carregar(LAGO_US)
    n = contar_tentativas()
    print(f"Coverage: {dados.cobertura}; DSR N {n}. Dead names from Tiingo in this run: "
          f"{int((dados.papeis['source'] == 'tiingo').sum())} (list of {CANDIDATOS_MORTOS:,} candidates, largest first).")
    rf = dados.fatores["rf"]
    p_spy: dict[str, list[float]] = {nome: [] for nome in BRACOS_COM_PORTAO}
    dsr: dict[str, list[float]] = {nome: [] for nome in BRACOS_COM_PORTAO}
    fase_zero = {}
    for fase in FASES:
        resultados = curvas_da_fase(dados, fase)
        curvas = {nome: resultado.valores.dropna() for nome, resultado in resultados.items()}
        datas = curvas[COMPOSTO].index
        spy = ut.referencia_nas_datas(dados.etfs[SPY], datas)
        tabela = medidas_da_fase(curvas, spy, rf, n, dados.fatores)
        giro = {nome: resultado.turnover_anual_medio for nome, resultado in resultados.items()}
        print(f"\n=== Phase {fase}: {datas[0]:%m/%Y} to {datas[-1]:%m/%Y}, {len(datas) - 1} months; refusals "
              f"{ {nome: len(r.recusas) for nome, r in resultados.items()} } ===")
        print(_texto(tabela.assign(Giro=pd.Series(giro))))
        print(f"SPY: {ut.medidas(spy, rf)}")
        print(_texto(anos({**curvas, SPY: spy}) * 100, 1))
        for nome in BRACOS_COM_PORTAO:
            p_spy[nome].append(float(tabela.loc[nome, "P[> SPY]"]))
            dsr[nome].append(float(tabela.loc[nome, "DSR"]))
        if fase == 0:
            fase_zero = curvas
        print(f"(phase {fase} done in {time.perf_counter() - inicio:.0f} s)", flush=True)

    print("\n=== Gate: P[> SPY] and DSR >= 0.95 in all three phases ===")
    portao = pd.DataFrame({nome: {"P[> SPY] mínimo": min(p_spy[nome]), "DSR mínimo": min(dsr[nome]),
                                  "Passou": ut.passa_o_portao(p_spy[nome], dsr[nome])} for nome in BRACOS_COM_PORTAO}).T
    for nome in BRACOS_COM_PORTAO:
        print(f"{nome}: P {[round(p, 3) for p in p_spy[nome]]}, DSR {[round(d, 3) for d in dsr[nome]]}: "
              + ("CANDIDATE for a forward log" if ut.passa_o_portao(p_spy[nome], dsr[nome]) else "not a candidate"))

    reais = comparacao_em_reais(dados, fase_zero)
    resumo = pd.DataFrame({nome: ut.medidas(serie.dropna(), pd.Series(0.0, index=serie.dropna().index))
                           for nome, serie in reais.items()}).T.drop(columns="Sharpe")
    resumo["R$ 1 vira"] = pd.Series({nome: serie.dropna().iloc[-1] / serie.dropna().iloc[0] for nome, serie in reais.items()})
    resumo["Desde"] = pd.Series({nome: serie.dropna().index[0] for nome, serie in reais.items()})
    print("\n=== In R$, phase 0's dates. The hindsight lines could not have been chosen at the start ===")
    print(_texto(resumo.sort_values("CAGR", ascending=False)))
    gravar(reais, resumo, portao, carteiras_de_hoje(dados), dados, n)

    if not opcoes.sem_sensibilidades:
        print("\n=== Sensitivities, phase 0: CAGR ===")
        paradas = dados.paradas[dados.paradas["kind"] != "acquired"]["security"]
        tiingo = frozenset(dados.papeis.loc[dados.papeis["source"] == "tiingo", "security"])
        casos = {"Base": curvas_da_fase(dados, 0),
                 "Perda de 30% ao sair (quem não foi comprada)": curvas_da_fase(
                     dados, 0, pd.Series(PERDA_TERMINAL, index=[p for p in paradas if p in dados.retornos.columns])),
                 "Sem os nomes mortos do Tiingo": curvas_da_fase(dados, 0, sem=tiingo)}
        if LAGO_US_PISO_5.exists():
            casos["Piso de US$ 5 milhões"] = curvas_da_fase(carregar(LAGO_US_PISO_5), 0)
        print(_texto(pd.DataFrame({caso: {nome: ut.medidas(r.valores.dropna(), rf)["CAGR"] for nome, r in resultados.items()}
                                   for caso, resultados in casos.items()})))
    print(f"\nDone in {time.perf_counter() - inicio:.0f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
