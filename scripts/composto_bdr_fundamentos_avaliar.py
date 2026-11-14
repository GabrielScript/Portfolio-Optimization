"""
The Composite's score on B3 BDRs: SEC fundamentals, B3 prices, 15 names a month, alone and mixed with the B3
Composite at 90/10 to 50/50 (spec ``docs/superpowers/specs/2026-11-14-composto-bdr-fundamentos-design.md``, pre-registered
before this ran).

    python -X utf8 scripts/composto_bdr_fundamentos_avaliar.py > docs/composto-bdr-fundamentos-2026-11-14.txt
"""

from __future__ import annotations

import datetime as dt
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))

from core import bdr_fundamentos as bf  # noqa: E402
from core import manga_bdr as mb  # noqa: E402
from core.sinais import NOMES, POR_SETOR, carteira_da_nota  # noqa: E402
from core.valor import Balancos  # noqa: E402
from scripts.paper_v3 import nota_do_composto  # noqa: E402
from scripts.xquad_avaliar import COMPOSITE, DSR_MIN, P_MIN, metricas  # noqa: E402

GOLD = Path("D:/b3-datalake/gold")
US = Path("D:/b3-datalake/exports/us_stocks")
MAPA = RAIZ / "docs" / "dados" / "bdr-sec-mapa-2026-11-14.csv"
CURVAS = RAIZ / "docs" / "dados" / "pos-idade-curvas-fase{}-2026-11-13.parquet"
CDI = RAIZ.parent / "Portfolio-Optimization" / "data" / "b3lake" / "cdi_daily.parquet"   # git-ignored: the main checkout
SAIDA = RAIZ / "docs" / "dados" / "composto-bdr-fundamentos-escolhas-2026-11-14.csv"
LEITURA, FIM = dt.date(2016, 1, 1), pd.Timestamp("2026-10-16")
INICIO = pd.Timestamp("2021-01-04")
PRESENCA, VOLUME_MINIMO = 0.90, 200_000.0
PESOS, FASES = (0.10, 0.20, 0.30, 0.40, 0.50), (0, 21, 42)
N_DSR = 188                             # 182 after DECISOES 368 + these 6 rows
SO_BDR = "CompBDR"
RENOMEADO = ("FBOK34", "M1TA34")        # Meta before and after the rename
SEGUNDA_CLASSE = "GOGL35"               # Alphabet's other class: one issuer, the rule keeps one
COLUNAS = ["CAGR", "Vol", "Sharpe", "Sortino", "MDD", "Calmar", "P[> Comp]", "DSR", "alpha vs Comp", "t vs Comp"]


def carregar_b3(mapa: pd.Series) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.DataFrame]:
    """Corrected daily returns and volumes of the mapped BDRs under their US codes, BOVA11, and the jumps used."""
    nomes = list(mapa.index)
    filtros = [("date", ">=", LEITURA), ("date", "<=", FIM.date())]
    bdr = pq.read_table(GOLD / "equity_adjusted.parquet", columns=["date", "security", "tr_close", "volume_brl"],
                        filters=[*filtros, ("security", "in", nomes)]).to_pandas()
    bova = pq.read_table(GOLD / "equity_adjusted.parquet", columns=["date", "tr_close"],
                         filters=[*filtros, ("security", "==", "BOVA11")]).to_pandas()
    calendario = pd.DatetimeIndex(pd.to_datetime(bova["date"])).sort_values()
    bdr["date"] = pd.to_datetime(bdr["date"])
    precos = bdr.pivot(index="date", columns="security", values="tr_close").reindex(calendario)
    volumes = bdr.pivot(index="date", columns="security", values="volume_brl").reindex(calendario)
    saltos = pd.read_parquet(GOLD / "jumps.parquet", columns=["security", "date"])
    saltos = saltos[saltos["security"].isin(nomes)].assign(date=lambda t: pd.to_datetime(t["date"]))
    retornos = bf.retornos_corrigidos(precos, saltos)
    velho, novo = RENOMEADO
    retornos[novo], volumes[novo] = bf.juntar(retornos, velho, novo), bf.juntar(volumes, velho, novo)
    retornos = retornos.drop(columns=[velho, SEGUNDA_CLASSE], errors="ignore")
    volumes = volumes.drop(columns=[velho, SEGUNDA_CLASSE], errors="ignore")
    return (retornos.rename(columns=mapa.to_dict()), volumes.rename(columns=mapa.to_dict()),
            pd.Series(bova["tr_close"].to_numpy(float), index=calendario), saltos)


def carregar_sec() -> tuple[Balancos, dict[str, str], pd.DataFrame, pd.Series]:
    fundamentos = pd.read_parquet(US / "fundamentals.parquet")
    for coluna in ("dt_refer", "available_from"):
        fundamentos[coluna] = pd.to_datetime(fundamentos[coluna])
    valor_mercado = pd.read_parquet(US / "market_caps.parquet")
    valor_mercado["month"] = pd.to_datetime(valor_mercado["month"])
    por_emissor = pd.read_parquet(US / "sectors.parquet").set_index("issuer")["sector"].to_dict()
    setores = pd.read_parquet(US / "securities.parquet").set_index("security")["sector"].dropna().to_dict()
    ptax = pd.read_parquet(US / "ptax_daily.parquet")
    ptax = ptax.set_index(pd.DatetimeIndex(pd.to_datetime(ptax["date"])))["ptax"].sort_index().astype(float)
    return Balancos(fundamentos, valor_mercado, por_emissor), setores, valor_mercado, ptax


def conferir_valor_de_mercado(valor_mercado: pd.DataFrame, mensal: pd.DataFrame, ptax: pd.Series) -> None:
    """The scraped input against B3: the market value's month in reais against the BDR's month on B3."""
    vm = valor_mercado[valor_mercado["issuer"].isin([c[:4] for c in mensal.columns])]
    tabela = vm.pivot(index="month", columns="issuer", values="market_cap").sort_index()
    variacao = tabela.shift(-1) / tabela - 1.0                    # row m: the change over month m
    cambio = ptax.resample("MS").last()
    cambio = (cambio / cambio.shift(1)).reindex(variacao.index)   # end of m over end of m-1
    em_reais = (1.0 + variacao).mul(cambio, axis=0) - 1.0
    b3 = mensal.rename(columns=lambda c: c[:4]).T.groupby(level=0).first().T
    comum = [c for c in em_reais.columns if c in b3.columns]
    gap = (em_reais[comum] - b3[comum].reindex(em_reais.index)).loc[INICIO:FIM]
    absoluto = gap.abs().stack()
    print(f"Market value (scraped US close x SEC shares) vs B3, monthly, {len(comum)} names, {len(absoluto)} months: "
          f"median |gap| {absoluto.median():.2%}, 90th pct {absoluto.quantile(0.9):.2%}; worst names (median |gap|): "
          f"{gap.abs().median().sort_values(ascending=False).head(5).round(4).to_dict()}")


def escolhas(retornos: pd.DataFrame, volumes: pd.DataFrame, nota, setores: dict[str, str],
             datas: pd.DatetimeIndex) -> tuple[dict[pd.Timestamp, list[str]], pd.DataFrame]:
    nivel = bf.indice(retornos)
    calendario = retornos.index
    picks, linhas = {}, []
    for data in datas:
        ate = calendario[calendario.get_loc(data) - 1]
        candidatos = mb.elegiveis(nivel, volumes, ate, presenca=PRESENCA, volume_minimo=VOLUME_MINIMO)
        mes = data.to_period("M").to_timestamp()
        notas = nota(mes, candidatos) if candidatos else pd.Series(dtype=float)
        carteira = carteira_da_nota(notas.dropna(), setores, NOMES, POR_SETOR) if len(notas.dropna()) else pd.Series()
        picks[data] = list(carteira[carteira > 0].index)
        linhas += [{"decisao": data, "papel": p, "nota": float(notas[p]), "elegiveis": len(candidatos),
                    "com_nota": int(notas.notna().sum())} for p in picks[data]]
        if not picks[data]:
            linhas.append({"decisao": data, "papel": "", "nota": np.nan, "elegiveis": len(candidatos),
                           "com_nota": int(notas.notna().sum())})
    return picks, pd.DataFrame(linhas)


def main() -> int:
    mapa = pd.read_csv(MAPA).set_index("bdr")["us_security"]
    retornos, volumes, bova, saltos = carregar_b3(mapa)
    print(f"Jumps zeroed (lake gold/jumps, B3): {[(s, f'{d:%d/%m/%Y}') for s, d in saltos.itertuples(index=False)]}")
    mensal = bf.retornos_mensais(retornos)
    balancos, setores, valor_mercado, ptax = carregar_sec()
    conferir_valor_de_mercado(valor_mercado, mensal, ptax)
    proventos = pd.DataFrame({"security": pd.Series(dtype=str), "ex_date": pd.Series(dtype="datetime64[ns]"),
                              "yield": pd.Series(dtype=float)})
    nota = nota_do_composto(mensal, proventos, balancos)
    sessoes = retornos.index[(retornos.index >= INICIO) & (retornos.index <= FIM)]
    picks, tabela = escolhas(retornos, volumes, nota, setores, mb.decisoes(sessoes))
    tabela.to_csv(SAIDA, index=False, lineterminator="\n")
    por_decisao = tabela.drop_duplicates("decisao")
    tamanhos = pd.Series({d: len(p) for d, p in picks.items()})
    print(f"Window {sessoes[0]:%d/%m/%Y} to {sessoes[-1]:%d/%m/%Y}, {len(picks)} decisions; eligible BDRs min "
          f"{por_decisao['elegiveis'].min()} median {por_decisao['elegiveis'].median():.0f} max "
          f"{por_decisao['elegiveis'].max()}; with a score median {por_decisao['com_nota'].median():.0f}; names held "
          f"min {tamanhos.min()} median {tamanhos.median():.0f}; N for the DSR {N_DSR}")
    print(f"Most held (months): {tabela['papel'].value_counts().head(20).to_dict()}")
    print(f"Last decision {max(picks):%d/%m/%Y}: {picks[max(picks)]}")

    nivel = bf.indice(retornos)
    cdi = pd.read_parquet(CDI)
    rf_todo = pd.Series(cdi["cdi_pct"].to_numpy(float) / 100.0, index=pd.to_datetime(cdi["date"]))
    resumo = []
    for fase in FASES:
        tabela_curvas = pd.read_parquet(str(CURVAS).format(fase))
        composto = pd.Series(tabela_curvas[COMPOSITE].to_numpy(float),
                             index=pd.to_datetime(tabela_curvas["date"])).loc[INICIO:FIM]
        escolhas_fase = {d: p for d, p in picks.items() if d in composto.index}
        curvas = {COMPOSITE: composto / composto.iloc[0]}
        for peso in PESOS:
            curvas[f"Comp{round(100 * (1 - peso))}+CompBDR"] = mb.simular(composto, nivel, escolhas_fase, peso)[0]
        caixa = pd.Series(1.0, index=composto.index)
        curvas[SO_BDR] = mb.simular(caixa, nivel, escolhas_fase, 1.0, custo_composto=0.0)[0]
        b = bova.reindex(composto.index).ffill()
        curvas["BOVA11"] = b / b.iloc[0]
        rf = rf_todo.reindex(composto.index).ffill().fillna(0.0)
        curvas["CDI"] = pd.Series(np.cumprod(1.0 + rf.to_numpy()), index=composto.index)
        resultado = metricas(curvas, rf, N_DSR)
        print(f"\n=== Phase {fase}: {composto.index[0]:%d/%m/%Y} to {composto.index[-1]:%d/%m/%Y}")
        print(resultado[[c for c in COLUNAS if c in resultado.columns]].to_string(float_format=lambda x: f"{x:.3f}"))
        resumo.append(resultado)
    todas = pd.concat(resumo)
    print(f"\nGate (P[> Composite] >= {P_MIN} and DSR >= {DSR_MIN} in every phase):")
    for braco in [f"Comp{round(100 * (1 - p))}+CompBDR" for p in PESOS] + [SO_BDR]:
        linhas = todas.loc[braco]
        ok = bool(((linhas["P[> Comp]"] >= P_MIN) & (linhas["DSR"] >= DSR_MIN)).all())
        print(f"  {braco}: CAGR {' / '.join(f'{c:.1%}' for c in linhas['CAGR'])}, (P, DSR) "
              f"{[(round(p, 3), round(d, 3)) for p, d in zip(linhas['P[> Comp]'], linhas['DSR'])]} -> "
              f"{'CANDIDATE (forward ghost only)' if ok else 'fails'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
