"""
The Composite with a BDR sleeve, 95/5 to 80/20 and top-1 to top-10 (spec
``docs/superpowers/specs/2026-11-14-composto-bdr-design.md``, pre-registered before this ran).

    python -X utf8 scripts/composto_bdr_avaliar.py > docs/composto-bdr-2026-11-14.txt
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

from core import manga_bdr as mb  # noqa: E402
from scripts.xquad_avaliar import COMPOSITE, DSR_MIN, P_MIN, metricas  # noqa: E402

GOLD = Path("D:/b3-datalake/gold/equity_adjusted.parquet")
CURVAS = RAIZ / "docs" / "dados" / "pos-idade-curvas-fase{}-2026-11-13.parquet"
CDI = RAIZ / "data" / "b3lake" / "cdi_daily.parquet"
SAIDA = RAIZ / "docs" / "dados" / "composto-bdr-escolhas-2026-11-14.csv"
INICIO, FIM = pd.Timestamp("2021-01-04"), pd.Timestamp("2026-10-16")
LEITURA = dt.date(2019, 11, 1)          # 252 sessions before the first decision
PESOS, TAMANHOS, FASES = (0.05, 0.10, 0.15, 0.20), (1, 3, 5, 10), (0, 21, 42)
N_DSR = 149                             # the spec's addendum: DECISOES' 133 + these 16 rows


def nome(peso: float, k: int) -> str:
    return f"Comp{round(100 * (1 - peso))}+BDR-top{k}"


def carregar() -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """The BDRs' closes and volumes on the B3 calendar (BOVA11's sessions) and BOVA11's total-return close."""
    filtros = [("date", ">=", LEITURA), ("date", "<=", FIM.date())]
    bdr = pq.read_table(GOLD, columns=["date", "security", "tr_close", "volume_brl"],
                        filters=[*filtros, ("asset_type", "==", "bdr")]).to_pandas()
    bova = pq.read_table(GOLD, columns=["date", "tr_close"], filters=[*filtros, ("security", "==", "BOVA11")]).to_pandas()
    calendario = pd.DatetimeIndex(pd.to_datetime(bova["date"])).sort_values()
    bdr["date"] = pd.to_datetime(bdr["date"])
    precos = bdr.pivot(index="date", columns="security", values="tr_close").reindex(calendario)
    volumes = bdr.pivot(index="date", columns="security", values="volume_brl").reindex(calendario)
    return precos, volumes, pd.Series(bova["tr_close"].to_numpy(float), index=calendario)


def escolhas_do_mes(precos: pd.DataFrame, volumes: pd.DataFrame, datas: pd.DatetimeIndex) -> pd.DataFrame:
    """One row per (decision, k, rank): the sleeve's names, with each decision's count of eligible BDRs."""
    linhas = []
    for data in datas:
        candidatos = mb.elegiveis(precos, volumes, data)
        notas = mb.nota(precos[candidatos], data) if candidatos else pd.Series(dtype=float)
        for k in TAMANHOS:
            for posicao, papel in enumerate(mb.top(notas, candidatos, k), start=1):
                linhas.append({"decisao": data, "k": k, "posicao": posicao, "papel": papel,
                               "nota": float(notas[papel]), "elegiveis": len(candidatos)})
    return pd.DataFrame(linhas)


def main() -> int:
    precos, volumes, bova = carregar()
    cdi = pd.read_parquet(CDI)
    rf_todo = pd.Series(cdi["cdi_pct"].to_numpy(float) / 100.0, index=pd.to_datetime(cdi["date"]))
    sessoes = precos.index[(precos.index >= INICIO) & (precos.index <= FIM)]
    datas = mb.decisoes(sessoes)
    escolhas = escolhas_do_mes(precos, volumes, datas)
    escolhas.to_csv(SAIDA, index=False, lineterminator="\n")
    por_k = {k: {d: list(g.sort_values("posicao")["papel"]) for d, g in escolhas[escolhas["k"] == k].groupby("decisao")}
             for k in TAMANHOS}
    for k in TAMANHOS:          # a decision with no eligible BDR still resets the book (all Composite)
        por_k[k] |= {d: [] for d in datas if d not in por_k[k]}
    elegiveis = escolhas.drop_duplicates("decisao").set_index("decisao")["elegiveis"]
    print(f"Window {sessoes[0]:%d/%m/%Y} to {sessoes[-1]:%d/%m/%Y}, {len(sessoes)} sessions, {len(datas)} monthly "
          f"decisions; eligible BDRs per decision: min {elegiveis.min()}, median {elegiveis.median():.0f}, "
          f"max {elegiveis.max()}; N for the DSR {N_DSR}")

    resumo = []
    for fase in FASES:
        tabela_curvas = pd.read_parquet(str(CURVAS).format(fase))
        composto = pd.Series(tabela_curvas[COMPOSITE].to_numpy(float), index=pd.to_datetime(tabela_curvas["date"]))
        composto = composto.loc[INICIO:FIM]
        indice = composto.index
        curvas = {COMPOSITE: composto / composto.iloc[0]}
        brutas = {}
        for peso in PESOS:
            for k in TAMANHOS:
                liquido, bruto = mb.simular(composto, precos, por_k[k], peso)
                curvas[nome(peso, k)], brutas[nome(peso, k)] = liquido, bruto
        b = bova.reindex(indice).ffill()
        curvas["BOVA11"] = b / b.iloc[0]
        rf = rf_todo.reindex(indice).ffill().fillna(0.0)
        curvas["CDI"] = pd.Series(np.cumprod(1.0 + rf.to_numpy()), index=indice)
        tabela = metricas(curvas, rf, N_DSR)
        tabela["CAGR pre-tax"] = pd.Series({n: (v.iloc[-1] / v.iloc[0]) ** (252 / (len(v) - 1)) - 1
                                            for n, v in brutas.items()})
        colunas = ["CAGR", "CAGR pre-tax", "Vol", "Sharpe", "Sortino", "MDD", "Calmar", "P[> Comp]", "DSR",
                   "alpha vs Comp", "t vs Comp"]
        print(f"\n=== Phase {fase}: {indice[0]:%d/%m/%Y} to {indice[-1]:%d/%m/%Y}")
        print(tabela[[c for c in colunas if c in tabela.columns]].to_string(float_format=lambda x: f"{x:.3f}"))
        resumo.append(tabela.assign(fase=fase))

    todas = pd.concat(resumo)
    print(f"\nGate per arm, net of the BDR tax (P[> Composite] >= {P_MIN} and DSR >= {DSR_MIN} in every phase):")
    passaram = []
    for braco in [nome(p, k) for p in PESOS for k in TAMANHOS]:
        linhas = todas.loc[braco]
        ok = bool(((linhas["P[> Comp]"] >= P_MIN) & (linhas["DSR"] >= DSR_MIN)).all())
        passaram += [braco] if ok else []
        leituras = [(round(p, 3), round(d, 3)) for p, d in zip(linhas["P[> Comp]"], linhas["DSR"])]
        print(f"  {braco}: CAGR {' / '.join(f'{c:.1%}' for c in linhas['CAGR'])}, (P, DSR) {leituras} -> "
              f"{'CANDIDATE (forward ghost only)' if ok else 'fails'}")
    print(f"\nArms passing the gate: {passaram or 'none'}")
    ultima = escolhas[escolhas["decisao"] == escolhas["decisao"].max()]
    print(f"\nLast decision {ultima['decisao'].iloc[0]:%d/%m/%Y}, top-10: {list(ultima[ultima['k'] == 10]['papel'])}")
    contagem = escolhas[escolhas["k"] == 10].groupby("papel").size().sort_values(ascending=False).head(15)
    print(f"Most chosen in the top-10 (months): {contagem.to_dict()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
