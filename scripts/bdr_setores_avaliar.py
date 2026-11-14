"""
10, 15 and 20 BDRs, one per high-chance sector first, alone and mixed with the Composite at 90/10 to 50/50 (spec
``docs/superpowers/specs/2026-11-14-bdr-setores-design.md``, pre-registered before this ran).

    python -X utf8 scripts/bdr_setores_avaliar.py > docs/bdr-setores-2026-11-14.txt
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

from core import bdr_setores as bs  # noqa: E402
from core import manga_bdr as mb  # noqa: E402
from scripts.xquad_avaliar import COMPOSITE, metricas  # noqa: E402

GOLD = Path("D:/b3-datalake/gold/equity_adjusted.parquet")
CURVAS = RAIZ / "docs" / "dados" / "pos-idade-curvas-fase{}-2026-11-13.parquet"
CDI = RAIZ.parent / "Portfolio-Optimization" / "data" / "b3lake" / "cdi_daily.parquet"   # git-ignored: the main checkout
SAIDA = RAIZ / "docs" / "dados" / "bdr-setores-escolhas-2026-11-14.csv"
LEITURA, ATE = dt.date(2021, 1, 4), dt.date(2026, 11, 13)
ESCOLHA = (pd.Timestamp("2024-01-02"), pd.Timestamp("2026-11-13"))
JANELAS = ((pd.Timestamp("2024-01-02"), pd.Timestamp("2026-10-16"), "2024-2026, read by the pick (in-sample)"),
           (pd.Timestamp("2021-01-04"), pd.Timestamp("2026-10-16"), "2021-2026, names enter when priced"),
           (pd.Timestamp("2021-01-04"), pd.Timestamp("2023-12-29"), "2021-2023, not read by the pick"))
PESOS, FASES = (0.10, 0.20, 0.30, 0.40, 0.50), (0, 21, 42)
N_DSR = 167                             # 149 after DECISOES 366 + 6 rows for each of the 3 sizes (the addendum)
TAMANHOS = (10, 15, 20)
COLUNAS = ["CAGR", "Vol", "Sharpe", "Sortino", "MDD", "Calmar", "P[> Comp]", "DSR", "alpha vs Comp", "t vs Comp"]


def carregar() -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """The candidates' closes and volumes on the B3 calendar (BOVA11's sessions) and BOVA11's total-return close."""
    nomes = sorted({p for setor in bs.SETORES for p in setor.candidatos})
    filtros = [("date", ">=", LEITURA), ("date", "<=", ATE)]
    bdr = pq.read_table(GOLD, columns=["date", "security", "tr_close", "volume_brl"],
                        filters=[*filtros, ("security", "in", nomes)]).to_pandas()
    bova = pq.read_table(GOLD, columns=["date", "tr_close"], filters=[*filtros, ("security", "==", "BOVA11")]).to_pandas()
    calendario = pd.DatetimeIndex(pd.to_datetime(bova["date"])).sort_values()
    bdr["date"] = pd.to_datetime(bdr["date"])
    precos = bdr.pivot(index="date", columns="security", values="tr_close").reindex(calendario)
    volumes = bdr.pivot(index="date", columns="security", values="volume_brl").reindex(calendario)
    return precos, volumes, pd.Series(bova["tr_close"].to_numpy(float), index=calendario)


def curvas_da_janela(composto: pd.Series, precos: pd.DataFrame, papeis: list[str]) -> dict[str, pd.Series]:
    """The Composite, then for each size its five mixes and the names alone, on the Composite's sessions."""
    curvas = {COMPOSITE: composto / composto.iloc[0]}
    caixa = pd.Series(1.0, index=composto.index)
    for k in TAMANHOS:
        escolhas = {d: papeis[:k] for d in mb.decisoes(composto.index)}
        for peso in PESOS:
            curvas[f"Comp{round(100 * (1 - peso))}+{k}BDR"] = mb.simular(composto, precos, escolhas, peso)[0]
        curvas[f"{k}BDR"] = mb.simular(caixa, precos, escolhas, 1.0, custo_composto=0.0)[0]
    return curvas


def main() -> int:
    precos, volumes, bova = carregar()
    escolha = bs.escolher(precos, volumes, bs.SETORES, *ESCOLHA, quantos=max(TAMANHOS))
    escolha.to_csv(SAIDA, index=False, lineterminator="\n")
    papeis = list(escolha["papel"])
    anos = (len(precos.loc[ESCOLHA[0]:ESCOLHA[1]]) - 1) / 252
    tabela = escolha.assign(**{"retorno a.a.": (1 + escolha["retorno"]) ** (1 / anos) - 1,
                               "mediana R$ mil": escolha["mediana"] / 1e3})
    tabela["tamanhos"] = [",".join(str(k) for k in TAMANHOS if i < k) for i in range(len(tabela))]
    print(f"The pick: {ESCOLHA[0]:%d/%m/%Y} to {ESCOLHA[1]:%d/%m/%Y}, {len(papeis)} names; N for the DSR {N_DSR}")
    print(tabela[["tamanhos", "rodada", "ordem", "setor", "nota", "papel", "nivel", "retorno", "retorno a.a.", "mediana R$ mil",
                  "presenca"]].to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    pulados = [s.nome for s in bs.SETORES if s.nome not in set(escolha["setor"])]
    print(f"Sectors skipped (no candidate at R$ 10 thousand and 90% of the sessions): {pulados}")

    cdi = pd.read_parquet(CDI)
    rf_todo = pd.Series(cdi["cdi_pct"].to_numpy(float) / 100.0, index=pd.to_datetime(cdi["date"]))
    for inicio, fim, rotulo in JANELAS:
        for fase in FASES:
            tabela_curvas = pd.read_parquet(str(CURVAS).format(fase))
            composto = pd.Series(tabela_curvas[COMPOSITE].to_numpy(float), index=pd.to_datetime(tabela_curvas["date"]))
            composto = composto.loc[inicio:fim]
            curvas = curvas_da_janela(composto, precos, papeis)
            b = bova.reindex(composto.index).ffill()
            curvas["BOVA11"] = b / b.iloc[0]
            rf = rf_todo.reindex(composto.index).ffill().fillna(0.0)
            curvas["CDI"] = pd.Series(np.cumprod(1.0 + rf.to_numpy()), index=composto.index)
            resultado = metricas(curvas, rf, N_DSR)
            print(f"\n=== {rotulo}, phase {fase}: {composto.index[0]:%d/%m/%Y} to {composto.index[-1]:%d/%m/%Y}")
            print(resultado[[c for c in COLUNAS if c in resultado.columns]].to_string(
                float_format=lambda x: f"{x:.3f}"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
