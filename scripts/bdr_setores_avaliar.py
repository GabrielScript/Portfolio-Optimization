"""
10, 15 and 20 BDRs, one per high-chance sector first, alone and mixed with the Composite at 90/10 to 50/50 (spec
``docs/superpowers/specs/2026-11-14-bdr-setores-design.md``, pre-registered before this ran).

    python -X utf8 scripts/bdr_setores_avaliar.py precos composite > docs/bdr-setores-2026-11-14.txt
    python -X utf8 scripts/bdr_setores_avaliar.py dividendos composite > docs/bdr-setores-dividendos-2026-11-14.txt
    python -X utf8 scripts/bdr_setores_avaliar.py dividendos fermionica > docs/bdr-setores-fermionica-2026-11-14.txt

The picks always read prices only (the spec); ``dividendos`` adds the BDRs' dividends to the backtests (addendum 2).
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

from core import bdr_dividendos as bd  # noqa: E402
from core import bdr_setores as bs  # noqa: E402
from core import manga_bdr as mb  # noqa: E402
from scripts.xquad_avaliar import COMPOSITE, metricas  # noqa: E402

GOLD = Path("D:/b3-datalake/gold/equity_adjusted.parquet")
CAIXA_B3 = Path("D:/b3-datalake/gold/cash.parquet")
EUA = Path("D:/dados_stocks_USA_datalake/prices")
DECODIFICADOR = "D:/sa"                 # the US lake's decoder (decode.decode_auto)
FERMIONICA = "LogDet-15 t0.5"           # the Fermiônica account's rule (DECISOES 348)
FOLGA_B3 = pd.Timedelta(days=7)         # a US ex-date can precede the B3 one by a few days
CURVAS = RAIZ / "docs" / "dados" / "pos-idade-curvas-fase{}-2026-11-13.parquet"
CDI = RAIZ.parent / "Portfolio-Optimization" / "data" / "b3lake" / "cdi_daily.parquet"   # git-ignored: the main checkout
SAIDA = RAIZ / "docs" / "dados" / "bdr-setores-escolhas-2026-11-14.csv"
LEITURA, ATE = dt.date(2021, 1, 4), dt.date(2026, 11, 13)
ESCOLHA = (pd.Timestamp("2024-01-02"), pd.Timestamp("2026-11-13"))
JANELAS = ((pd.Timestamp("2024-01-02"), pd.Timestamp("2026-10-16"), "2024-2026, read by the pick (in-sample)"),
           (pd.Timestamp("2021-01-04"), pd.Timestamp("2026-10-16"), "2021-2026, names enter when priced"),
           (pd.Timestamp("2021-01-04"), pd.Timestamp("2023-12-29"), "2021-2023, not read by the pick"))
PESOS, FASES = (0.10, 0.20, 0.30, 0.40, 0.50), (0, 21, 42)
N_DSR = 182                             # 149 after DECISOES 366 + 18 (sizes) + 15 (Fermionic mixes), addendum 2
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


def com_dividendos(precos: pd.DataFrame) -> pd.DataFrame:
    """``precos`` with each mapped BDR's dividends from its US underlying, before its first B3 cash event; prints what
    was added."""
    sys.path.insert(0, DECODIFICADOR)
    from decode import decode_auto  # noqa: PLC0415 - the US lake's own decoder

    caixa = pd.read_parquet(CAIXA_B3)
    primeiro_b3 = pd.to_datetime(caixa.groupby("security")["last_date_prior_ex"].min())
    ajustados = precos.copy()
    print("Dividends added from the US underlyings (net of 30% withholding; spin-offs in full):")
    for papel, ticker in bd.SUBJACENTE.items():
        if papel not in precos.columns:
            continue
        arquivos = {k: EUA / ticker[0] / f"{ticker}_{k}.json.gz" for k in ("daily", "monthly")}
        tabelas = {k: pd.DataFrame(decode_auto(str(v)) if v.exists() else [], columns=["t", "c", "a"])
                   for k, v in arquivos.items()}
        eventos = bd.eventos(tabelas["daily"], tabelas["monthly"])
        corte = primeiro_b3[papel] - FOLGA_B3 if papel in primeiro_b3.index else None
        serie = precos[papel].dropna()
        nova = bd.aplicar(serie, eventos, corte)
        ajustados.loc[serie.index, papel] = nova
        usados = eventos[eventos.index >= serie.index.min()]
        if corte is not None:
            usados = usados[usados.index < corte]
        cisoes = usados[usados > bd.LIMITE]
        b3 = f"B3 from {corte + FOLGA_B3:%d/%m/%Y}" if corte is not None else "no B3 event"
        extra = f", spin-offs {[(f'{d:%d/%m/%Y}', round(v, 3)) for d, v in cisoes.items()]}" if len(cisoes) else ""
        print(f"  {papel} ({ticker}): {int((usados.abs() > 1e-4).sum())} events, added {(nova / serie).iloc[-1] / (nova / serie).iloc[0] - 1:+.2%} over {serie.index[0]:%m/%Y}-{serie.index[-1]:%m/%Y}"
              f"{extra}; {b3}")
    return ajustados


def curvas_da_janela(base: pd.Series, nome_base: str, rotulo: str, composto: pd.Series, precos: pd.DataFrame,
                     papeis: list[str]) -> dict[str, pd.Series]:
    """The Composite, the base alone (when not the Composite), then for each size the base's five mixes and the
    names alone, on the base's sessions."""
    curvas = {COMPOSITE: composto / composto.iloc[0]}
    if nome_base != COMPOSITE:
        curvas[nome_base] = base / base.iloc[0]
    caixa = pd.Series(1.0, index=base.index)
    for k in TAMANHOS:
        escolhas = {d: papeis[:k] for d in mb.decisoes(base.index)}
        for peso in PESOS:
            curvas[f"{rotulo}{round(100 * (1 - peso))}+{k}BDR"] = mb.simular(base, precos, escolhas, peso)[0]
        curvas[f"{k}BDR"] = mb.simular(caixa, precos, escolhas, 1.0, custo_composto=0.0)[0]
    return curvas


def main(argv: list[str]) -> int:
    dados, base = argv
    nome_base, rotulo = {"composite": (COMPOSITE, "Comp"), "fermionica": (FERMIONICA, "Ferm")}[base]
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
    print(f"Backtests: {dados}, base {nome_base}")
    if dados == "dividendos":
        precos = com_dividendos(precos)

    cdi = pd.read_parquet(CDI)
    rf_todo = pd.Series(cdi["cdi_pct"].to_numpy(float) / 100.0, index=pd.to_datetime(cdi["date"]))
    for inicio, fim, rotulo in JANELAS:
        for fase in FASES:
            tabela_curvas = pd.read_parquet(str(CURVAS).format(fase))
            composto = pd.Series(tabela_curvas[COMPOSITE].to_numpy(float), index=pd.to_datetime(tabela_curvas["date"]))
            composto = composto.loc[inicio:fim]
            base_serie = pd.Series(tabela_curvas[nome_base].to_numpy(float),
                                   index=pd.to_datetime(tabela_curvas["date"])).loc[inicio:fim]
            curvas = curvas_da_janela(base_serie, nome_base, rotulo, composto, precos, papeis)
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
    raise SystemExit(main(sys.argv[1:]))
