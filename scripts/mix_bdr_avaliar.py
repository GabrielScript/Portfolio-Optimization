"""
mix_bdr_avaliar.py - The fixed BR/BDR mixes' one reading (docs/superpowers/specs/2026-11-07-mix-bdr-design.md):
BDR-1 to BDR-5 on 2021-01 to 16/10/2026, phases 0/21/42, the DSR at the pre-registered union N = 100. Runs only after
the pre-registration commit (`a7f62f0`). Reads the lake's gold table live (hashed); nothing goes to the app.

    python -X utf8 scripts/mix_bdr_avaliar.py > kan_run/mix_bdr_leitura.txt
"""

from __future__ import annotations

import hashlib
import sys
import time
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core import market_data  # noqa: E402
from core.backtest import Congelamento, ResultadoBacktest, walk_forward_dinamico  # noqa: E402
from core.cdi_lago import daily_cdi  # noqa: E402
from core.composto import estrategia_por_nota  # noqa: E402
from core.metrics import cagr, drawdown  # noqa: E402
from dataclasses import replace  # noqa: E402
from core.risco import LIMITES  # noqa: E402
from core.mix_bdr import NOMES, curva_mista, escolher_eua  # noqa: E402
from core.robustez import psr_da_diferenca  # noqa: E402
from core.sinais import estrategia_conservadora  # noqa: E402
from scripts import avaliar_conservadora, avaliar_robustez  # noqa: E402
from scripts.avaliar_robustez import FASES, _acoes, conferir_hashes, referencias, tabela_do_periodo  # noqa: E402
from scripts.avaliar_universo_dinamico import ANOS_DE_CDI, CFG_BT  # noqa: E402
from scripts.qml_avaliar_b3 import COMPOSTO, MACRO, conferir_composto  # noqa: E402
from scripts.qpo_avaliar import CONSERVADORA, _year_table, check_plumbing  # noqa: E402
from scripts.qpo_montar import FOLDER, LAST_SESSION  # noqa: E402
from scripts.qubo_universo_avaliar import P_MIN, _value_score, strategy_gate  # noqa: E402

GOLD = Path("D:/b3-datalake/gold/equity_adjusted.parquet")
INICIO_DA_LEITURA = pd.Timestamp("2021-01-04")
INICIO_DOS_DADOS = date(2019, 1, 1)
N_DSR = 100
KS = (1, 2, 3, 4, 5)
CUSTO_BPS = 10.0
PRECOS_RUINS = {("AVGO34", "2021-09-17")}
EUA = ["NVDC34", "MUTC34", "GOGL34", "M2ST34", "MSFT34", "M1TA34", "AAPL34", "ITLC34", "A1MD34", "ORCL34", "NFLX34",
       "P2LT34", "AVGO34", "C2OI34", "W1DC34", "S1TX34", "D1EL34", "M2RV34", "AMZO34", "TSLA34", "COCA34", "DISB34",
       "NIKE34", "BERK34", "JPMC34", "GSGI34", "VISA34", "LILY34", "FCXO34", "EXXO34", "IVVB11", "NASD11"]


def ler_eua() -> tuple[pd.DataFrame, pd.DataFrame, str]:
    tabela = pq.read_table(GOLD, columns=["date", "security", "tr_close", "volume_brl"],
                           filters=[("security", "in", EUA), ("date", ">=", INICIO_DOS_DADOS),
                                    ("date", "<=", LAST_SESSION.date())]).to_pandas()
    tabela["date"] = pd.to_datetime(tabela["date"])
    for papel, dia in PRECOS_RUINS:
        tabela = tabela[~((tabela["security"] == papel) & (tabela["date"] == pd.Timestamp(dia)))]
    tabela = tabela.sort_values(["security", "date"]).reset_index(drop=True)
    resumo = hashlib.sha256(pd.util.hash_pandas_object(tabela, index=False).to_numpy().tobytes()).hexdigest()
    precos = tabela.pivot(index="date", columns="security", values="tr_close").sort_index()
    volume = tabela.pivot(index="date", columns="security", values="volume_brl").reindex(precos.index)
    return precos, volume, resumo


def _recorte(valores: pd.Series, inicio: pd.Timestamp) -> pd.Series:
    serie = valores.dropna().loc[inicio:]
    return serie / serie.iloc[0] * 10_000.0


def main() -> int:
    inicio = time.perf_counter()
    avaliar_robustez.PASTA_PRECOS = avaliar_robustez.PASTA_UNIVERSO = FOLDER
    dados = avaliar_robustez.carregar()
    mensal = avaliar_conservadora.carregar_mensal(FOLDER)
    nota, sha256_cvm = _value_score(mensal, com_crescimento=True)
    sha256_cdi = conferir_hashes(MACRO, ["cdi_daily.parquet"])
    cdi = daily_cdi(ANOS_DE_CDI, LAST_SESSION.date(), path=MACRO / "cdi_daily.parquet")
    precos_eua, volume_eua, sha256_eua = ler_eua()
    print(f"Files (sha256): {dados.sha256 | mensal.sha256 | sha256_cvm | sha256_cdi}; US slice {sha256_eua}; "
          f"US names in the lake {precos_eua.shape[1]} of {len(EUA)}; DSR N {N_DSR}")
    bracos = [f"BDR-{k}" for k in KS]
    p_composto = {b: [] for b in bracos}
    p_conservadora = {b: [] for b in bracos}
    p_bova = {b: [] for b in bracos}
    dsr = {b: [] for b in bracos}
    for fase in FASES:
        acoes = _acoes(dados.precos.loc[:LAST_SESSION].iloc[fase:])

        def rodar(estrategia, nome) -> ResultadoBacktest:
            return walk_forward_dinamico(acoes, dados.universo, estrategia, CFG_BT, nome, Congelamento())

        conservadora = rodar(estrategia_conservadora(mensal.retornos, mensal.proventos, dados.setores), CONSERVADORA)
        check_plumbing(fase, cagr(conservadora.valores.dropna()))
        composto = rodar(estrategia_por_nota(nota, dados.setores), COMPOSTO)
        conferir_composto(fase, cagr(composto.valores.dropna()))
        refs = referencias(dados.precos.loc[:LAST_SESSION].iloc[fase:], CFG_BT)
        mistas, picks_de_hoje, poucos = {}, {}, {}
        for k in KS:
            escala = NOMES / (NOMES - k)      # the whole portfolio's caps on the sleeve's share (spec addendum)
            limites = replace(LIMITES, teto_setor=LIMITES.teto_setor * escala,
                              peso_maximo_acao=LIMITES.peso_maximo_acao * escala,
                              peso_maximo_emissor=LIMITES.peso_maximo_emissor * escala)
            br = rodar(estrategia_por_nota(nota, dados.setores, nomes=NOMES - k, limites=limites), f"BR-{NOMES - k}")
            datas = [d for d in br.pesos_alvo.index if d >= INICIO_DA_LEITURA]
            escolhas = {d: escolher_eua(precos_eua, volume_eua, d, k) for d in datas}
            poucos[k] = sum(len(v) < k for v in escolhas.values())
            picks_de_hoje[k] = escolhas[datas[-1]]
            mistas[f"BDR-{k}"] = curva_mista(br.valores.dropna(), precos_eua, escolhas, k, CUSTO_BPS)
        comeco = mistas["BDR-1"].index[0]
        valores = {nome: _recorte(curva, comeco) for nome, curva in mistas.items()}
        valores |= {COMPOSTO: _recorte(composto.valores, comeco), CONSERVADORA: _recorte(conservadora.valores, comeco),
                    "BOVA11": _recorte(refs["BOVA11"].valores, comeco)}
        indice = valores[COMPOSTO].index
        valores["IVVB11"] = _recorte(precos_eua["IVVB11"].reindex(indice).ffill(), comeco)
        valores = {nome: serie.reindex(indice) for nome, serie in valores.items()}
        rf = market_data.serie_rf_diaria(indice, cdi, (1.0 + float(cdi.iloc[-1])) ** 252 - 1.0)
        comparadas = {nome: valores[nome] for nome in (*bracos, COMPOSTO, CONSERVADORA)}
        tabela = tabela_do_periodo(comparadas, valores["BOVA11"], rf, N_DSR)
        for nome in (*comparadas, "BOVA11", "IVVB11"):
            tabela.loc[nome, "CAGR"] = cagr(valores[nome].dropna())
            tabela.loc[nome, "MDD"] = drawdown(valores[nome].dropna())[1]
            tabela.loc[nome, "vol"] = valores[nome].pct_change().std() * np.sqrt(252)
        for nome in bracos:
            tabela.loc[nome, "P[> Composto]"] = psr_da_diferenca(valores[nome], valores[COMPOSTO])
            tabela.loc[nome, "P[> Conservadora]"] = psr_da_diferenca(valores[nome], valores[CONSERVADORA])
            p_composto[nome].append(float(tabela.loc[nome, "P[> Composto]"]))
            p_conservadora[nome].append(float(tabela.loc[nome, "P[> Conservadora]"]))
            p_bova[nome].append(float(tabela.loc[nome, "P aritmético"]))
            dsr[nome].append(float(tabela.loc[nome, "DSR"]))
        print(f"\n=== Phase {fase}: {indice[0]:%d/%m/%Y} to {indice[-1]:%d/%m/%Y} ===")
        print(tabela.to_string(float_format=lambda valor: f"{valor:.3f}"))
        print("rebalances with fewer eligible US names than k: " + ", ".join(f"k={k} {n}" for k, n in poucos.items()))
        if fase == FASES[0]:
            print(_year_table({**comparadas, "BOVA11": valores["BOVA11"], "IVVB11": valores["IVVB11"]})
                  .to_string(float_format=lambda valor: f"{valor:.1%}"))
            for k, picks in picks_de_hoje.items():
                print(f"BDR-{k} US picks at the last rebalance: {', '.join(picks)}")
        print(f"(phase {fase} done in {time.perf_counter() - inicio:.0f} s)", flush=True)

    print(f"\nGate, all >= {P_MIN} in every phase:")
    for nome in bracos:
        passa = strategy_gate([*p_composto[nome], *p_conservadora[nome]], p_bova[nome], dsr[nome])
        print(f"{nome}: P[> Composite] {np.round(p_composto[nome], 3)}, P[> Conservadora] {np.round(p_conservadora[nome], 3)}, "
              f"P[> BOVA11] {np.round(p_bova[nome], 3)}, DSR {np.round(dsr[nome], 3)}: {'PASSES' if passa else 'does not pass'}")
    print(f"\nDone in {time.perf_counter() - inicio:.0f} s")
    return 0


if __name__ == "__main__":
    pd.set_option("display.width", 220)
    raise SystemExit(main())
