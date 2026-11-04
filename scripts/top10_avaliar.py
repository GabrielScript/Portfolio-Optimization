"""
top10_avaliar.py - the one reading of attempt 55 (docs/superpowers/specs/2026-11-04-top10-design.md, committed in
a113617 before this script): the Composite with up to 4 names from the top-10 returns against the Composite itself, on
attempt 51's frozen data and engine, in the 3 phases. No network.

    python -X utf8 scripts/top10_avaliar.py > docs/top10-2026-11-04.txt
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core import market_data  # noqa: E402
from core.backtest import Congelamento, walk_forward_dinamico  # noqa: E402
from core.cdi_lago import daily_cdi  # noqa: E402
from core.composto import estrategia_por_nota  # noqa: E402
from core.metrics import cagr, drawdown  # noqa: E402
from core.robustez import psr_da_diferenca  # noqa: E402
from core.sinais import estrategia_conservadora  # noqa: E402
from core.top10 import estrategia_top10  # noqa: E402
from scripts import avaliar_conservadora, avaliar_robustez  # noqa: E402
from scripts.avaliar_robustez import (FASES, _acoes, conferir_hashes, contar_tentativas, referencias,  # noqa: E402
                                      tabela_do_periodo)
from scripts.avaliar_universo_dinamico import ANOS_DE_CDI, CFG_BT  # noqa: E402
from scripts.composto_crescimento_avaliar import MACRO_FOLDER  # noqa: E402
from scripts.qpo_avaliar import CONSERVADORA, _year_table, check_aligned, check_plumbing  # noqa: E402
from scripts.qpo_montar import FOLDER, LAST_SESSION  # noqa: E402
from scripts.qubo_universo_avaliar import _value_score  # noqa: E402

ARM, COMPOSTO = "Top10", "Composto"
P_MIN = DSR_MIN = 0.95


def gate(p_vs_composto: list[float], dsr: list[float]) -> bool:
    """The pre-registered gate: P[> Composite] and the DSR at least 0.95 in every phase."""
    return all(p >= P_MIN for p in p_vs_composto) and all(d >= DSR_MIN for d in dsr)


def trocas_por_rebalanceio(braco: pd.DataFrame, composto: pd.DataFrame) -> pd.Series:
    """Names in the arm's target that the Composite's target of the same rebalance does not hold."""
    datas = braco.index.intersection(composto.index)
    return pd.Series([int(((braco.loc[data] > 0) & ~(composto.loc[data].reindex(braco.columns).fillna(0) > 0)).sum())
                      for data in datas], index=datas)


def main() -> int:
    inicio = time.perf_counter()
    avaliar_robustez.PASTA_PRECOS = avaliar_robustez.PASTA_UNIVERSO = FOLDER
    dados = avaliar_robustez.carregar()
    mensal = avaliar_conservadora.carregar_mensal(FOLDER)
    nota, sha256_cvm = _value_score(mensal, com_crescimento=True)          # attempt 51's score, same code
    sha256_cdi = conferir_hashes(MACRO_FOLDER, ["cdi_daily.parquet"])
    cdi = daily_cdi(ANOS_DE_CDI, LAST_SESSION.date(), path=MACRO_FOLDER / "cdi_daily.parquet")
    n_tentativas = contar_tentativas()
    print(f"Files (sha256): {dados.sha256 | mensal.sha256 | sha256_cvm | sha256_cdi}; arm {ARM}; DSR N {n_tentativas}")

    p_comp, dsr = [], []
    for fase in FASES:
        acoes = _acoes(dados.precos.loc[:LAST_SESSION].iloc[fase:])

        def rodar(estrategia, nome):
            return walk_forward_dinamico(acoes, dados.universo, estrategia, CFG_BT, nome, Congelamento())

        conservadora = rodar(estrategia_conservadora(mensal.retornos, mensal.proventos, dados.setores), CONSERVADORA)
        check_plumbing(fase, cagr(conservadora.valores.dropna()))          # before the arms: fail fast
        composto = rodar(estrategia_por_nota(nota, dados.setores), COMPOSTO)
        braco = rodar(estrategia_top10(nota, dados.setores), ARM)
        curvas = {ARM: braco, COMPOSTO: composto, CONSERVADORA: conservadora,
                  **referencias(dados.precos.loc[:LAST_SESSION].iloc[fase:], CFG_BT)}
        valores = {nome: curva.valores for nome, curva in curvas.items()}
        check_aligned(valores)
        rf = market_data.serie_rf_diaria(valores[ARM].index, cdi, (1.0 + float(cdi.iloc[-1])) ** 252 - 1.0)
        tabela = tabela_do_periodo({nome: valores[nome] for nome in (ARM, COMPOSTO, CONSERVADORA)}, valores["BOVA11"],
                                   rf, n_tentativas)
        for nome in (ARM, COMPOSTO, CONSERVADORA, "BOVA11"):
            tabela.loc[nome, "CAGR"] = cagr(valores[nome].dropna())
            tabela.loc[nome, "MDD"] = drawdown(valores[nome].dropna())[1]
        tabela.loc[ARM, "P[> Composto]"] = psr_da_diferenca(valores[ARM], valores[COMPOSTO])
        tabela.loc[ARM, "P[> Conservadora]"] = psr_da_diferenca(valores[ARM], valores[CONSERVADORA])
        p_comp.append(float(tabela.loc[ARM, "P[> Composto]"]))
        dsr.append(float(tabela.loc[ARM, "DSR"]))
        trocas = trocas_por_rebalanceio(braco.pesos_alvo, composto.pesos_alvo)
        print(f"\n=== Phase {fase}: {valores[ARM].index[0]:%d/%m/%Y} to {valores[ARM].index[-1]:%d/%m/%Y}; refusals "
              f"{len(braco.recusas)} (Composto {len(composto.recusas)}); annual turnover "
              f"{braco.turnover_anual_medio:.2f} (Composto {composto.turnover_anual_medio:.2f}) ===")
        print(tabela.to_string(float_format=lambda x: f"{x:.3f}"))
        print(f"Names not in the Composite's portfolio per rebalance: mean {trocas.mean():.2f}, "
              f"distribution {trocas.value_counts().sort_index().to_dict()}")
        print(_year_table({nome: valores[nome] for nome in (ARM, COMPOSTO, CONSERVADORA, "BOVA11")})
              .to_string(float_format=lambda x: f"{x:.1%}"))
        print(f"(phase {fase} done in {time.perf_counter() - inicio:.0f} s)", flush=True)

    passa = gate(p_comp, dsr)
    print(f"\nGate: P[> Composto] {[round(p, 3) for p in p_comp]}, DSR {[round(d, 3) for d in dsr]} "
          f"(all >= {P_MIN}): " + ("CANDIDATE: pre-register a forward paper log before any use in the plan" if passa
                                   else "not passed: the rule is closed and the plan stays the Composite"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
