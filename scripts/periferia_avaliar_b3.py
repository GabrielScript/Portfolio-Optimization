"""
periferia_avaliar_b3.py - Periferia-15's one reading on the B3 (docs/superpowers/specs/2026-11-07-periferia-design.md,
section 4): the KAN-15 reading's engine, frozen data and plumbing checks, the DSR at the pre-registered union N = 95.
Runs only after the pre-registration commit (`29a075b`). No network; nothing goes to the app.

    python -X utf8 scripts/periferia_avaliar_b3.py > kan_run/periferia_leitura.txt
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core import market_data  # noqa: E402
from core.backtest import Congelamento, ResultadoBacktest, walk_forward_dinamico  # noqa: E402
from core.cdi_lago import daily_cdi  # noqa: E402
from core.composto import estrategia_por_nota  # noqa: E402
from core.metrics import cagr, drawdown  # noqa: E402
from core.periferia import estrategia_periferia  # noqa: E402
from core.robustez import psr_da_diferenca  # noqa: E402
from core.sinais import estrategia_conservadora  # noqa: E402
from scripts import avaliar_conservadora, avaliar_robustez  # noqa: E402
from scripts.avaliar_robustez import FASES, _acoes, conferir_hashes, contar_tentativas, referencias, tabela_do_periodo  # noqa: E402
from scripts.avaliar_universo_dinamico import ANOS_DE_CDI, CFG_BT  # noqa: E402
from scripts.qml_avaliar_b3 import COMPOSTO, MACRO, conferir_composto  # noqa: E402
from scripts.qpo_avaliar import CONSERVADORA, _year_table, check_aligned, check_plumbing  # noqa: E402
from scripts.qpo_montar import FOLDER, LAST_SESSION  # noqa: E402
from scripts.qubo_universo_avaliar import P_MIN, _value_score, strategy_gate  # noqa: E402

BRACO = "Periferia-15"
N_DSR = 95          # the union of every branch's trials, section 4


def main() -> int:
    inicio = time.perf_counter()
    avaliar_robustez.PASTA_PRECOS = avaliar_robustez.PASTA_UNIVERSO = FOLDER
    dados = avaliar_robustez.carregar()
    mensal = avaliar_conservadora.carregar_mensal(FOLDER)
    nota, sha256_cvm = _value_score(mensal, com_crescimento=True)
    sha256_cdi = conferir_hashes(MACRO, ["cdi_daily.parquet"])
    cdi = daily_cdi(ANOS_DE_CDI, LAST_SESSION.date(), path=MACRO / "cdi_daily.parquet")
    print(f"Files (sha256): {dados.sha256 | mensal.sha256 | sha256_cvm | sha256_cdi}; "
          f"DSR N {N_DSR} (this branch's file counts {contar_tentativas()})")

    p_composto, p_conservadora, p_bova, dsr = [], [], [], []
    for fase in FASES:
        acoes = _acoes(dados.precos.loc[:LAST_SESSION].iloc[fase:])

        def rodar(estrategia, nome) -> ResultadoBacktest:
            return walk_forward_dinamico(acoes, dados.universo, estrategia, CFG_BT, nome, Congelamento())

        conservadora = rodar(estrategia_conservadora(mensal.retornos, mensal.proventos, dados.setores), CONSERVADORA)
        check_plumbing(fase, cagr(conservadora.valores.dropna()))
        composto = rodar(estrategia_por_nota(nota, dados.setores), COMPOSTO)
        conferir_composto(fase, cagr(composto.valores.dropna()))
        braco = rodar(estrategia_periferia(nota, dados.setores), BRACO)
        valores = {BRACO: braco.valores, COMPOSTO: composto.valores, CONSERVADORA: conservadora.valores}
        valores |= {nome: curva.valores for nome, curva in referencias(dados.precos.loc[:LAST_SESSION].iloc[fase:], CFG_BT).items()}
        check_aligned(valores)
        rf = market_data.serie_rf_diaria(valores[COMPOSTO].index, cdi, (1.0 + float(cdi.iloc[-1])) ** 252 - 1.0)
        comparadas = {nome: valores[nome] for nome in (BRACO, COMPOSTO, CONSERVADORA)}
        tabela = tabela_do_periodo(comparadas, valores["BOVA11"], rf, N_DSR)
        for nome in (*comparadas, "BOVA11"):
            tabela.loc[nome, "CAGR"] = cagr(valores[nome].dropna())
            tabela.loc[nome, "MDD"] = drawdown(valores[nome].dropna())[1]
        tabela.loc[BRACO, "P[> Composto]"] = psr_da_diferenca(valores[BRACO], valores[COMPOSTO])
        tabela.loc[BRACO, "P[> Conservadora]"] = psr_da_diferenca(valores[BRACO], valores[CONSERVADORA])
        p_composto.append(float(tabela.loc[BRACO, "P[> Composto]"]))
        p_conservadora.append(float(tabela.loc[BRACO, "P[> Conservadora]"]))
        p_bova.append(float(tabela.loc[BRACO, "P aritmético"]))
        dsr.append(float(tabela.loc[BRACO, "DSR"]))
        datas = valores[COMPOSTO].index
        print(f"\n=== Phase {fase}: {datas[0]:%d/%m/%Y} to {datas[-1]:%d/%m/%Y} ===")
        print(tabela.to_string(float_format=lambda valor: f"{valor:.3f}"))
        for nome in comparadas:
            diarios = valores[nome].dropna().pct_change().dropna()
            print(f"{nome}: annual volatility {diarios.std() * np.sqrt(252):.1%}")
        comuns = [len(set(a[a > 1e-6].index) & set(b[b > 1e-6].index)) / max(int((a > 1e-6).sum()), 1)
                  for (_, a), (_, b) in zip(braco.pesos_alvo.iterrows(), composto.pesos_alvo.iterrows(), strict=False)]
        print(f"picks shared with the Composite, mean over rebalances: {np.mean(comuns):.0%}")
        for nome, resultado in {BRACO: braco, COMPOSTO: composto, CONSERVADORA: conservadora}.items():
            print(f"{nome}: refusals {len(resultado.recusas)}, failures {len(resultado.falhas)}, "
                  f"annual turnover {resultado.turnover_anual_medio:.2f}")
        print(_year_table({**comparadas, "BOVA11": valores["BOVA11"]}).to_string(float_format=lambda valor: f"{valor:.1%}"))
        if fase == FASES[0]:
            ultima = braco.pesos_alvo.iloc[-1]
            print(f"Periferia-15 latest picks ({braco.pesos_alvo.index[-1]:%d/%m/%Y}): "
                  + ", ".join(sorted(ultima[ultima > 1e-6].index)))
        print(f"(phase {fase} done in {time.perf_counter() - inicio:.0f} s)", flush=True)

    passa = strategy_gate([*p_composto, *p_conservadora], p_bova, dsr)
    print(f"\nGate, all >= {P_MIN} in every phase:")
    print(f"{BRACO}: P[> Composite] {np.round(p_composto, 3)}, P[> Conservadora] {np.round(p_conservadora, 3)}, "
          f"P[> BOVA11] {np.round(p_bova, 3)}, DSR {np.round(dsr, 3)}: {'PASSES' if passa else 'does not pass'}")
    print(f"\nDone in {time.perf_counter() - inicio:.0f} s")
    return 0


if __name__ == "__main__":
    pd.set_option("display.width", 200)
    raise SystemExit(main())
