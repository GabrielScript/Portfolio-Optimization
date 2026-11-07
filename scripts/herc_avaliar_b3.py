"""
herc_avaliar_b3.py - HERC-CDaR and HERC-CDaR-15's one reading on the B3 (docs/superpowers/specs/2026-11-07-herc-design.md
and its addendum): the KAN-15 reading's engine, frozen data and plumbing checks, the DSR at the union N = 102. Runs only
after the pre-registration commits (`c25962d`, `04e3c44`). No network; nothing goes to the app.

    python -X utf8 scripts/herc_avaliar_b3.py > kan_run/herc_leitura.txt
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
from core.herc import estrategia_herc_cdar, estrategia_herc_cdar_15  # noqa: E402
from core.metrics import cagr, drawdown  # noqa: E402
from core.robustez import psr_da_diferenca  # noqa: E402
from core.sinais import estrategia_conservadora  # noqa: E402
from scripts import avaliar_conservadora, avaliar_robustez  # noqa: E402
from scripts.avaliar_robustez import FASES, _acoes, conferir_hashes, contar_tentativas, referencias, tabela_do_periodo  # noqa: E402
from scripts.avaliar_universo_dinamico import ANOS_DE_CDI, CFG_BT  # noqa: E402
from scripts.qml_avaliar_b3 import COMPOSTO, MACRO, conferir_composto  # noqa: E402
from scripts.qpo_avaliar import CONSERVADORA, _year_table, check_aligned, check_plumbing  # noqa: E402
from scripts.qpo_montar import FOLDER, LAST_SESSION  # noqa: E402
from scripts.qubo_universo_avaliar import P_MIN, _value_score, strategy_gate  # noqa: E402

BRACOS = ("HERC-CDaR", "HERC-CDaR-15")
N_DSR = 102         # the union of every branch's trials, section 4


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
    medidas = {braco: {"composto": [], "conservadora": [], "bova": [], "dsr": []} for braco in BRACOS}
    for fase in FASES:
        acoes = _acoes(dados.precos.loc[:LAST_SESSION].iloc[fase:])

        def rodar(estrategia, nome) -> ResultadoBacktest:
            return walk_forward_dinamico(acoes, dados.universo, estrategia, CFG_BT, nome, Congelamento())

        conservadora = rodar(estrategia_conservadora(mensal.retornos, mensal.proventos, dados.setores), CONSERVADORA)
        check_plumbing(fase, cagr(conservadora.valores.dropna()))
        composto = rodar(estrategia_por_nota(nota, dados.setores), COMPOSTO)
        conferir_composto(fase, cagr(composto.valores.dropna()))
        resultados = {"HERC-CDaR": rodar(estrategia_herc_cdar(dados.setores), "HERC-CDaR"),
                      "HERC-CDaR-15": rodar(estrategia_herc_cdar_15(nota, dados.setores), "HERC-CDaR-15")}
        valores = {nome: resultado.valores for nome, resultado in resultados.items()}
        valores |= {COMPOSTO: composto.valores, CONSERVADORA: conservadora.valores}
        valores |= {nome: curva.valores for nome, curva in referencias(dados.precos.loc[:LAST_SESSION].iloc[fase:], CFG_BT).items()}
        check_aligned(valores)
        rf = market_data.serie_rf_diaria(valores[COMPOSTO].index, cdi, (1.0 + float(cdi.iloc[-1])) ** 252 - 1.0)
        comparadas = {nome: valores[nome] for nome in (*BRACOS, COMPOSTO, CONSERVADORA)}
        tabela = tabela_do_periodo(comparadas, valores["BOVA11"], rf, N_DSR)
        for nome in (*comparadas, "BOVA11"):
            tabela.loc[nome, "CAGR"] = cagr(valores[nome].dropna())
            tabela.loc[nome, "MDD"] = drawdown(valores[nome].dropna())[1]
            tabela.loc[nome, "vol"] = valores[nome].dropna().pct_change().std() * np.sqrt(252)
        for braco in BRACOS:
            tabela.loc[braco, "P[> Composto]"] = psr_da_diferenca(valores[braco], valores[COMPOSTO])
            tabela.loc[braco, "P[> Conservadora]"] = psr_da_diferenca(valores[braco], valores[CONSERVADORA])
            medidas[braco]["composto"].append(float(tabela.loc[braco, "P[> Composto]"]))
            medidas[braco]["conservadora"].append(float(tabela.loc[braco, "P[> Conservadora]"]))
            medidas[braco]["bova"].append(float(tabela.loc[braco, "P aritmético"]))
            medidas[braco]["dsr"].append(float(tabela.loc[braco, "DSR"]))
        datas = valores[COMPOSTO].index
        print(f"\n=== Phase {fase}: {datas[0]:%d/%m/%Y} to {datas[-1]:%d/%m/%Y} ===")
        print(tabela.to_string(float_format=lambda valor: f"{valor:.3f}"))
        for nome, resultado in {**resultados, COMPOSTO: composto, CONSERVADORA: conservadora}.items():
            print(f"{nome}: refusals {len(resultado.recusas)}, failures {len(resultado.falhas)}, "
                  f"annual turnover {resultado.turnover_anual_medio:.2f}")
        if fase == FASES[0]:
            print(_year_table({**comparadas, "BOVA11": valores["BOVA11"]}).to_string(float_format=lambda v: f"{v:.1%}"))
            for braco, resultado in resultados.items():
                ultima = resultado.pesos_alvo.iloc[-1]
                ultima = ultima[ultima > 1e-6].sort_values(ascending=False)
                print(f"{braco} latest ({resultado.pesos_alvo.index[-1]:%d/%m/%Y}): {len(ultima)} names, largest "
                      + ", ".join(f"{papel} {peso:.1%}" for papel, peso in ultima.head(5).items()))
        print(f"(phase {fase} done in {time.perf_counter() - inicio:.0f} s)", flush=True)

    print(f"\nGate, all >= {P_MIN} in every phase:")
    for braco in BRACOS:
        m = medidas[braco]
        passa = strategy_gate([*m["composto"], *m["conservadora"]], m["bova"], m["dsr"])
        print(f"{braco}: P[> Composite] {np.round(m['composto'], 3)}, P[> Conservadora] {np.round(m['conservadora'], 3)}, "
              f"P[> BOVA11] {np.round(m['bova'], 3)}, DSR {np.round(m['dsr'], 3)}: {'PASSES' if passa else 'does not pass'}")
    print(f"\nDone in {time.perf_counter() - inicio:.0f} s")
    return 0


if __name__ == "__main__":
    pd.set_option("display.width", 220)
    raise SystemExit(main())
