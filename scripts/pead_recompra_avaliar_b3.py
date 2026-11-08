"""
pead_recompra_avaliar_b3.py - Stage B, the arms' one reading on the B3 (docs/superpowers/specs/
2026-11-08-pead-recompra-design.md, section 6): the periphery/MARS readings' engine, frozen data and plumbing checks,
the DSR at the pre-registered union N = 108. Only an arm whose signal passed stage A is run (argument "sue" and/or
"recompra"). No network; nothing goes to the app.

    python -X utf8 scripts/pead_recompra_avaliar_b3.py sue recompra > docs/pead-recompra-2026-11-08.txt
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
from core.pead import nota_sue  # noqa: E402
from core.qubo_universo import composite_score  # noqa: E402
from core.recompra import estrategia_recompra, eventos  # noqa: E402
from core.robustez import psr_da_diferenca  # noqa: E402
from core.sinais import conservadora, estrategia_conservadora  # noqa: E402
from core.views import nota_media_dos_escores  # noqa: E402
from core.crescimento_lucro import nota_de_crescimento  # noqa: E402
from scripts import avaliar_conservadora, avaliar_robustez, avaliar_valor  # noqa: E402
from scripts.avaliar_robustez import FASES, _acoes, conferir_hashes, contar_tentativas, referencias, tabela_do_periodo  # noqa: E402
from scripts.avaliar_universo_dinamico import ANOS_DE_CDI, CFG_BT  # noqa: E402
from scripts.qml_avaliar_b3 import COMPOSTO, MACRO, conferir_composto  # noqa: E402
from scripts.qpo_avaliar import CONSERVADORA, _year_table, check_aligned, check_plumbing  # noqa: E402
from scripts.qpo_montar import FOLDER, LAST_SESSION  # noqa: E402
from scripts.qubo_universo_avaliar import P_MIN, _value_score, strategy_gate  # noqa: E402

N_DSR = 108         # the union of every branch's trials, section 6
PASTA_CVM = FOLDER.parent / "2026-10-24-cvm"
PASTA_IPE = FOLDER.parent / "2026-11-08-ipe"
SUE_ARM = "Composto+SUE-15"
RECOMPRA_ARM = "Composto+Recompra-15"


def estrategias_dos_bracos(argumentos: list[str], dados, mensal) -> dict:
    balancos, _ = avaliar_valor.carregar_balancos(PASTA_CVM)
    bracos = {}
    if "sue" in argumentos:
        conservadora_do_mes = lambda mes, candidatos: conservadora(mensal.retornos, mensal.proventos, mes, candidatos)  # noqa: E731
        valor = nota_media_dos_escores(*avaliar_valor.notas(balancos, mensal).values())
        nota = composite_score(conservadora_do_mes, valor, nota_de_crescimento(balancos.fundamentos),
                               nota_sue(balancos.fundamentos))
        bracos[SUE_ARM] = estrategia_por_nota(nota, dados.setores)
    if "recompra" in argumentos:
        conferir_hashes(PASTA_IPE, ["recompras.parquet"])
        linhas = pd.read_parquet(PASTA_IPE / "recompras.parquet").dropna(subset=["issuer"])
        composto, _ = _value_score(mensal, com_crescimento=True)
        bracos[RECOMPRA_ARM] = estrategia_recompra(composto, eventos(linhas, dados.precos.loc[:LAST_SESSION].index),
                                                   dados.setores)
    return bracos


def main(argumentos: list[str]) -> int:
    inicio = time.perf_counter()
    avaliar_robustez.PASTA_PRECOS = avaliar_robustez.PASTA_UNIVERSO = avaliar_valor.PASTA_UNIVERSO = FOLDER
    dados = avaliar_robustez.carregar()
    mensal = avaliar_conservadora.carregar_mensal(FOLDER)
    nota, sha256_cvm = _value_score(mensal, com_crescimento=True)
    sha256_cdi = conferir_hashes(MACRO, ["cdi_daily.parquet"])
    cdi = daily_cdi(ANOS_DE_CDI, LAST_SESSION.date(), path=MACRO / "cdi_daily.parquet")
    bracos = estrategias_dos_bracos(argumentos, dados, mensal)
    if not bracos:
        raise SystemExit("no arm passed stage A: nothing to run")
    print(f"Files (sha256): {dados.sha256 | mensal.sha256 | sha256_cvm | sha256_cdi}; arms {list(bracos)}; "
          f"DSR N {N_DSR} (this branch's file counts {contar_tentativas()})")

    portao = {nome: {"composto": [], "conservadora": [], "bova": [], "dsr": []} for nome in bracos}
    for fase in FASES:
        acoes = _acoes(dados.precos.loc[:LAST_SESSION].iloc[fase:])

        def rodar(estrategia, nome) -> ResultadoBacktest:
            return walk_forward_dinamico(acoes, dados.universo, estrategia, CFG_BT, nome, Congelamento())

        conservadora_ = rodar(estrategia_conservadora(mensal.retornos, mensal.proventos, dados.setores), CONSERVADORA)
        check_plumbing(fase, cagr(conservadora_.valores.dropna()))
        composto = rodar(estrategia_por_nota(nota, dados.setores), COMPOSTO)
        conferir_composto(fase, cagr(composto.valores.dropna()))
        resultados = {nome: rodar(estrategia, nome) for nome, estrategia in bracos.items()}
        valores = {nome: r.valores for nome, r in resultados.items()} | {COMPOSTO: composto.valores,
                                                                          CONSERVADORA: conservadora_.valores}
        valores |= {nome: curva.valores
                    for nome, curva in referencias(dados.precos.loc[:LAST_SESSION].iloc[fase:], CFG_BT).items()}
        check_aligned(valores)
        rf = market_data.serie_rf_diaria(valores[COMPOSTO].index, cdi, (1.0 + float(cdi.iloc[-1])) ** 252 - 1.0)
        comparadas = {nome: valores[nome] for nome in (*bracos, COMPOSTO, CONSERVADORA)}
        tabela = tabela_do_periodo(comparadas, valores["BOVA11"], rf, N_DSR)
        for nome in (*comparadas, "BOVA11"):
            tabela.loc[nome, "CAGR"] = cagr(valores[nome].dropna())
            tabela.loc[nome, "MDD"] = drawdown(valores[nome].dropna())[1]
        for nome in bracos:
            tabela.loc[nome, "P[> Composto]"] = psr_da_diferenca(valores[nome], valores[COMPOSTO])
            tabela.loc[nome, "P[> Conservadora]"] = psr_da_diferenca(valores[nome], valores[CONSERVADORA])
            portao[nome]["composto"].append(float(tabela.loc[nome, "P[> Composto]"]))
            portao[nome]["conservadora"].append(float(tabela.loc[nome, "P[> Conservadora]"]))
            portao[nome]["bova"].append(float(tabela.loc[nome, "P aritmético"]))
            portao[nome]["dsr"].append(float(tabela.loc[nome, "DSR"]))
        datas = valores[COMPOSTO].index
        print(f"\n=== Phase {fase}: {datas[0]:%d/%m/%Y} to {datas[-1]:%d/%m/%Y} ===")
        print(tabela.to_string(float_format=lambda valor: f"{valor:.3f}"))
        for nome in comparadas:
            diarios = valores[nome].dropna().pct_change().dropna()
            print(f"{nome}: annual volatility {diarios.std() * np.sqrt(252):.1%}")
        for nome, resultado in resultados.items():
            comuns = [len(set(a[a > 1e-6].index) & set(b[b > 1e-6].index)) / max(int((a > 1e-6).sum()), 1)
                      for (_, a), (_, b) in zip(resultado.pesos_alvo.iterrows(), composto.pesos_alvo.iterrows(),
                                                strict=False)]
            print(f"{nome}: picks shared with the Composite, mean over rebalances {np.mean(comuns):.0%}")
        for nome, resultado in {**resultados, COMPOSTO: composto, CONSERVADORA: conservadora_}.items():
            print(f"{nome}: refusals {len(resultado.recusas)}, failures {len(resultado.falhas)}, "
                  f"annual turnover {resultado.turnover_anual_medio:.2f}")
        print(_year_table({**comparadas, "BOVA11": valores["BOVA11"]}).to_string(float_format=lambda v: f"{v:.1%}"))
        if fase == FASES[0]:
            for nome, resultado in resultados.items():
                ultima = resultado.pesos_alvo.iloc[-1]
                print(f"{nome} latest picks ({resultado.pesos_alvo.index[-1]:%d/%m/%Y}): "
                      + ", ".join(sorted(ultima[ultima > 1e-6].index)))
        print(f"(phase {fase} done in {time.perf_counter() - inicio:.0f} s)", flush=True)

    print(f"\nGate, all >= {P_MIN} in every phase:")
    for nome, p in portao.items():
        passa = strategy_gate([*p["composto"], *p["conservadora"]], p["bova"], p["dsr"])
        print(f"{nome}: P[> Composite] {np.round(p['composto'], 3)}, P[> Conservadora] {np.round(p['conservadora'], 3)}, "
              f"P[> BOVA11] {np.round(p['bova'], 3)}, DSR {np.round(p['dsr'], 3)}: "
              f"{'PASSES' if passa else 'does not pass'}")
    print(f"\nDone in {time.perf_counter() - inicio:.0f} s")
    return 0


if __name__ == "__main__":
    pd.set_option("display.width", 200)
    raise SystemExit(main(sys.argv[1:]))
