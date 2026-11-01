"""
lote2_b3.py - The one reading of the book's second batch on the B3 (docs/superpowers/specs/2026-11-01-lote2-livro-design.md, section 2).
Runs only after the pre-registration commit (`5e5c376`). No network: everything frozen.

    python -X utf8 scripts/lote2_b3.py > D:/b3-datalake/exports/lote2/b3/leitura.txt
    python -X utf8 scripts/lote2_b3.py --so-fase 21     # one worker per phase, then the plain run reads what they saved

Each arm's result is saved per phase, so a crash resumes where it stopped. Writes docs/dados/lote2-{anos,portao}-b3.parquet.
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
from core.estrategias_lote2 import bracos_lote2  # noqa: E402
from core.estrategias_todas import Construtor  # noqa: E402
from core.metrics import cagr, drawdown  # noqa: E402
from core.periodos import anos_em_reais  # noqa: E402
from core.robustez import psr_da_diferenca  # noqa: E402
from core.sinais import estrategia_conservadora  # noqa: E402
from scripts import avaliar_conservadora, avaliar_robustez  # noqa: E402
from scripts.avaliar_robustez import FASES, _acoes, conferir_hashes, contar_tentativas, referencias, tabela_do_periodo  # noqa: E402
from scripts.avaliar_universo_dinamico import ANOS_DE_CDI, CFG_BT  # noqa: E402
from scripts.qml_avaliar_b3 import COMPOSTO, JUROS, MACRO, SAIDA, conferir_composto  # noqa: E402
from scripts.qpo_avaliar import CONSERVADORA, _year_table, check_aligned, check_plumbing  # noqa: E402
from scripts.qpo_montar import FOLDER, LAST_SESSION  # noqa: E402
from scripts.qubo_universo_avaliar import P_MIN, _value_score, strategy_gate  # noqa: E402
from scripts.todas_comum import esperar_a_janela_passar, resultado_salvo  # noqa: E402

PASTA = Path("D:/b3-datalake/exports/lote2/b3")
DIAS_DO_ANO = 252


def fase_do_trabalhador(argv: list[str]) -> int | None:
    """``--so-fase N`` runs only that phase's arms and prints no table."""
    return int(argv[argv.index("--so-fase") + 1]) if "--so-fase" in argv else None


def resultado_do_braco(nome: str, construir: Construtor, fase: int, acoes: pd.DataFrame, dados: avaliar_robustez.Dados,
                       na_janela: bool = False) -> ResultadoBacktest:
    def rodar() -> ResultadoBacktest:
        if not na_janela:
            esperar_a_janela_passar()
        inicio = time.perf_counter()
        resultado = walk_forward_dinamico(acoes, dados.universo, construir(dados.setores, DIAS_DO_ANO), CFG_BT, nome, Congelamento())
        print(f"{nome} phase {fase}: {time.perf_counter() - inicio:.0f} s; first failures {resultado.falhas[:2]}", flush=True)
        return resultado
    return resultado_salvo(PASTA / f"{nome}-fase{fase}.pkl", rodar)


def trabalhador(fase: int, dados: avaliar_robustez.Dados, construtores: dict[str, Construtor], ao_contrario: bool) -> None:
    """Saves that phase's arms. About 500 MB, measured, so it may run through the nightly window: no gold, no ClickHouse."""
    acoes = _acoes(dados.precos.loc[:LAST_SESSION].iloc[fase:])
    ordem = list(construtores.items())
    for nome, construir in reversed(ordem) if ao_contrario else ordem:      # two workers on a phase meet in the middle
        resultado_do_braco(nome, construir, fase, acoes, dados, na_janela=True)


def main(argv: list[str]) -> int:
    inicio = time.perf_counter()
    PASTA.mkdir(parents=True, exist_ok=True)
    avaliar_robustez.PASTA_PRECOS = avaliar_robustez.PASTA_UNIVERSO = FOLDER
    dados = avaliar_robustez.carregar()
    mensal = avaliar_conservadora.carregar_mensal(FOLDER)
    nota, sha256_cvm = _value_score(mensal, com_crescimento=True)
    sha256_cdi = conferir_hashes(MACRO, ["cdi_daily.parquet"])
    cdi = daily_cdi(ANOS_DE_CDI, LAST_SESSION.date(), path=MACRO / "cdi_daily.parquet")
    n_tentativas = contar_tentativas()
    print(f"Files (sha256): {dados.sha256 | mensal.sha256 | sha256_cvm | sha256_cdi}; DSR N {n_tentativas}", flush=True)

    construtores = bracos_lote2(nota)
    fase_pedida = fase_do_trabalhador(argv)
    if fase_pedida is not None:
        trabalhador(fase_pedida, dados, construtores, "--ao-contrario" in argv)
        return 0
    bracos = tuple(construtores)
    medidas: dict[str, dict[str, list[float]]] = {nome: {"composto": [], "conservadora": [], "bova": [], "dsr": []} for nome in bracos}
    anos = None
    for fase in FASES:
        precos = dados.precos.loc[:LAST_SESSION].iloc[fase:]
        acoes = _acoes(precos)
        conservadora = walk_forward_dinamico(acoes, dados.universo, estrategia_conservadora(mensal.retornos, mensal.proventos, dados.setores),
                                             CFG_BT, CONSERVADORA, Congelamento())
        check_plumbing(fase, cagr(conservadora.valores.dropna()))
        composto = walk_forward_dinamico(acoes, dados.universo, estrategia_por_nota(nota, dados.setores), CFG_BT, COMPOSTO, Congelamento())
        conferir_composto(fase, cagr(composto.valores.dropna()))
        resultados = {nome: resultado_do_braco(nome, construir, fase, acoes, dados) for nome, construir in construtores.items()}
        valores = {nome: resultado.valores for nome, resultado in resultados.items()}
        valores |= {COMPOSTO: composto.valores, CONSERVADORA: conservadora.valores}
        valores |= {nome: curva.valores for nome, curva in referencias(precos, CFG_BT).items()}
        check_aligned(valores)
        rf = market_data.serie_rf_diaria(valores[COMPOSTO].index, cdi, (1.0 + float(cdi.iloc[-1])) ** DIAS_DO_ANO - 1.0)
        comparadas = {nome: valores[nome] for nome in (*bracos, COMPOSTO, CONSERVADORA)}
        tabela = tabela_do_periodo(comparadas, valores["BOVA11"], rf, n_tentativas)
        for nome in (*comparadas, "BOVA11"):
            tabela.loc[nome, "CAGR"] = cagr(valores[nome].dropna())
            tabela.loc[nome, "MDD"] = drawdown(valores[nome].dropna())[1]
        for nome in bracos:
            tabela.loc[nome, "P[> Composto]"] = psr_da_diferenca(valores[nome], valores[COMPOSTO])
            tabela.loc[nome, "P[> Conservadora]"] = psr_da_diferenca(valores[nome], valores[CONSERVADORA])
            medidas[nome]["composto"].append(float(tabela.loc[nome, "P[> Composto]"]))
            medidas[nome]["conservadora"].append(float(tabela.loc[nome, "P[> Conservadora]"]))
            medidas[nome]["bova"].append(float(tabela.loc[nome, "P aritmético"]))
            medidas[nome]["dsr"].append(float(tabela.loc[nome, "DSR"]))
        datas = valores[COMPOSTO].index
        print(f"\n=== Phase {fase}: {datas[0]:%d/%m/%Y} to {datas[-1]:%d/%m/%Y} ===")
        print(tabela.to_string(float_format=lambda valor: f"{valor:.3f}"))
        for nome, resultado in resultados.items():
            print(f"{nome}: refusals {len(resultado.recusas)}, failures {len(resultado.falhas)}, annual turnover {resultado.turnover_anual_medio:.2f}")
        print(_year_table({nome: valores[nome] for nome in comparadas | {"BOVA11": valores["BOVA11"]}}).to_string(float_format=lambda valor: f"{valor:.1%}"))
        print(f"(phase {fase} done in {time.perf_counter() - inicio:.0f} s)", flush=True)
        if fase == FASES[0]:
            anos = anos_em_reais(pd.DataFrame(comparadas), valores["BOVA11"], pd.read_parquet(JUROS).set_index("date"))

    portao = []
    print(f"\nGate, all >= {P_MIN} in every phase:")
    for nome in bracos:
        medida = medidas[nome]
        passa = strategy_gate([*medida["composto"], *medida["conservadora"]], medida["bova"], medida["dsr"])
        portao.append({"braco": nome, "p_composto": medida["composto"], "p_conservadora": medida["conservadora"],
                       "p_referencia": medida["bova"], "dsr": medida["dsr"], "passa": passa})
        print(f"{nome}: P[> Composite] {np.round(medida['composto'], 3)}, P[> Conservadora] {np.round(medida['conservadora'], 3)}, "
              f"P[> BOVA11] {np.round(medida['bova'], 3)}, DSR {np.round(medida['dsr'], 3)}: {'PASSES' if passa else 'does not pass'}")
    anos.to_parquet(SAIDA / "lote2-anos-b3.parquet", index=False)
    pd.DataFrame(portao).to_parquet(SAIDA / "lote2-portao-b3.parquet", index=False)
    print("\n" + anos.to_string(float_format=lambda valor: f"{valor:.1%}"))
    print(f"\nDone in {time.perf_counter() - inicio:.0f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
