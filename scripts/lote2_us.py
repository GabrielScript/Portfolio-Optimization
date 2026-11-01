"""
lote2_us.py - The one reading of the book's second batch on US stocks (docs/superpowers/specs/2026-11-01-lote2-livro-design.md, section 2),
on the exports frozen for the hybrid QPO. Runs after the first batch's US reading, one heavy job at a time.

    python -X utf8 scripts/lote2_us.py D:/b3-datalake/exports/qml/us/exports docs/us-estrategias-2026-11-01.txt
        > D:/b3-datalake/exports/lote2/us/leitura.txt

Each arm's result is saved per phase, so a crash resumes where it stopped. Writes docs/dados/lote2-{anos,portao}-us.parquet.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core import us_teste as ut  # noqa: E402
from core.backtest import ResultadoBacktest, estrategia_equiponderada, walk_forward_dinamico  # noqa: E402
from core.composto import estrategia_por_nota  # noqa: E402
from core.estrategias_lote2 import bracos_lote2  # noqa: E402
from core.estrategias_todas import Construtor, no_grid_mensal_real  # noqa: E402
from core.periodos import anos_em_reais  # noqa: E402
from core.robustez import psr_da_diferenca  # noqa: E402
from core.sinais import estrategia_conservadora  # noqa: E402
from scripts.avaliar_robustez import contar_tentativas  # noqa: E402
from scripts.paper_v3 import nota_do_composto  # noqa: E402
from scripts.qml_avaliar_b3 import COMPOSTO, JUROS, SAIDA  # noqa: E402
from scripts.qml_avaliar_us import CURVAS_DA_B3  # noqa: E402
from scripts.todas_comum import esperar_a_janela_passar, resultado_salvo  # noqa: E402
from scripts.todas_us import cagrs_do_relatorio, conferir_encanamento  # noqa: E402
from scripts.us_avaliar import CONSERVADORA, FASES, SPY, UM_PARA_N, Dados, _texto, anos, carregar, medidas_da_fase  # noqa: E402

PASTA = Path("D:/b3-datalake/exports/lote2/us")
PERIODOS = 12


def _rodar_braco(nome: str, construir: Construtor, fase: int, precos: pd.DataFrame, dados: Dados) -> ResultadoBacktest:
    def rodar() -> ResultadoBacktest:
        esperar_a_janela_passar()
        inicio = time.perf_counter()
        resultado = walk_forward_dinamico(precos, dados.universo, no_grid_mensal_real(construir(dados.setores, PERIODOS)), ut.CONFIG, nome)
        print(f"{nome} phase {fase}: {time.perf_counter() - inicio:.0f} s; first failures {resultado.falhas[:2]}", flush=True)
        return resultado
    return resultado_salvo(PASTA / f"{nome}-fase{fase}.pkl", rodar)


def main(argv: list[str]) -> int:
    inicio = time.perf_counter()
    PASTA.mkdir(parents=True, exist_ok=True)
    dados = carregar(Path(argv[0]))
    publicado = cagrs_do_relatorio(Path(argv[1]).read_text(encoding="utf-8"), (COMPOSTO, CONSERVADORA, UM_PARA_N))
    n = contar_tentativas()
    rf = dados.fatores["rf"]
    nota = nota_do_composto(dados.retornos, dados.proventos, dados.balancos)
    construtores = bracos_lote2(nota)
    bracos = tuple(construtores)
    print(f"Coverage: {dados.cobertura}; DSR N {n}; last month {dados.retornos.index[-1]:%Y-%m}; {len(bracos)} arms", flush=True)

    p_composto: dict[str, list[float]] = {}
    p_conservadora: dict[str, list[float]] = {}
    p_spy: dict[str, list[float]] = {}
    dsr: dict[str, list[float]] = {}
    tabela_de_anos = None
    for fase in FASES:
        precos = ut.indice_total(dados.retornos).iloc[fase:]
        composto = walk_forward_dinamico(precos, dados.universo, ut.no_grid_mensal(estrategia_por_nota(nota, dados.setores)), ut.CONFIG, COMPOSTO)
        cons = walk_forward_dinamico(precos, dados.universo, ut.no_grid_mensal(estrategia_conservadora(dados.retornos, dados.proventos, dados.setores)), ut.CONFIG, CONSERVADORA)
        um_para_n = walk_forward_dinamico(precos, dados.universo, ut.no_grid_mensal(estrategia_equiponderada), ut.CONFIG, UM_PARA_N)
        referencias = {COMPOSTO: composto.valores.dropna(), CONSERVADORA: cons.valores.dropna()}
        datas = referencias[COMPOSTO].index
        spy = ut.referencia_nas_datas(dados.etfs[SPY], datas)
        medidos = {nome: ut.medidas(curva, rf)["CAGR"] for nome, curva in {**referencias, UM_PARA_N: um_para_n.valores.dropna()}.items()}
        conferir_encanamento(publicado[fase], medidos, fase)

        resultados = {nome: _rodar_braco(nome, construir, fase, precos, dados) for nome, construir in construtores.items()}
        curvas = {nome: resultado.valores.dropna() for nome, resultado in resultados.items()}
        comparadas = {**curvas, **referencias}
        tabela = medidas_da_fase(comparadas, spy, rf, n, dados.fatores)
        for nome in bracos:
            tabela.loc[nome, "P[> Composto]"] = psr_da_diferenca(curvas[nome], referencias[COMPOSTO])
            tabela.loc[nome, "P[> Conservadora]"] = psr_da_diferenca(curvas[nome], referencias[CONSERVADORA])
            p_composto.setdefault(nome, []).append(float(tabela.loc[nome, "P[> Composto]"]))
            p_conservadora.setdefault(nome, []).append(float(tabela.loc[nome, "P[> Conservadora]"]))
            p_spy.setdefault(nome, []).append(float(tabela.loc[nome, "P[> SPY]"]))
            dsr.setdefault(nome, []).append(float(tabela.loc[nome, "DSR"]))
        print(f"\n=== Phase {fase}: {datas[0]:%m/%Y} to {datas[-1]:%m/%Y}, {len(datas) - 1} months ===")
        print(_texto(tabela))
        for nome, resultado in {**resultados, COMPOSTO: composto, CONSERVADORA: cons}.items():
            print(f"{nome}: refusals {len(resultado.recusas)}, failures {len(resultado.falhas)}, annual turnover {ut.giro_anual_mensal(resultado):.2f}")
        print(f"SPY: {ut.medidas(spy, rf)}")
        print(_texto(anos({**comparadas, SPY: spy}) * 100, 1))
        print(f"(phase {fase} done in {time.perf_counter() - inicio:.0f} s)", flush=True)
        if fase == FASES[0]:
            vivas = pd.read_parquet(CURVAS_DA_B3)
            bova11 = vivas[vivas["fase"] == vivas["fase"].min()].set_index("date")["BOVA11"]
            em_reais = pd.DataFrame({f"{nome} EUA": ut.em_reais(curva, dados.ptax) for nome, curva in comparadas.items()})
            tabela_de_anos = anos_em_reais(em_reais, bova11, pd.read_parquet(JUROS).set_index("date"))

    portao = []
    print("\nGate, all >= 0.95 in every phase:")
    for nome in bracos:
        passa = ut.passa_o_portao([*p_composto[nome], *p_conservadora[nome], *p_spy[nome]], dsr[nome])
        portao.append({"braco": nome, "p_composto": p_composto[nome], "p_conservadora": p_conservadora[nome],
                       "p_referencia": p_spy[nome], "dsr": dsr[nome], "passa": passa})
        print(f"{nome}: P[> Composite] {np.round(p_composto[nome], 3)}, P[> Conservadora] {np.round(p_conservadora[nome], 3)}, "
              f"P[> SPY] {np.round(p_spy[nome], 3)}, DSR {np.round(dsr[nome], 3)}: {'PASSES' if passa else 'does not pass'}")
    tabela_de_anos.to_parquet(SAIDA / "lote2-anos-us.parquet", index=False)
    pd.DataFrame(portao).to_parquet(SAIDA / "lote2-portao-us.parquet", index=False)
    print("\n" + tabela_de_anos.to_string(float_format=lambda valor: f"{valor:.1%}"))
    print(f"\nDone in {time.perf_counter() - inicio:.0f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
