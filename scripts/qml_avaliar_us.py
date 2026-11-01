"""
qml_avaliar_us.py - The hybrid QPO's reading on US stocks (docs/superpowers/specs/2026-11-01-qpo-hibrido-design.md, sections 4
and 5), on the exports frozen after the 01/11 Tiingo batch. Same configurations as the B3, so N does not grow.

    python -X utf8 scripts/qml_avaliar_us.py D:/b3-datalake/exports/qml/us/exports > D:/b3-datalake/exports/qml/us/leitura.txt

Writes docs/dados/qpo-hibrido-{curvas,anos,portao,carteiras}-us.parquet: what the screen shows.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core import us_teste as ut  # noqa: E402
from core.backtest import walk_forward_dinamico  # noqa: E402
from core.composto import estrategia_por_nota  # noqa: E402
from core.mv_continuo import estrategia_mv_continua_grande  # noqa: E402
from core.periodos import anos_em_reais  # noqa: E402
from core.qml_nota import nota_da_previsao  # noqa: E402
from core.qubo_giro import TAU, estrategia_qubo_giro  # noqa: E402
from core.robustez import psr_da_diferenca  # noqa: E402
from core.sinais import estrategia_conservadora  # noqa: E402
from scripts.avaliar_robustez import contar_tentativas  # noqa: E402
from scripts.paper_v3 import nota_do_composto  # noqa: E402
from scripts.qml_avaliar_b3 import BRACOS, COMPOSTO, JUROS, SAIDA, conferir_previsoes, previsoes_do_modelo  # noqa: E402
from scripts.qml_ler_ic import PASTA, tabela_das_previsoes  # noqa: E402
from scripts.qubo_universo_avaliar import MATCH_MIN, anneal_status  # noqa: E402
from scripts.us_avaliar import CONSERVADORA, FASES, SPY, _texto, anos, carregar, medidas_da_fase  # noqa: E402

CURVAS_DA_B3 = SAIDA / "curvas-ao-vivo.parquet"


def main(argv: list[str]) -> int:
    inicio = time.perf_counter()
    dados = carregar(Path(argv[0]))
    n = contar_tentativas()
    previsoes = tabela_das_previsoes(PASTA / "us")
    nota = nota_do_composto(dados.retornos, dados.proventos, dados.balancos)
    rf = dados.fatores["rf"]
    print(f"Coverage: {dados.cobertura}; DSR N {n}; tau {TAU:.3e}; last month {dados.retornos.index[-1]:%Y-%m}")

    p_composto: dict[str, list[float]] = {nome: [] for nome in BRACOS}
    p_conservadora: dict[str, list[float]] = {nome: [] for nome in BRACOS}
    p_spy: dict[str, list[float]] = {nome: [] for nome in BRACOS}
    dsr: dict[str, list[float]] = {nome: [] for nome in BRACOS}
    logs: dict[pd.Timestamp, object] = {}
    linhas_das_curvas, carteiras, tabela_de_anos = [], None, None
    for fase in FASES:
        precos = ut.indice_total(dados.retornos).iloc[fase:]
        log_da_fase: dict[pd.Timestamp, object] = {}
        estrategias = {COMPOSTO: estrategia_por_nota(nota, dados.setores),
                       CONSERVADORA: estrategia_conservadora(dados.retornos, dados.proventos, dados.setores),
                       "VQC-15": estrategia_por_nota(nota_da_previsao(previsoes_do_modelo(previsoes, "VQC")), dados.setores),
                       "Fourier-15": estrategia_por_nota(nota_da_previsao(previsoes_do_modelo(previsoes, "Fourier")), dados.setores),
                       "QUBO-giro": estrategia_qubo_giro(nota, dados.setores, log=log_da_fase),
                       "MV-continuo": estrategia_mv_continua_grande(nota, dados.setores)}
        resultados = {nome: walk_forward_dinamico(precos, dados.universo, ut.no_grid_mensal(estrategia), ut.CONFIG, nome)
                      for nome, estrategia in estrategias.items()}
        conferir_previsoes(resultados)
        logs.update(log_da_fase)
        curvas = {nome: resultado.valores.dropna() for nome, resultado in resultados.items()}
        datas = curvas[COMPOSTO].index
        spy = ut.referencia_nas_datas(dados.etfs[SPY], datas)
        tabela = medidas_da_fase(curvas, spy, rf, n, dados.fatores)
        for nome in BRACOS:
            tabela.loc[nome, "P[> Composto]"] = psr_da_diferenca(curvas[nome], curvas[COMPOSTO])
            tabela.loc[nome, "P[> Conservadora]"] = psr_da_diferenca(curvas[nome], curvas[CONSERVADORA])
            p_composto[nome].append(float(tabela.loc[nome, "P[> Composto]"]))
            p_conservadora[nome].append(float(tabela.loc[nome, "P[> Conservadora]"]))
            p_spy[nome].append(float(tabela.loc[nome, "P[> SPY]"]))
            dsr[nome].append(float(tabela.loc[nome, "DSR"]))
        print(f"\n=== Phase {fase}: {datas[0]:%m/%Y} to {datas[-1]:%m/%Y}, {len(datas) - 1} months ===")
        print(_texto(tabela))
        for nome, resultado in resultados.items():
            print(f"{nome}: refusals {len(resultado.recusas)}, failures {len(resultado.falhas)}, "
                  f"annual turnover {resultado.turnover_anual_medio:.2f}")
        print(f"SPY: {ut.medidas(spy, rf)}")
        print(_texto(anos({**curvas, SPY: spy}) * 100, 1))
        print(f"(phase {fase} done in {time.perf_counter() - inicio:.0f} s)", flush=True)
        linhas_das_curvas.append(pd.DataFrame({**curvas, SPY: spy}).rename_axis("date").reset_index().assign(fase=fase))
        if fase == FASES[0]:
            vivas = pd.read_parquet(CURVAS_DA_B3)
            bova11 = vivas[vivas["fase"] == vivas["fase"].min()].set_index("date")["BOVA11"]
            em_reais = pd.DataFrame({f"{nome} EUA": ut.em_reais(curvas[nome], dados.ptax) for nome in curvas})
            tabela_de_anos = anos_em_reais(em_reais, bova11, pd.read_parquet(JUROS).set_index("date"))
            carteiras = pd.concat([resultado.pesos_alvo.iloc[[-1]].T.iloc[:, 0].pipe(lambda s: s[s > 1e-6]).rename("peso")
                                   .rename_axis("papel").reset_index()
                                   .assign(braco=nome, data=resultado.pesos_alvo.index[-1])
                                   for nome, resultado in resultados.items() if nome != CONSERVADORA], ignore_index=True)

    print("\n" + anneal_status(logs))
    concordancia = float(np.mean([registro.agree for registro in logs.values()]))
    portao = []
    print("\nGate, all >= 0.95 in every phase:")
    for nome in BRACOS:
        passa = ut.passa_o_portao([*p_composto[nome], *p_conservadora[nome], *p_spy[nome]], dsr[nome])
        if nome == "QUBO-giro":
            passa = passa and concordancia >= MATCH_MIN - 1e-12
        portao.append({"braco": nome, "p_composto": p_composto[nome], "p_conservadora": p_conservadora[nome],
                       "p_referencia": p_spy[nome], "dsr": dsr[nome], "passa": passa})
        print(f"{nome}: P[> Composite] {np.round(p_composto[nome], 3)}, P[> Conservadora] {np.round(p_conservadora[nome], 3)}, "
              f"P[> SPY] {np.round(p_spy[nome], 3)}, DSR {np.round(dsr[nome], 3)}: {'PASSES' if passa else 'does not pass'}")

    pd.concat(linhas_das_curvas, ignore_index=True).to_parquet(SAIDA / "qpo-hibrido-curvas-us.parquet", index=False)
    tabela_de_anos.to_parquet(SAIDA / "qpo-hibrido-anos-us.parquet", index=False)
    pd.DataFrame(portao).to_parquet(SAIDA / "qpo-hibrido-portao-us.parquet", index=False)
    carteiras.to_parquet(SAIDA / "qpo-hibrido-carteiras-us.parquet", index=False)
    print("\n" + tabela_de_anos.to_string(float_format=lambda valor: f"{valor:.1%}"))
    print(f"\nDone in {time.perf_counter() - inicio:.0f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
