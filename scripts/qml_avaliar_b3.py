"""
qml_avaliar_b3.py - The hybrid QPO's one reading on the B3 (docs/superpowers/specs/2026-11-01-qpo-hibrido-design.md, sections
4 and 5). Runs only after the pre-registration commit (`e44265e`) and ``qml_prever``. No network: everything frozen.

    python -X utf8 scripts/qml_avaliar_b3.py > D:/b3-datalake/exports/qml/b3/leitura.txt

Writes docs/dados/qpo-hibrido-{curvas,anos,portao,carteiras}-b3.parquet: what the screen shows.
"""

from __future__ import annotations

import sys
import time
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core import market_data  # noqa: E402
from core.backtest import Congelamento, ResultadoBacktest, walk_forward_dinamico  # noqa: E402
from core.cdi_lago import daily_cdi  # noqa: E402
from core.composto import estrategia_por_nota  # noqa: E402
from core.metrics import cagr, drawdown  # noqa: E402
from core.mv_continuo import estrategia_mv_continua  # noqa: E402
from core.periodos import anos_em_reais  # noqa: E402
from core.qml_nota import nota_da_previsao  # noqa: E402
from core.qubo_giro import TAU, estrategia_qubo_giro  # noqa: E402
from core.robustez import psr_da_diferenca  # noqa: E402
from core.sinais import estrategia_conservadora  # noqa: E402
from scripts import avaliar_conservadora, avaliar_robustez  # noqa: E402
from scripts.avaliar_robustez import FASES, _acoes, conferir_hashes, contar_tentativas, referencias, tabela_do_periodo  # noqa: E402
from scripts.avaliar_universo_dinamico import ANOS_DE_CDI, CFG_BT  # noqa: E402
from scripts.qml_ler_ic import PASTA, tabela_das_previsoes  # noqa: E402
from scripts.qpo_avaliar import CONSERVADORA, _year_table, check_aligned, check_plumbing  # noqa: E402
from scripts.qpo_montar import FOLDER, LAST_SESSION  # noqa: E402
from scripts.qubo_universo_avaliar import MATCH_MIN, P_MIN, _value_score, anneal_status, strategy_gate  # noqa: E402

COMPOSTO = "Composto"
BRACOS = ("VQC-15", "Fourier-15", "QUBO-giro", "MV-continuo")
PUBLICADO_DO_COMPOSTO = {0: 0.214, 21: 0.199, 42: 0.191}
TOLERANCIA_DO_COMPOSTO = 0.0006     # the published figures have one decimal
MACRO = FOLDER.parent / "2026-10-21-macro"
SAIDA = Path(__file__).resolve().parent.parent / "docs" / "dados"
JUROS = SAIDA / "juros-diarios.parquet"


def previsoes_do_modelo(tabela: pd.DataFrame, modelo: str) -> pd.DataFrame:
    return tabela[["month", "security", modelo]].rename(columns={modelo: "previsao"}).dropna(subset=["previsao"])


def conferir_composto(fase: int, valor: float) -> None:
    if abs(valor - PUBLICADO_DO_COMPOSTO[fase]) > TOLERANCIA_DO_COMPOSTO:
        raise SystemExit(f"phase {fase}: Composite CAGR {valor:.2%} != published {PUBLICADO_DO_COMPOSTO[fase]:.1%}")


def conferir_previsoes(resultados: Mapping[str, ResultadoBacktest]) -> None:
    """The engine records a failure and keeps the portfolio; a month without a forecast must stop the reading instead."""
    for nome, resultado in resultados.items():
        faltou = [falha for falha in resultado.falhas if "no forecast" in falha]
        if faltou:
            raise SystemExit(f"{nome}: {faltou[:3]}")


def main() -> int:
    inicio = time.perf_counter()
    avaliar_robustez.PASTA_PRECOS = avaliar_robustez.PASTA_UNIVERSO = FOLDER
    dados = avaliar_robustez.carregar()
    mensal = avaliar_conservadora.carregar_mensal(FOLDER)
    nota, sha256_cvm = _value_score(mensal, com_crescimento=True)
    sha256_cdi = conferir_hashes(MACRO, ["cdi_daily.parquet"])
    cdi = daily_cdi(ANOS_DE_CDI, LAST_SESSION.date(), path=MACRO / "cdi_daily.parquet")
    previsoes = tabela_das_previsoes(PASTA / "b3")
    n_tentativas = contar_tentativas()
    print(f"Files (sha256): {dados.sha256 | mensal.sha256 | sha256_cvm | sha256_cdi}; DSR N {n_tentativas}; tau {TAU:.3e}")

    p_composto: dict[str, list[float]] = {nome: [] for nome in BRACOS}
    p_conservadora: dict[str, list[float]] = {nome: [] for nome in BRACOS}
    p_bova: dict[str, list[float]] = {nome: [] for nome in BRACOS}
    dsr: dict[str, list[float]] = {nome: [] for nome in BRACOS}
    logs: dict[pd.Timestamp, object] = {}
    linhas_das_curvas, carteiras, anos = [], None, None
    for fase in FASES:
        acoes = _acoes(dados.precos.loc[:LAST_SESSION].iloc[fase:])

        def rodar(estrategia, nome) -> ResultadoBacktest:
            return walk_forward_dinamico(acoes, dados.universo, estrategia, CFG_BT, nome, Congelamento())

        conservadora = rodar(estrategia_conservadora(mensal.retornos, mensal.proventos, dados.setores), CONSERVADORA)
        check_plumbing(fase, cagr(conservadora.valores.dropna()))
        composto = rodar(estrategia_por_nota(nota, dados.setores), COMPOSTO)
        conferir_composto(fase, cagr(composto.valores.dropna()))
        log_da_fase: dict[pd.Timestamp, object] = {}
        estrategias = {"VQC-15": estrategia_por_nota(nota_da_previsao(previsoes_do_modelo(previsoes, "VQC")), dados.setores),
                       "Fourier-15": estrategia_por_nota(nota_da_previsao(previsoes_do_modelo(previsoes, "Fourier")),
                                                         dados.setores),
                       "QUBO-giro": estrategia_qubo_giro(nota, dados.setores, log=log_da_fase),
                       "MV-continuo": estrategia_mv_continua(nota, dados.setores)}
        resultados = {nome: rodar(estrategia, nome) for nome, estrategia in estrategias.items()}
        conferir_previsoes(resultados)
        logs.update(log_da_fase)
        valores = {nome: resultado.valores for nome, resultado in resultados.items()}
        valores |= {COMPOSTO: composto.valores, CONSERVADORA: conservadora.valores}
        valores |= {nome: curva.valores for nome, curva in referencias(dados.precos.loc[:LAST_SESSION].iloc[fase:], CFG_BT).items()}
        check_aligned(valores)
        rf = market_data.serie_rf_diaria(valores[COMPOSTO].index, cdi, (1.0 + float(cdi.iloc[-1])) ** 252 - 1.0)
        comparadas = {nome: valores[nome] for nome in (*BRACOS, COMPOSTO, CONSERVADORA)}
        tabela = tabela_do_periodo(comparadas, valores["BOVA11"], rf, n_tentativas)
        for nome in (*comparadas, "BOVA11"):
            tabela.loc[nome, "CAGR"] = cagr(valores[nome].dropna())
            tabela.loc[nome, "MDD"] = drawdown(valores[nome].dropna())[1]
        for nome in BRACOS:
            tabela.loc[nome, "P[> Composto]"] = psr_da_diferenca(valores[nome], valores[COMPOSTO])
            tabela.loc[nome, "P[> Conservadora]"] = psr_da_diferenca(valores[nome], valores[CONSERVADORA])
            p_composto[nome].append(float(tabela.loc[nome, "P[> Composto]"]))
            p_conservadora[nome].append(float(tabela.loc[nome, "P[> Conservadora]"]))
            p_bova[nome].append(float(tabela.loc[nome, "P aritmético"]))
            dsr[nome].append(float(tabela.loc[nome, "DSR"]))
        datas = valores[COMPOSTO].index
        print(f"\n=== Phase {fase}: {datas[0]:%d/%m/%Y} to {datas[-1]:%d/%m/%Y} ===")
        print(tabela.to_string(float_format=lambda valor: f"{valor:.3f}"))
        for nome, resultado in {**resultados, COMPOSTO: composto, CONSERVADORA: conservadora}.items():
            print(f"{nome}: refusals {len(resultado.recusas)}, failures {len(resultado.falhas)}, "
                  f"annual turnover {resultado.turnover_anual_medio:.2f}")
        print(_year_table({nome: valores[nome] for nome in comparadas | {"BOVA11": valores["BOVA11"]}})
              .to_string(float_format=lambda valor: f"{valor:.1%}"))
        print(f"(phase {fase} done in {time.perf_counter() - inicio:.0f} s)", flush=True)
        linhas_das_curvas.append(pd.DataFrame(valores).rename_axis("date").reset_index().assign(fase=fase))
        if fase == FASES[0]:
            juros = pd.read_parquet(JUROS).set_index("date")
            anos = anos_em_reais(pd.DataFrame(comparadas), valores["BOVA11"], juros)
            carteiras = pd.concat([resultado.pesos_alvo.iloc[[-1]].T.iloc[:, 0].pipe(lambda s: s[s > 1e-6]).rename("peso")
                                   .rename_axis("papel").reset_index()
                                   .assign(braco=nome, data=resultado.pesos_alvo.index[-1])
                                   for nome, resultado in {**resultados, COMPOSTO: composto}.items()], ignore_index=True)

    print("\n" + anneal_status(logs))
    concordancia = float(np.mean([registro.agree for registro in logs.values()]))
    portao = []
    print(f"\nGate, all >= {P_MIN} in every phase:")
    for nome in BRACOS:
        passa = strategy_gate([*p_composto[nome], *p_conservadora[nome]], p_bova[nome], dsr[nome])
        if nome == "QUBO-giro":
            passa = passa and concordancia >= MATCH_MIN - 1e-12
        portao.append({"braco": nome, "p_composto": p_composto[nome], "p_conservadora": p_conservadora[nome],
                       "p_bova": p_bova[nome], "dsr": dsr[nome], "passa": passa})
        print(f"{nome}: P[> Composite] {np.round(p_composto[nome], 3)}, P[> Conservadora] {np.round(p_conservadora[nome], 3)}, "
              f"P[> BOVA11] {np.round(p_bova[nome], 3)}, DSR {np.round(dsr[nome], 3)}: {'PASSES' if passa else 'does not pass'}")

    pd.concat(linhas_das_curvas, ignore_index=True).to_parquet(SAIDA / "qpo-hibrido-curvas-b3.parquet", index=False)
    anos.to_parquet(SAIDA / "qpo-hibrido-anos-b3.parquet", index=False)
    pd.DataFrame(portao).to_parquet(SAIDA / "qpo-hibrido-portao-b3.parquet", index=False)
    carteiras.to_parquet(SAIDA / "qpo-hibrido-carteiras-b3.parquet", index=False)
    print("\n" + anos.to_string(float_format=lambda valor: f"{valor:.1%}"))
    print(f"\nDone in {time.perf_counter() - inicio:.0f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
