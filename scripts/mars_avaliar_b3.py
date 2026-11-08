"""
mars_avaliar_b3.py - the one reading of MARS-50 and MARS-50+score on the B3 (docs/superpowers/specs/
2026-11-08-mars-design.md, section 5): the 213 engine, frozen data and plumbing checks, each arm's daily curve (equal
capital in seeds 42, 0, 1) on each phase's dates, the DSR at the pre-registered union N = 106; then the descriptive
extras (per seed, MARS-Static, the controller's group weights, the paper's table for 2022 and 2024).

    python -X utf8 scripts/mars_avaliar_b3.py kan_run/mars/saidas > docs/mars-2026-11-08.txt
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
from core.robustez import psr_da_diferenca  # noqa: E402
from core.sinais import estrategia_conservadora  # noqa: E402
from scripts import avaliar_conservadora, avaliar_robustez  # noqa: E402
from scripts.avaliar_robustez import FASES, _acoes, conferir_hashes, contar_tentativas, referencias, tabela_do_periodo  # noqa: E402
from scripts.avaliar_universo_dinamico import ANOS_DE_CDI, CFG_BT  # noqa: E402
from scripts.qml_avaliar_b3 import COMPOSTO, MACRO, conferir_composto  # noqa: E402
from scripts.qpo_avaliar import CONSERVADORA, _year_table, check_aligned, check_plumbing  # noqa: E402
from scripts.qpo_montar import FOLDER, LAST_SESSION  # noqa: E402
from scripts.qubo_universo_avaliar import P_MIN, _value_score, strategy_gate  # noqa: E402

BRACOS = {"MARS-50": "MARS-50", "MARS-50+score": "MARS-50_score"}
SEMENTES = (42, 0, 1)
N_DSR = 106         # the union of every branch's trials, section 5
GRUPOS = {"conservative (0-2)": slice(0, 3), "neutral (3-6)": slice(3, 7), "aggressive (7-9)": slice(7, 10)}


def curva_da_semente(pasta: Path, braco: str, semente: int, chave: str = "valores") -> pd.Series:
    arquivo = np.load(pasta / f"mars_{BRACOS[braco]}_s{semente}.npz")
    serie = pd.Series(arquivo[chave], index=pd.DatetimeIndex(arquivo["datas"].astype("datetime64[ns]")))
    return serie / serie.iloc[0]


def curva_do_braco(pasta: Path, braco: str, chave: str = "valores") -> pd.Series:
    """One third of the capital in each seed's run from the first test session, not rebalanced."""
    return sum(curva_da_semente(pasta, braco, semente, chave) for semente in SEMENTES) / len(SEMENTES)


def na_fase(curva: pd.Series, datas: pd.DatetimeIndex, inicio: float) -> pd.Series:
    """The curve on the phase's sessions, scaled to start at the engine's starting value."""
    if not datas.isin(curva.index).all():
        faltam = datas[~datas.isin(curva.index)]
        raise SystemExit(f"MARS curve misses {len(faltam)} engine sessions, first {faltam[0]:%d/%m/%Y}")
    recorte = curva.reindex(datas)
    return recorte / recorte.iloc[0] * inicio


def tabela_do_artigo(curvas: dict[str, pd.Series], ano: int, rf: pd.Series) -> pd.DataFrame:
    """The paper's Table 2 columns for one calendar year: CR, AR, SR (excess over the CDI), AVol, MDD."""
    linhas = {}
    for nome, curva in curvas.items():
        ano_ = curva[curva.index.year == ano]
        anterior = curva[curva.index < ano_.index[0]]
        serie = pd.concat([anterior.iloc[-1:], ano_]) if len(anterior) else ano_
        retornos = serie.pct_change().dropna()
        excesso = retornos - rf.reindex(retornos.index).fillna(0.0)
        acumulado = serie.iloc[-1] / serie.iloc[0] - 1.0
        linhas[nome] = {"CR%": 100 * acumulado, "AR%": 100 * ((1 + acumulado) ** (252 / len(retornos)) - 1),
                        "SR": excesso.mean() / retornos.std() * np.sqrt(252), "AVol%": 100 * retornos.std() * np.sqrt(252),
                        "MDD%": 100 * drawdown(serie)[1]}
    return pd.DataFrame(linhas).T


def main(argv: list[str]) -> int:
    inicio = time.perf_counter()
    pasta = Path(argv[0])
    avaliar_robustez.PASTA_PRECOS = avaliar_robustez.PASTA_UNIVERSO = FOLDER
    dados = avaliar_robustez.carregar()
    mensal = avaliar_conservadora.carregar_mensal(FOLDER)
    nota, sha256_cvm = _value_score(mensal, com_crescimento=True)
    sha256_cdi = conferir_hashes(MACRO, ["cdi_daily.parquet"])
    cdi = daily_cdi(ANOS_DE_CDI, LAST_SESSION.date(), path=MACRO / "cdi_daily.parquet")
    mars = {braco: curva_do_braco(pasta, braco) for braco in BRACOS}
    print(f"Files (sha256): {dados.sha256 | mensal.sha256 | sha256_cvm | sha256_cdi}; "
          f"DSR N {N_DSR} (this branch's file counts {contar_tentativas()})")

    gate = {braco: {"composto": [], "conservadora": [], "bova": [], "dsr": []} for braco in BRACOS}
    for fase in FASES:
        acoes = _acoes(dados.precos.loc[:LAST_SESSION].iloc[fase:])

        def rodar(estrategia, nome) -> ResultadoBacktest:
            return walk_forward_dinamico(acoes, dados.universo, estrategia, CFG_BT, nome, Congelamento())

        conservadora = rodar(estrategia_conservadora(mensal.retornos, mensal.proventos, dados.setores), CONSERVADORA)
        check_plumbing(fase, cagr(conservadora.valores.dropna()))
        composto = rodar(estrategia_por_nota(nota, dados.setores), COMPOSTO)
        conferir_composto(fase, cagr(composto.valores.dropna()))
        datas = composto.valores.index
        valores = {braco: na_fase(curva, datas, float(composto.valores.iloc[0])) for braco, curva in mars.items()}
        valores |= {COMPOSTO: composto.valores, CONSERVADORA: conservadora.valores}
        valores |= {nome: curva.valores
                    for nome, curva in referencias(dados.precos.loc[:LAST_SESSION].iloc[fase:], CFG_BT).items()}
        check_aligned(valores)
        rf = market_data.serie_rf_diaria(datas, cdi, (1.0 + float(cdi.iloc[-1])) ** 252 - 1.0)
        comparadas = {nome: valores[nome] for nome in (*BRACOS, COMPOSTO, CONSERVADORA)}
        tabela = tabela_do_periodo(comparadas, valores["BOVA11"], rf, N_DSR)
        for nome in (*comparadas, "BOVA11"):
            tabela.loc[nome, "CAGR"] = cagr(valores[nome].dropna())
            tabela.loc[nome, "MDD"] = drawdown(valores[nome].dropna())[1]
        for braco in BRACOS:
            tabela.loc[braco, "P[> Composto]"] = psr_da_diferenca(valores[braco], valores[COMPOSTO])
            tabela.loc[braco, "P[> Conservadora]"] = psr_da_diferenca(valores[braco], valores[CONSERVADORA])
            gate[braco]["composto"].append(float(tabela.loc[braco, "P[> Composto]"]))
            gate[braco]["conservadora"].append(float(tabela.loc[braco, "P[> Conservadora]"]))
            gate[braco]["bova"].append(float(tabela.loc[braco, "P aritmético"]))
            gate[braco]["dsr"].append(float(tabela.loc[braco, "DSR"]))
        print(f"\n=== Phase {fase}: {datas[0]:%d/%m/%Y} to {datas[-1]:%d/%m/%Y} ===")
        print(tabela.to_string(float_format=lambda valor: f"{valor:.3f}"))
        print(_year_table({**comparadas, "BOVA11": valores["BOVA11"]}).to_string(
            float_format=lambda valor: f"{valor:.1%}"))
        if fase == FASES[0]:
            referencia = {COMPOSTO: composto.valores, "BOVA11": valores["BOVA11"],
                          **{braco: valores[braco] for braco in BRACOS}}
            rf_fase0 = rf
        print(f"(phase {fase} done in {time.perf_counter() - inicio:.0f} s)", flush=True)

    print(f"\nGate, all >= {P_MIN} in every phase:")
    for braco, partes in gate.items():
        passa = strategy_gate([*partes["composto"], *partes["conservadora"]], partes["bova"], partes["dsr"])
        print(f"{braco}: P[> Composite] {np.round(partes['composto'], 3)}, P[> Conservadora] "
              f"{np.round(partes['conservadora'], 3)}, P[> BOVA11] {np.round(partes['bova'], 3)}, "
              f"DSR {np.round(partes['dsr'], 3)}: {'PASSES' if passa else 'does not pass'}")

    print("\n--- Descriptive (not trials) ---")
    for braco in BRACOS:
        for semente in SEMENTES:
            curva = curva_da_semente(pasta, braco, semente)
            estatica = curva_da_semente(pasta, braco, semente, "valores_estaticos")
            print(f"{braco} seed {semente}: CAGR {cagr(curva):.1%}, MDD {drawdown(curva)[1]:.1%}; "
                  f"MARS-Static CAGR {cagr(estatica):.1%}, MDD {drawdown(estatica)[1]:.1%}")
        estatica = curva_do_braco(pasta, braco, "valores_estaticos")
        print(f"{braco} three seeds: CAGR {cagr(mars[braco]):.1%}; MARS-Static {cagr(estatica):.1%}")
    for braco in BRACOS:
        arquivo = np.load(pasta / f"mars_{BRACOS[braco]}_s42.npz")
        datas = pd.DatetimeIndex(arquivo["datas"].astype("datetime64[ns]"))[:-1]
        pesos = pd.DataFrame({grupo: arquivo["pesos"][:, fatia].sum(1) for grupo, fatia in GRUPOS.items()},
                             index=datas)
        print(f"\n{braco} seed 42, the controller's mean weight per group and year (std of the daily weight):")
        anual = pesos.groupby(pesos.index.year)
        print(pd.concat([anual.mean().add_suffix(" mean"), anual.std().add_suffix(" sd")], axis=1).to_string(
            float_format=lambda valor: f"{valor:.3f}"))
        resumo = pd.DataFrame(arquivo["resumo"], columns=["year", "best episode", "validation Sharpe", "seconds"])
        print(resumo.to_string(index=False, float_format=lambda valor: f"{valor:.2f}"))
    for ano in (2022, 2024):
        print(f"\nThe paper's table on the B3, {ano} (phase 0 curves; SR over the CDI):")
        print(tabela_do_artigo(referencia, ano, rf_fase0).to_string(float_format=lambda valor: f"{valor:.2f}"))
    print(f"\nDone in {time.perf_counter() - inicio:.0f} s")
    return 0


if __name__ == "__main__":
    pd.set_option("display.width", 200)
    raise SystemExit(main(sys.argv[1:]))
