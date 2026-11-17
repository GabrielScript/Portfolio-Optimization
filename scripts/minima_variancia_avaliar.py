"""
The one reading of the minimum-variance arms (``docs/superpowers/specs/2026-11-16-minima-variancia-design.md``,
committed before any number).

    ../Portfolio-Optimization/.venv/Scripts/python -X utf8 scripts/minima_variancia_avaliar.py > docs/minima-variancia-2026-11-16.txt

Per phase: the Composite and the screen's minimum variance (the base) replayed by today's code, the three arms, BOVA11
and the CDI; the house gate against the Composite and the spec's "improves minimum variance" criterion against the base.
"""

from __future__ import annotations

import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))

from core import market_data  # noqa: E402
from core.backtest import (Congelamento, Estrategia, ResultadoBacktest, estrategia_nucleo_satelite,  # noqa: E402
                           estrategia_otimizada, walk_forward_dinamico)
from core.composto import estrategia_por_nota  # noqa: E402
from core.config import Objetivo  # noqa: E402
from core.minvar import estrategia_minima_variancia  # noqa: E402
from core.robustez import psr_da_diferenca  # noqa: E402
from scripts import avaliar_conservadora, avaliar_robustez  # noqa: E402
from scripts.avaliar_robustez import FASES, _acoes, referencias  # noqa: E402
from scripts.avaliar_universo_dinamico import ANOS_DE_CDI, CFG_BT, CFG_KELLY  # noqa: E402
from scripts.comparacao_estrategias import resumo_de_giro  # noqa: E402
from scripts.qpo_avaliar import _year_table  # noqa: E402
from scripts.qpo_montar import FOLDER, LAST_SESSION, RUN_DATE  # noqa: E402
from scripts.qubo_universo_avaliar import _value_score, with_retries  # noqa: E402
from scripts.xquad_avaliar import COMPOSITE, DSR_MIN, P_MIN, metricas  # noqa: E402

BASE = "Minimum variance (screen)"
RULES, SCORE, MIX = "MV-regras", "MV-nota", "Composite + MV 50/50"
ARMS = (RULES, SCORE, MIX)
N_DSR = 200                        # 197 (DECISOES 377) + the three arms
VOL_SLACK, MDD_SLACK = 0.010, 0.020
CURVES_FILE = RAIZ / "docs" / "dados" / "minima-variancia-curvas-2026-11-16.parquet"


def estrategias(nota, setores, cdi: pd.Series) -> dict[str, Estrategia]:
    composite = estrategia_por_nota(nota, setores)
    regras = estrategia_minima_variancia(setores)
    return {COMPOSITE: composite,
            BASE: estrategia_otimizada(replace(CFG_KELLY, objetivo=Objetivo.MIN_VOL), True, cdi),
            RULES: regras,
            SCORE: estrategia_minima_variancia(setores, nota),
            MIX: estrategia_nucleo_satelite(composite, regras, 0.5)}


def melhora_a_base(tabela: pd.DataFrame, braco: str) -> bool:
    """The spec's criterion in one phase: more CAGR than the base at about its volatility and drawdown."""
    arm, base = tabela.loc[braco], tabela.loc[BASE]
    return bool(arm["CAGR"] > base["CAGR"] and arm["Vol"] <= base["Vol"] + VOL_SLACK
                and arm["MDD"] >= base["MDD"] - MDD_SLACK and arm["P[> base]"] >= P_MIN)


def main() -> int:
    inicio = time.perf_counter()
    avaliar_robustez.PASTA_PRECOS = avaliar_robustez.PASTA_UNIVERSO = FOLDER
    dados = avaliar_robustez.carregar()
    mensal = avaliar_conservadora.carregar_mensal(FOLDER)
    nota, _ = _value_score(mensal, com_crescimento=True)
    selic = market_data.obter_selic_meta(RUN_DATE)
    cdi = with_retries(lambda: market_data.baixar_cdi_diario(ANOS_DE_CDI, hoje=RUN_DATE))
    precos = dados.precos.loc[:LAST_SESSION]
    print(f"DSR N {N_DSR}; arms {ARMS}; data to {LAST_SESSION:%d/%m/%Y}")

    gate: dict[str, list[tuple[float, float]]] = {braco: [] for braco in ARMS}
    melhora: dict[str, list[bool]] = {braco: [] for braco in ARMS}
    salvas = []
    for fase in FASES:
        resultados: dict[str, ResultadoBacktest] = {
            nome: walk_forward_dinamico(_acoes(precos.iloc[fase:]), dados.universo, estrategia, CFG_BT, nome,
                                        Congelamento())
            for nome, estrategia in estrategias(nota, dados.setores, cdi).items()}
        indice = resultados[COMPOSITE].valores.index
        for nome, resultado in resultados.items():
            if not resultado.valores.index.equals(indice):
                raise SystemExit(f"phase {fase}: {nome} does not share the Composite's dates")
        print(f"\n=== Phase {fase}: {indice[0]:%d/%m/%Y} to {indice[-1]:%d/%m/%Y} ===", flush=True)
        curvas = {nome: resultado.valores for nome, resultado in resultados.items()}
        curvas["BOVA11"] = referencias(precos.iloc[fase:], CFG_BT)["BOVA11"].valores.reindex(indice)
        rf = market_data.serie_rf_diaria(indice, cdi, selic.valor)
        curvas["CDI"] = pd.Series(np.cumprod(1.0 + rf.to_numpy()), index=indice)
        salvas.append(pd.DataFrame(curvas).assign(fase=fase).rename_axis("date").reset_index())
        tabela = metricas(curvas, rf, N_DSR)
        tabela["P[> base]"] = pd.Series({braco: psr_da_diferenca(curvas[braco], curvas[BASE]) for braco in ARMS})
        print(tabela.to_string(float_format=lambda x: f"{x:.3f}"))
        print("Turnover a year, mean names, refusals and failures:")
        print(resumo_de_giro(resultados).to_string(float_format=lambda x: f"{x:.2f}"))
        for nome, resultado in resultados.items():
            for falha in resultado.falhas[:3]:
                print(f"  {nome} failure: {falha}")
        print("Calendar years:")
        print(_year_table(curvas).to_string(float_format=lambda x: f"{x:.1%}"))
        for braco in ARMS:
            gate[braco].append((float(tabela.loc[braco, "P[> Comp]"]), float(tabela.loc[braco, "DSR"])))
            melhora[braco].append(melhora_a_base(tabela, braco))
        print(f"(phase {fase} done in {time.perf_counter() - inicio:.0f} s)", flush=True)

    CURVES_FILE.parent.mkdir(parents=True, exist_ok=True)
    pd.concat(salvas).to_parquet(CURVES_FILE, index=False)
    print(f"\nCurves saved: {CURVES_FILE.name}")
    for braco in ARMS:
        passa = all(p >= P_MIN and d >= DSR_MIN for p, d in gate[braco])
        print(f"{braco}: improves the base in every phase (CAGR above, vol <= +{VOL_SLACK:.1%}, MDD <= {MDD_SLACK:.1%} "
              f"worse, P[> base] >= {P_MIN}): {melhora[braco]} -> {'yes' if all(melhora[braco]) else 'no'}; house gate "
              f"{[(round(p, 3), round(d, 3)) for p, d in gate[braco]]} -> "
              + ("CANDIDATE (forward fictitious account only)" if passa else "fails"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
