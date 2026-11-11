"""
The second reading of the determinantal selection: review fixes, the theta ladder and the von Neumann entropy
(``docs/superpowers/specs/2026-11-11-fermionic-v2-design.md``, committed before any number of the new arms).

    ../Portfolio-Optimization/.venv/Scripts/python -X utf8 scripts/fermionica_v2_avaliar.py > docs/fermionica-v2-2026-11-11.txt

Per phase, three anchors first (the Composite's replay is its published curve; theta = 1e6 picks the Composite's
names up to score ties; the fixed code with the first reading's C reproduces DECISOES 343's CAGR), then the six arms
next to the Composite, the Conservadora, BOVA11 and the CDI.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core import market_data  # noqa: E402
from core.backtest import Congelamento, walk_forward_dinamico  # noqa: E402
from core.composto import estrategia_por_nota  # noqa: E402
from core.diversidade import effective_bets, log_det, von_neumann  # noqa: E402
from core.fermionica import covariance_correlation, estrategia_fermionica  # noqa: E402
from core.metrics import cagr, comparar_benchmark  # noqa: E402
from core.sinais import estrategia_conservadora  # noqa: E402
from scripts import avaliar_conservadora, avaliar_robustez  # noqa: E402
from scripts.avaliar_robustez import FASES, _acoes, contar_tentativas, referencias  # noqa: E402
from scripts.avaliar_universo_dinamico import ANOS_DE_CDI, CFG_BT  # noqa: E402
from scripts.fermionica_avaliar import ANCHOR_TOL, THETA_ANCHOR, anchor_ties, recording  # noqa: E402
from scripts.qpo_avaliar import CONSERVADORA, _year_table  # noqa: E402
from scripts.qpo_montar import FOLDER, LAST_SESSION, RUN_DATE  # noqa: E402
from scripts.qubo_universo_avaliar import _value_score, with_retries  # noqa: E402
from scripts.xquad_avaliar import COMPOSITE, CURVAS_51, DSR_MIN, P_MIN, metricas  # noqa: E402

THETAS = (1.0, 0.5, 0.25)
FUNCTIONALS = {"LogDet": log_det, "vonNeumann": von_neumann}
ARMS = {f"{nome} t{theta:g}": (funcional, theta) for nome, funcional in FUNCTIONALS.items() for theta in THETAS}
CAGR_343 = {0: 0.218, 21: 0.191, 42: 0.199}      # DECISOES 343, three decimals: the regression anchor
CAGR_TOL = 0.0005


def picks_profile(registro: dict, do_composite: dict) -> dict[str, float]:
    """Mean over the decisions: names shared with the Composite, pairwise correlation and effective bets."""
    comuns, correlacoes, apostas = [], [], []
    for data, (candidatos, picks, _, corr) in registro.items():
        if data not in do_composite:
            continue
        indices = list(picks)
        sub = corr[np.ix_(indices, indices)]
        k = len(indices)
        comuns.append(len({candidatos[i] for i in indices} & do_composite[data]))
        correlacoes.append((sub.sum() - k) / (k * (k - 1)))
        apostas.append(effective_bets(sub))
    return {"shared": float(np.mean(comuns)), "min shared": float(min(comuns)),
            "pair corr": float(np.mean(correlacoes)), "eff bets": float(np.mean(apostas))}


def composite_profile(registro: dict, do_composite: dict) -> dict[str, float]:
    """The same profile for the Composite's own picks, on an arm's recorded correlation matrices."""
    correlacoes, apostas = [], []
    for data, (candidatos, _, _, corr) in registro.items():
        if data not in do_composite:
            continue
        indices = [i for i, nome in enumerate(candidatos) if nome in do_composite[data]]
        sub = corr[np.ix_(indices, indices)]
        k = len(indices)
        correlacoes.append((sub.sum() - k) / (k * (k - 1)))
        apostas.append(effective_bets(sub))
    return {"shared": float("nan"), "min shared": float("nan"), "pair corr": float(np.mean(correlacoes)),
            "eff bets": float(np.mean(apostas))}


def main() -> int:
    inicio = time.perf_counter()
    avaliar_robustez.PASTA_PRECOS = avaliar_robustez.PASTA_UNIVERSO = FOLDER
    dados = avaliar_robustez.carregar()
    mensal = avaliar_conservadora.carregar_mensal(FOLDER)
    nota, _ = _value_score(mensal, com_crescimento=True)
    selic = market_data.obter_selic_meta(RUN_DATE)
    cdi = with_retries(lambda: market_data.baixar_cdi_diario(ANOS_DE_CDI, hoje=RUN_DATE))
    publicadas = pd.read_parquet(CURVAS_51)
    n = contar_tentativas() + len(ARMS)
    print(f"DSR N {n} (with these {len(ARMS)} rows); arms {list(ARMS)}; data to {LAST_SESSION:%d/%m/%Y}")

    gate: dict[str, list[tuple[float, float]]] = {nome: [] for nome in ARMS}
    for fase in FASES:
        acoes = _acoes(dados.precos.loc[:LAST_SESSION].iloc[fase:])

        def rodar(estrategia, nome: str):
            return walk_forward_dinamico(acoes, dados.universo, estrategia, CFG_BT, nome, Congelamento())

        do_composite, reg_ancora = {}, {}
        replay = rodar(recording(estrategia_por_nota(nota, dados.setores), do_composite), COMPOSITE)
        indice = replay.valores.index
        composite = publicadas.loc[publicadas["fase"] == fase].set_index("date")["Composto"].reindex(indice)
        desvio = float((replay.valores / composite - 1.0).abs().max())
        if not desvio < ANCHOR_TOL:
            raise SystemExit(f"phase {fase}: anchor 1 failed, the Composite's replay is {desvio:.2e} off its curve")
        rodar(estrategia_fermionica(nota, dados.setores, THETA_ANCHOR, registro=reg_ancora), "anchor")
        empates = anchor_ties(reg_ancora, do_composite)
        primeira = rodar(estrategia_fermionica(nota, dados.setores, 1.0, correlation=covariance_correlation), "343")
        if abs(cagr(primeira.valores) - CAGR_343[fase]) > CAGR_TOL:
            raise SystemExit(f"phase {fase}: anchor 3 failed, the first reading's CAGR is {cagr(primeira.valores):.4f}")
        print(f"\n=== Phase {fase}: {indice[0]:%d/%m/%Y} to {indice[-1]:%d/%m/%Y}; anchors: Composite replay off by "
              f"{desvio:.1e}; theta {THETA_ANCHOR:g} differs on {empates} score ties; first reading's CAGR "
              f"{cagr(primeira.valores):.4f} (DECISOES 343: {CAGR_343[fase]}) ===", flush=True)

        curvas, perfis, giros = {}, {}, {COMPOSITE: replay.turnover_anual_medio}
        for nome, (funcional, theta) in ARMS.items():
            registro: dict = {}
            braco = rodar(estrategia_fermionica(nota, dados.setores, theta, diversity=funcional, registro=registro),
                          nome)
            curvas[nome], giros[nome] = braco.valores, braco.turnover_anual_medio
            perfis[nome] = picks_profile(registro, do_composite) | {"refusals": float(len(braco.recusas))}
            if COMPOSITE not in perfis:
                perfis[COMPOSITE] = composite_profile(registro, do_composite) | {"refusals": float(len(replay.recusas))}
        curvas[COMPOSITE] = composite
        curvas[CONSERVADORA] = rodar(estrategia_conservadora(mensal.retornos, mensal.proventos, dados.setores),
                                     CONSERVADORA).valores
        curvas["BOVA11"] = referencias(dados.precos.loc[:LAST_SESSION].iloc[fase:], CFG_BT)["BOVA11"].valores \
            .reindex(indice)
        rf = market_data.serie_rf_diaria(indice, cdi, selic.valor)
        curvas["CDI"] = pd.Series(np.cumprod(1.0 + rf.to_numpy()), index=indice)
        tabela = metricas(curvas, rf, n)
        for nome in tabela.index:
            if nome != COMPOSITE:
                tabela.loc[nome, "beta vs Comp"] = comparar_benchmark(curvas[nome], curvas[COMPOSITE], rf).beta
            if nome != "BOVA11":
                tabela.loc[nome, "beta vs BOVA"] = comparar_benchmark(curvas[nome], curvas["BOVA11"], rf).beta
        tabela["turnover"] = pd.Series(giros)
        print(tabela.to_string(float_format=lambda x: f"{x:.3f}"))
        print("Picks (mean over the decisions; Composite's own picks on the same matrices):")
        print(pd.DataFrame(perfis).T.to_string(float_format=lambda x: f"{x:.3f}"))
        print("Calendar years:")
        print(_year_table({k: v for k, v in curvas.items() if k != CONSERVADORA})
              .to_string(float_format=lambda x: f"{x:.1%}"))
        for nome in ARMS:
            gate[nome].append((float(tabela.loc[nome, "P[> Comp]"]), float(tabela.loc[nome, "DSR"])))
        print(f"(phase {fase} done in {time.perf_counter() - inicio:.0f} s)", flush=True)

    print(f"\nGate per arm (P[> Composite] >= {P_MIN} and DSR >= {DSR_MIN} in every phase; DSR's V comes from the "
          f"curves of this table, see the spec's caveat):")
    for nome, leituras in gate.items():
        passa = all(p >= P_MIN and d >= DSR_MIN for p, d in leituras)
        print(f"  {nome}: {[(round(p, 3), round(d, 3)) for p, d in leituras]} -> "
              + ("CANDIDATE (family of 6: forward fictitious account only)" if passa else "fails"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
