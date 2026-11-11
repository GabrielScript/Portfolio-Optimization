"""
The one reading of the fermionic (determinantal) selection on the Composite score
(``docs/superpowers/specs/2026-11-11-fermionic-portfolio-design.md``, committed before any number).

    ../Portfolio-Optimization/.venv/Scripts/python -X utf8 scripts/fermionica_avaliar.py > docs/fermionica-2026-11-11.txt

Per phase: the anchor first (the Composite's own strategy must be attempt 51's published curve, and theta = 1e6 must
pick the Composite's names or, on a score tie, names of the same total score), then the arm (theta = 1)
next to the Composite, BOVA11 and the CDI: metrics, calendar years and the picks' profile.
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
from core.fermionica import THETA, estrategia_fermionica  # noqa: E402
from scripts import avaliar_conservadora, avaliar_robustez  # noqa: E402
from scripts.avaliar_robustez import FASES, _acoes, contar_tentativas, referencias  # noqa: E402
from scripts.avaliar_universo_dinamico import ANOS_DE_CDI, CFG_BT  # noqa: E402
from scripts.qpo_avaliar import _year_table  # noqa: E402
from scripts.qpo_montar import FOLDER, LAST_SESSION, RUN_DATE  # noqa: E402
from scripts.qubo_universo_avaliar import _value_score, with_retries  # noqa: E402
from scripts.xquad_avaliar import COMPOSITE, CURVAS_51, DSR_MIN, P_MIN, metricas  # noqa: E402

ARM = "Fermionic"
THETA_ANCHOR = 1e6
ANCHOR_TOL = 1e-9


def recording(estrategia, picks: dict):
    """``estrategia`` keeping each decision's names."""
    def _estrategia(treino: pd.DataFrame) -> pd.Series:
        pesos = estrategia(treino)
        picks[treino.index[-1]] = frozenset(pesos.index)
        return pesos
    return _estrategia


def anchor_ties(reg_ancora: dict, do_composite: dict) -> int:
    """Decisions where theta = 1e6 differs from the Composite on a score tie; SystemExit on any other difference."""
    empates = 0
    for data, (candidatos, picks, z, _) in reg_ancora.items():
        nomes = {candidatos[i] for i in picks}
        if data not in do_composite or nomes == do_composite[data]:
            continue
        posicao = {nome: i for i, nome in enumerate(candidatos)}
        diferenca = z[list(picks)].sum() - z[[posicao[nome] for nome in do_composite[data]]].sum()
        if abs(diferenca) > ANCHOR_TOL:
            raise SystemExit(f"{data:%Y-%m-%d}: anchor failed, theta {THETA_ANCHOR:g} picks differ by {diferenca:.2e}")
        empates += 1
    return empates


def picks_profile(registro: dict, do_composite: dict) -> str:
    """Names shared with the Composite's picks, and both sets' mean pairwise correlation and log det."""
    comuns, linhas = [], {"Fermionic": [], "Composite": []}
    for data, (candidatos, picks, _, corr) in registro.items():
        if data not in do_composite:
            continue
        de_51 = [i for i, nome in enumerate(candidatos) if nome in do_composite[data]]
        comuns.append(len(set(picks) & set(de_51)))
        for nome, conjunto in (("Fermionic", list(picks)), ("Composite", de_51)):
            sub = corr[np.ix_(conjunto, conjunto)]
            k = len(conjunto)
            linhas[nome].append(((sub.sum() - k) / (k * (k - 1)), np.linalg.slogdet(sub)[1]))
    texto = f"{len(comuns)} decisions; names shared with the Composite: mean {np.mean(comuns):.1f} of 15 "             f"(min {min(comuns)}, max {max(comuns)})"
    for nome, valores in linhas.items():
        media = np.mean(valores, axis=0)
        texto += f"; {nome} picks: mean pairwise correlation {media[0]:.3f}, mean log det {media[1]:.2f}"
    return texto


def main() -> int:
    inicio = time.perf_counter()
    avaliar_robustez.PASTA_PRECOS = avaliar_robustez.PASTA_UNIVERSO = FOLDER
    dados = avaliar_robustez.carregar()
    mensal = avaliar_conservadora.carregar_mensal(FOLDER)
    nota, _ = _value_score(mensal, com_crescimento=True)
    selic = market_data.obter_selic_meta(RUN_DATE)
    cdi = with_retries(lambda: market_data.baixar_cdi_diario(ANOS_DE_CDI, hoje=RUN_DATE))
    publicadas = pd.read_parquet(CURVAS_51)
    n = contar_tentativas() + 1
    print(f"DSR N {n} (with this row); theta {THETA}; data to {LAST_SESSION:%d/%m/%Y}")

    gate = []
    for fase in FASES:
        acoes = _acoes(dados.precos.loc[:LAST_SESSION].iloc[fase:])

        def rodar(estrategia, nome: str):
            return walk_forward_dinamico(acoes, dados.universo, estrategia, CFG_BT, nome, Congelamento())

        do_composite, reg_ancora, reg_braco = {}, {}, {}
        referencia = rodar(recording(estrategia_por_nota(nota, dados.setores), do_composite), COMPOSITE).valores
        composite = publicadas.loc[publicadas["fase"] == fase].set_index("date")["Composto"].reindex(referencia.index)
        desvio = float((referencia / composite - 1.0).abs().max())
        if not desvio < ANCHOR_TOL:
            raise SystemExit(f"phase {fase}: anchor failed, the Composite's replay is {desvio:.2e} off its curve")
        rodar(estrategia_fermionica(nota, dados.setores, THETA_ANCHOR, registro=reg_ancora), "anchor")
        empates = anchor_ties(reg_ancora, do_composite)
        braco = rodar(estrategia_fermionica(nota, dados.setores, THETA, registro=reg_braco), ARM)
        indice = braco.valores.index
        curvas = {ARM: braco.valores, COMPOSITE: composite,
                  "BOVA11": referencias(dados.precos.loc[:LAST_SESSION].iloc[fase:], CFG_BT)["BOVA11"].valores
                  .reindex(indice)}
        rf = market_data.serie_rf_diaria(indice, cdi, selic.valor)
        curvas["CDI"] = pd.Series(np.cumprod(1.0 + rf.to_numpy()), index=indice)
        tabela = metricas(curvas, rf, n)
        gate.append((float(tabela.loc[ARM, "P[> Comp]"]), float(tabela.loc[ARM, "DSR"])))
        print(f"\n=== Phase {fase}: {indice[0]:%d/%m/%Y} to {indice[-1]:%d/%m/%Y}; Composite replay off by {desvio:.1e}, "
              f"theta {THETA_ANCHOR:g} differs on {empates} score ties; "
              f"refusals {len(braco.recusas)}; annual turnover {braco.turnover_anual_medio:.2f} ===")
        print(picks_profile(reg_braco, do_composite))
        print(tabela.to_string(float_format=lambda x: f"{x:.3f}"))
        print("Calendar years:")
        print(_year_table(curvas).to_string(float_format=lambda x: f"{x:.1%}"))
        print(f"(phase {fase} done in {time.perf_counter() - inicio:.0f} s)", flush=True)

    passa = all(p >= P_MIN and d >= DSR_MIN for p, d in gate)
    print(f"\nGate (P[{ARM} > Composite] >= {P_MIN} and DSR >= {DSR_MIN} in every phase): "
          f"{[(round(p, 3), round(d, 3)) for p, d in gate]} -> "
          + ("CANDIDATE: pre-register a forward fictitious account" if passa else
             "FAILS: the fermionic selection line is closed"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
