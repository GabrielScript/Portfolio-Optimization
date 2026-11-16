"""
The one reading of the Composite in three staggered thirds
(``docs/superpowers/specs/2026-11-16-composto-tercos-design.md``, committed before any number).

    ../Portfolio-Optimization/.venv/Scripts/python -X utf8 scripts/composto_tercos_avaliar.py > docs/composto-tercos-2026-11-16.txt

Replays the Composite's phases 0 / 21 / 42 with today's code, builds the book from them (``core.composto_tercos``)
and reads it against the Composite of each phase on that phase's dates, with the Conservadora, BOVA11 and the CDI.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core import market_data  # noqa: E402
from core.backtest import Congelamento, Estrategia, ResultadoBacktest, walk_forward_dinamico  # noqa: E402
from core.composto import estrategia_por_nota  # noqa: E402
from core.composto_tercos import livro_em_partes  # noqa: E402
from core.metrics import comparar_benchmark  # noqa: E402
from core.sinais import estrategia_conservadora  # noqa: E402
from scripts import avaliar_conservadora, avaliar_robustez  # noqa: E402
from scripts.avaliar_robustez import FASES, _acoes, referencias  # noqa: E402
from scripts.avaliar_universo_dinamico import ANOS_DE_CDI, CFG_BT  # noqa: E402
from scripts.qpo_avaliar import CONSERVADORA, _year_table  # noqa: E402
from scripts.qpo_montar import FOLDER, LAST_SESSION, RUN_DATE  # noqa: E402
from scripts.qubo_universo_avaliar import _value_score, with_retries  # noqa: E402
from scripts.xquad_avaliar import COMPOSITE, CURVAS_51, DSR_MIN, P_MIN, metricas  # noqa: E402

ARM = "Composite-tercos"
N_DSR = 190                        # 189 (DECISOES 370) + this arm
CURVES_FILE = CURVAS_51.parent / "composto-tercos-curvas-2026-11-16.parquet"


def main() -> int:
    inicio = time.perf_counter()
    avaliar_robustez.PASTA_PRECOS = avaliar_robustez.PASTA_UNIVERSO = FOLDER
    dados = avaliar_robustez.carregar()
    mensal = avaliar_conservadora.carregar_mensal(FOLDER)
    nota = _value_score(mensal, com_crescimento=True)[0]
    selic = market_data.obter_selic_meta(RUN_DATE)
    cdi = with_retries(lambda: market_data.baixar_cdi_diario(ANOS_DE_CDI, hoje=RUN_DATE))
    publicadas = pd.read_parquet(CURVAS_51)
    precos = dados.precos.loc[:LAST_SESSION]
    print(f"DSR N {N_DSR}; arm {ARM}; data to {LAST_SESSION:%d/%m/%Y}")

    def rodar(estrategia: Estrategia, nome: str, fase: int) -> ResultadoBacktest:
        return walk_forward_dinamico(_acoes(precos.iloc[fase:]), dados.universo, estrategia, CFG_BT, nome,
                                     Congelamento())

    replays: dict[int, ResultadoBacktest] = {}
    for fase in FASES:
        replays[fase] = rodar(estrategia_por_nota(nota, dados.setores), COMPOSITE, fase)
        indice = replays[fase].valores.index
        salva = publicadas.loc[publicadas["fase"] == fase].set_index("date")["Composto"].reindex(indice)
        desvio = float((replays[fase].valores / salva - 1.0).abs().max())
        print(f"phase {fase}: Composite replay {indice[0]:%d/%m/%Y} to {indice[-1]:%d/%m/%Y}, off attempt 51's saved "
              f"curve by {desvio:.1e} (printed, not enforced); turnover {replays[fase].turnover_anual_medio:.3f} a year"
              f" ({time.perf_counter() - inicio:.0f} s)", flush=True)

    indice_livro = replays[FASES[0]].valores.index
    rf_livro = market_data.serie_rf_diaria(indice_livro, cdi, selic.valor)
    livro = livro_em_partes([replays[f].valores for f in FASES],
                            pd.Series(np.cumprod(1.0 + rf_livro.to_numpy()), index=indice_livro))
    giro = float(np.mean([r.turnover_anual_medio for r in replays.values()]))
    print(f"\nBook {ARM}: {indice_livro[0]:%d/%m/%Y} to {indice_livro[-1]:%d/%m/%Y}; turnover (mean of the sleeves') "
          f"{giro:.3f} a year")

    gate: list[tuple[float, float]] = []
    cagrs: dict[str, float] = {}
    salvas = []
    for fase in FASES:
        indice = replays[fase].valores.index
        curvas = {ARM: livro.reindex(indice), COMPOSITE: replays[fase].valores}
        if curvas[ARM].isna().any():
            raise SystemExit(f"phase {fase}: the book misses some of the phase's dates")
        curvas[CONSERVADORA] = rodar(estrategia_conservadora(mensal.retornos, mensal.proventos, dados.setores),
                                     CONSERVADORA, fase).valores
        curvas["BOVA11"] = referencias(precos.iloc[fase:], CFG_BT)["BOVA11"].valores.reindex(indice)
        rf = market_data.serie_rf_diaria(indice, cdi, selic.valor)
        curvas["CDI"] = pd.Series(np.cumprod(1.0 + rf.to_numpy()), index=indice)
        salvas.append(pd.DataFrame(curvas).assign(fase=fase).rename_axis("date").reset_index())
        tabela = metricas(curvas, rf, N_DSR)
        for nome in tabela.index:
            if nome != COMPOSITE:
                tabela.loc[nome, "beta vs Comp"] = comparar_benchmark(curvas[nome], curvas[COMPOSITE], rf).beta
        print(f"\n=== Phase {fase}: {indice[0]:%d/%m/%Y} to {indice[-1]:%d/%m/%Y} (book restricted to these dates) ===")
        print(tabela.to_string(float_format=lambda x: f"{x:.3f}"))
        print("Calendar years:")
        print(_year_table({k: v for k, v in curvas.items() if k != CONSERVADORA})
              .to_string(float_format=lambda x: f"{x:.1%}"))
        gate.append((float(tabela.loc[ARM, "P[> Comp]"]), float(tabela.loc[ARM, "DSR"])))
        cagrs[f"Composite phase {fase}"] = float(tabela.loc[COMPOSITE, "CAGR"])
        cagrs[f"{ARM} on phase {fase}'s dates"] = float(tabela.loc[ARM, "CAGR"])
        print(f"(phase {fase} done in {time.perf_counter() - inicio:.0f} s)", flush=True)

    pd.concat(salvas).to_parquet(CURVES_FILE, index=False)
    print(f"\nCurves saved: {CURVES_FILE.name}")
    fases_cagr = [cagrs[f"Composite phase {f}"] for f in FASES]
    print("\nDescriptive (no gate): CAGR")
    for nome, valor in cagrs.items():
        print(f"  {nome}: {valor:.2%}")
    print(f"  Composite phases: mean {np.mean(fases_cagr):.2%}, spread {max(fases_cagr) - min(fases_cagr):.2%}")
    passa = all(p >= P_MIN and d >= DSR_MIN for p, d in gate)
    print(f"\nGate (P[> Composite] >= {P_MIN} and DSR >= {DSR_MIN} in every phase): {ARM} "
          f"{[(round(p, 3), round(d, 3)) for p, d in gate]} -> "
          + ("CANDIDATE (forward fictitious account only)" if passa else "fails"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
