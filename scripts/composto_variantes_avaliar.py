"""
The one reading of three design choices of the Composite found in the code review
(``docs/superpowers/specs/2026-11-11-composite-variants-design.md``, committed before any number).

    ../Portfolio-Optimization/.venv/Scripts/python -X utf8 scripts/composto_variantes_avaliar.py > docs/composto-variantes-2026-11-11.txt

Per phase: the anchor (the Composite's replay is attempt 51's published curve), then the four arms next to the
Composite, the Conservadora, BOVA11 and the CDI.
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core import market_data  # noqa: E402
from core.backtest import Congelamento, Estrategia, ResultadoBacktest, walk_forward_dinamico  # noqa: E402
from core.composto import estrategia_por_nota  # noqa: E402
from core.composto_variantes import neutral_composite_score  # noqa: E402
from core.config import ConfigBacktest  # noqa: E402
from core.crescimento_lucro import nota_de_crescimento  # noqa: E402
from core.metrics import comparar_benchmark  # noqa: E402
from core.sinais import conservadora, estrategia_conservadora  # noqa: E402
from core.valor import NotaDoMes  # noqa: E402
from core.views import nota_media_dos_escores  # noqa: E402
from scripts import avaliar_conservadora, avaliar_robustez, avaliar_valor  # noqa: E402
from scripts.avaliar_robustez import FASES, _acoes, contar_tentativas, referencias  # noqa: E402
from scripts.avaliar_universo_dinamico import ANOS_DE_CDI, CFG_BT  # noqa: E402
from scripts.qpo_avaliar import CONSERVADORA, _year_table  # noqa: E402
from scripts.qpo_montar import FOLDER, LAST_SESSION, RUN_DATE  # noqa: E402
from scripts.qubo_universo_avaliar import PASTA_CVM, _value_score, with_retries  # noqa: E402
from scripts.xquad_avaliar import COMPOSITE, CURVAS_51, DSR_MIN, P_MIN, metricas  # noqa: E402

ANCHOR_TOL = 1e-9
MONTHLY, THREE_YEARS = 21, 756
EXTRA_COST = 0.002                 # 30 bps instead of 10 bps a side, per unit of annual turnover (approximation)
CURVES_FILE = CURVAS_51.parent / "composto-variantes-curvas-2026-11-11.parquet"


@dataclass(frozen=True)
class Arm:
    cfg: ConfigBacktest
    neutral: bool = False

    @property
    def offset(self) -> int:
        """Sessions cut from the start of the prices so that the first decision is attempt 51's."""
        return CFG_BT.janela_treino - self.cfg.janela_treino


ARMS = {"Composite-21": Arm(replace(CFG_BT, janela_teste=MONTHLY)),
        "Composite-3y": Arm(replace(CFG_BT, janela_treino=THREE_YEARS)),
        "Composite-neutral": Arm(CFG_BT, neutral=True),
        "Composite-all": Arm(replace(CFG_BT, janela_teste=MONTHLY, janela_treino=THREE_YEARS), neutral=True)}


def neutral_score(mensal) -> NotaDoMes:
    """Attempt 51's three components (``scripts.qubo_universo_avaliar._value_score``) under the neutral fill."""
    avaliar_valor.PASTA_UNIVERSO = FOLDER
    balancos, _ = avaliar_valor.carregar_balancos(PASTA_CVM)
    return neutral_composite_score(
        lambda mes, candidatos: conservadora(mensal.retornos, mensal.proventos, mes, candidatos),
        nota_media_dos_escores(*avaliar_valor.notas(balancos, mensal).values()),
        nota_de_crescimento(balancos.fundamentos))


def recording(estrategia: Estrategia, registro: dict) -> Estrategia:
    """``estrategia`` keeping, per decision date, (number of eligible names, names held)."""
    def _estrategia(treino: pd.DataFrame) -> pd.Series:
        pesos = estrategia(treino)
        registro[treino.index[-1]] = (treino.shape[1], frozenset(pesos.index))
        return pesos
    return _estrategia


def profile(registro: dict, do_composite: dict, resultado: ResultadoBacktest) -> dict[str, float]:
    comuns = [len(nomes & do_composite[data][1]) for data, (_, nomes) in registro.items() if data in do_composite]
    giro = resultado.turnover_anual_medio
    return {"turnover": giro, "refusals": float(len(resultado.recusas)), "decisions": float(len(registro)),
            "eligible": float(np.mean([n for n, _ in registro.values()])),
            "shared": float(np.mean(comuns)) if comuns else float("nan"), "extra cost at 30 bps": EXTRA_COST * giro}


def main() -> int:
    inicio = time.perf_counter()
    avaliar_robustez.PASTA_PRECOS = avaliar_robustez.PASTA_UNIVERSO = FOLDER
    dados = avaliar_robustez.carregar()
    mensal = avaliar_conservadora.carregar_mensal(FOLDER)
    notas = {False: _value_score(mensal, com_crescimento=True)[0], True: neutral_score(mensal)}
    selic = market_data.obter_selic_meta(RUN_DATE)
    cdi = with_retries(lambda: market_data.baixar_cdi_diario(ANOS_DE_CDI, hoje=RUN_DATE))
    publicadas = pd.read_parquet(CURVAS_51)
    precos = dados.precos.loc[:LAST_SESSION]
    n = contar_tentativas() + len(ARMS)
    print(f"DSR N {n} (with these {len(ARMS)} rows); arms {list(ARMS)}; data to {LAST_SESSION:%d/%m/%Y}")

    gate: dict[str, list[tuple[float, float]]] = {nome: [] for nome in ARMS}
    salvas = []
    for fase in FASES:
        def rodar(estrategia: Estrategia, nome: str, arm: Arm = Arm(CFG_BT)) -> ResultadoBacktest:
            return walk_forward_dinamico(_acoes(precos.iloc[fase + arm.offset:]), dados.universo, estrategia, arm.cfg,
                                         nome, Congelamento())

        do_composite: dict = {}
        replay = rodar(recording(estrategia_por_nota(notas[False], dados.setores), do_composite), COMPOSITE)
        indice = replay.valores.index
        composite = publicadas.loc[publicadas["fase"] == fase].set_index("date")["Composto"].reindex(indice)
        desvio = float((replay.valores / composite - 1.0).abs().max())
        if not desvio < ANCHOR_TOL:
            raise SystemExit(f"phase {fase}: anchor failed, the Composite's replay is {desvio:.2e} off its curve")
        print(f"\n=== Phase {fase}: {indice[0]:%d/%m/%Y} to {indice[-1]:%d/%m/%Y}; anchor: Composite replay off by "
              f"{desvio:.1e} ===", flush=True)

        curvas, perfis = {}, {COMPOSITE: profile(do_composite, do_composite, replay)}
        for nome, arm in ARMS.items():
            registro: dict = {}
            braco = rodar(recording(estrategia_por_nota(notas[arm.neutral], dados.setores), registro), nome, arm)
            if not braco.valores.index.equals(indice):
                raise SystemExit(f"phase {fase}: {nome} does not share the Composite's dates")
            curvas[nome], perfis[nome] = braco.valores, profile(registro, do_composite, braco)
        curvas[COMPOSITE] = composite
        curvas[CONSERVADORA] = rodar(estrategia_conservadora(mensal.retornos, mensal.proventos, dados.setores),
                                     CONSERVADORA).valores
        curvas["BOVA11"] = referencias(precos.iloc[fase:], CFG_BT)["BOVA11"].valores.reindex(indice)
        rf = market_data.serie_rf_diaria(indice, cdi, selic.valor)
        curvas["CDI"] = pd.Series(np.cumprod(1.0 + rf.to_numpy()), index=indice)
        salvas.append(pd.DataFrame(curvas).assign(fase=fase).rename_axis("date").reset_index())
        tabela = metricas(curvas, rf, n)
        for nome in tabela.index:
            if nome != COMPOSITE:
                tabela.loc[nome, "beta vs Comp"] = comparar_benchmark(curvas[nome], curvas[COMPOSITE], rf).beta
            if nome != "BOVA11":
                tabela.loc[nome, "beta vs BOVA"] = comparar_benchmark(curvas[nome], curvas["BOVA11"], rf).beta
        print(tabela.to_string(float_format=lambda x: f"{x:.3f}"))
        print("Profile (turnover a year; eligible = names with the whole window per decision; shared = names in "
              "common with the Composite on its decision dates; extra cost = CAGR lost at 30 bps, approximate):")
        print(pd.DataFrame(perfis).T.to_string(float_format=lambda x: f"{x:.3f}"))
        print("Calendar years:")
        print(_year_table({k: v for k, v in curvas.items() if k != CONSERVADORA})
              .to_string(float_format=lambda x: f"{x:.1%}"))
        for nome in ARMS:
            gate[nome].append((float(tabela.loc[nome, "P[> Comp]"]), float(tabela.loc[nome, "DSR"])))
        print(f"(phase {fase} done in {time.perf_counter() - inicio:.0f} s)", flush=True)

    pd.concat(salvas).to_parquet(CURVES_FILE, index=False)
    print(f"\nCurves saved: {CURVES_FILE.name}")
    print(f"\nGate per arm (P[> Composite] >= {P_MIN} and DSR >= {DSR_MIN} in every phase):")
    for nome, leituras in gate.items():
        passa = all(p >= P_MIN and d >= DSR_MIN for p, d in leituras)
        print(f"  {nome}: {[(round(p, 3), round(d, 3)) for p, d in leituras]} -> "
              + ("CANDIDATE (family of 4: forward fictitious account only)" if passa else "fails"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
