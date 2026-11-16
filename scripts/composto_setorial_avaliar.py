"""
The one reading of the Composite with sector-relative value and growth
(``docs/superpowers/specs/2026-11-16-composto-setorial-design.md``, committed before any number).

    ../Portfolio-Optimization/.venv/Scripts/python -X utf8 scripts/composto_setorial_avaliar.py > docs/composto-setorial-2026-11-16.txt

Per phase: the Composite replayed by today's code (its gap to attempt 51's saved curve printed, not enforced), the arm,
the Conservadora, BOVA11 and the CDI, the gate. Then the descriptive IC of each leg, raw and sector-relative.
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
from core.crescimento_lucro import nota_de_crescimento  # noqa: E402
from core.setorial import dentro_do_setor  # noqa: E402
from core.poder import ic_por_mes  # noqa: E402
from core.qubo_universo import composite_score  # noqa: E402
from core.sinais import conservadora, estrategia_conservadora  # noqa: E402
from core.views import nota_media_dos_escores  # noqa: E402
from scripts import avaliar_conservadora, avaliar_robustez, avaliar_valor  # noqa: E402
from scripts.avaliar_conservadora import retornos_no_universo  # noqa: E402
from scripts.avaliar_robustez import FASES, _acoes, referencias  # noqa: E402
from scripts.avaliar_universo_dinamico import ANOS_DE_CDI, CFG_BT  # noqa: E402
from scripts.composto_variantes_avaliar import profile, recording  # noqa: E402
from scripts.qpo_avaliar import CONSERVADORA, _year_table  # noqa: E402
from scripts.qpo_montar import FOLDER, LAST_SESSION, RUN_DATE  # noqa: E402
from scripts.qubo_universo_avaliar import PASTA_CVM, with_retries  # noqa: E402
from scripts.xquad_avaliar import COMPOSITE, CURVAS_51, DSR_MIN, P_MIN, metricas  # noqa: E402

ARM = "Composite-setorial"
N_DSR = 195                        # 194 (DECISOES 375) + this arm
NW_LAGS = 2
CURVES_FILE = CURVAS_51.parent / "composto-setorial-curvas-2026-11-16.parquet"


def pernas(mensal, setores) -> dict[str, dict[str, object]]:
    """The value and growth legs, raw (today's) and sector-relative (the arm's)."""
    avaliar_valor.PASTA_UNIVERSO = FOLDER
    balancos, _ = avaliar_valor.carregar_balancos(PASTA_CVM)
    valor = list(avaliar_valor.notas(balancos, mensal).values())
    crescimento = nota_de_crescimento(balancos.fundamentos)
    return {"raw": {"value": nota_media_dos_escores(*valor), "growth": crescimento},
            "sector": {"value": nota_media_dos_escores(*(dentro_do_setor(n, setores) for n in valor)),
                       "growth": dentro_do_setor(crescimento, setores)}}


def scores(mensal, legs) -> dict[str, object]:
    """The Composite's score and the arm's: the same Conservadora leg, value and growth raw or sector-relative."""
    def com(tipo: str):
        return composite_score(lambda mes, candidatos: conservadora(mensal.retornos, mensal.proventos, mes, candidatos),
                               legs[tipo]["value"], legs[tipo]["growth"])
    return {COMPOSITE: com("raw"), ARM: com("sector")}


def t_newey_west(serie: pd.Series, lags: int = NW_LAGS) -> float:
    x = serie.dropna().to_numpy()
    d = x - x.mean()
    variancia = d @ d / len(x)
    for lag in range(1, lags + 1):
        variancia += 2.0 * (1.0 - lag / (lags + 1)) * (d[lag:] @ d[:-lag]) / len(x)
    return float(x.mean() / np.sqrt(variancia / len(x)))


def ic_descritivo(mensal, legs) -> None:
    universo = pd.read_parquet(FOLDER / "universe_monthly.parquet")
    elegiveis = universo["security"].groupby(pd.to_datetime(universo["month"])).apply(list)
    meses = pd.date_range("2012-01-01", LAST_SESSION.to_period("M").to_timestamp(), freq="MS")
    retornos = retornos_no_universo(mensal.retornos, universo, meses)
    print(f"
Descriptive (no gate): monthly Spearman IC of each leg against the month's total return, "
          f"{meses[0]:%Y-%m} to {meses[-1]:%Y-%m}, Newey-West t ({NW_LAGS} lags)")
    for perna in ("value", "growth"):
        ics = {}
        for tipo in ("raw", "sector"):
            painel = pd.DataFrame({mes: legs[tipo][perna](mes, elegiveis.get(mes, [])) for mes in meses}).T
            ics[tipo] = ic_por_mes(painel.reindex(columns=mensal.retornos.columns), retornos).dropna()
            print(f"  {perna} {tipo}: IC {ics[tipo].mean():.4f}, t {t_newey_west(ics[tipo]):.2f}, months {len(ics[tipo])}")
        diferenca = (ics["sector"] - ics["raw"]).dropna()
        print(f"  {perna} sector - raw: {diferenca.mean():+.4f}, t {t_newey_west(diferenca):+.2f}")


def main() -> int:
    inicio = time.perf_counter()
    avaliar_robustez.PASTA_PRECOS = avaliar_robustez.PASTA_UNIVERSO = FOLDER
    dados = avaliar_robustez.carregar()
    mensal = avaliar_conservadora.carregar_mensal(FOLDER)
    legs = pernas(mensal, dados.setores)
    notas = scores(mensal, legs)
    selic = market_data.obter_selic_meta(RUN_DATE)
    cdi = with_retries(lambda: market_data.baixar_cdi_diario(ANOS_DE_CDI, hoje=RUN_DATE))
    publicadas = pd.read_parquet(CURVAS_51)
    precos = dados.precos.loc[:LAST_SESSION]
    print(f"DSR N {N_DSR}; arm {ARM}; data to {LAST_SESSION:%d/%m/%Y}")

    gate: list[tuple[float, float]] = []
    salvas = []
    for fase in FASES:
        def rodar(estrategia: Estrategia, nome: str) -> ResultadoBacktest:
            return walk_forward_dinamico(_acoes(precos.iloc[fase:]), dados.universo, estrategia, CFG_BT, nome,
                                         Congelamento())

        do_composite: dict = {}
        replay = rodar(recording(estrategia_por_nota(notas[COMPOSITE], dados.setores), do_composite), COMPOSITE)
        indice = replay.valores.index
        salva = publicadas.loc[publicadas["fase"] == fase].set_index("date")["Composto"].reindex(indice)
        desvio = float((replay.valores / salva - 1.0).abs().max())
        registro: dict = {}
        braco = rodar(recording(estrategia_por_nota(notas[ARM], dados.setores), registro), ARM)
        if not braco.valores.index.equals(indice):
            raise SystemExit(f"phase {fase}: the arm does not share the Composite's dates")
        print(f"\n=== Phase {fase}: {indice[0]:%d/%m/%Y} to {indice[-1]:%d/%m/%Y}; Composite replay off attempt 51's "
              f"saved curve by {desvio:.1e} (printed, not enforced) ===", flush=True)
        curvas = {ARM: braco.valores, COMPOSITE: replay.valores}
        curvas[CONSERVADORA] = rodar(estrategia_conservadora(mensal.retornos, mensal.proventos, dados.setores),
                                     CONSERVADORA).valores
        curvas["BOVA11"] = referencias(precos.iloc[fase:], CFG_BT)["BOVA11"].valores.reindex(indice)
        rf = market_data.serie_rf_diaria(indice, cdi, selic.valor)
        curvas["CDI"] = pd.Series(np.cumprod(1.0 + rf.to_numpy()), index=indice)
        salvas.append(pd.DataFrame(curvas).assign(fase=fase).rename_axis("date").reset_index())
        tabela = metricas(curvas, rf, N_DSR)
        print(tabela.to_string(float_format=lambda x: f"{x:.3f}"))
        perfis = {COMPOSITE: profile(do_composite, do_composite, replay), ARM: profile(registro, do_composite, braco)}
        print("Profile (turnover a year; shared = names in common with the Composite on its decision dates):")
        print(pd.DataFrame(perfis).T.to_string(float_format=lambda x: f"{x:.3f}"))
        print("Calendar years:")
        print(_year_table({k: v for k, v in curvas.items() if k != CONSERVADORA})
              .to_string(float_format=lambda x: f"{x:.1%}"))
        gate.append((float(tabela.loc[ARM, "P[> Comp]"]), float(tabela.loc[ARM, "DSR"])))
        print(f"(phase {fase} done in {time.perf_counter() - inicio:.0f} s)", flush=True)

    pd.concat(salvas).to_parquet(CURVES_FILE, index=False)
    print(f"\nCurves saved: {CURVES_FILE.name}")
    ic_descritivo(mensal, legs)
    passa = all(p >= P_MIN and d >= DSR_MIN for p, d in gate)
    print(f"\nGate (P[> Composite] >= {P_MIN} and DSR >= {DSR_MIN} in every phase): {ARM} "
          f"{[(round(p, 3), round(d, 3)) for p, d in gate]} -> "
          + ("CANDIDATE (forward fictitious account only)" if passa else "fails"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
