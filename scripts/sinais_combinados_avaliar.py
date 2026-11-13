"""
The single reading of ``docs/superpowers/specs/2026-11-13-sinais-combinados-design.md`` (committed in ``8169f7f``
before any number of the arm): the four IC-passing signals (P, X, INS, RET-3y) as one fixed 0.25 tilt on the
Composite. One phase per process, then the summary:

    python -X utf8 scripts/sinais_combinados_avaliar.py 0 > docs/sinais-combinados-fase0-2026-11-13.txt
    ... 21 > docs/sinais-combinados-fase21-2026-11-13.txt      ... 42 > docs/sinais-combinados-fase42-2026-11-13.txt
    ... scripts/sinais_combinados_avaliar.py resumo > docs/sinais-combinados-2026-11-13.txt

Anchor per phase, before the arm is read: the Composite's replay reproduces its curve after the growth age rule
(``docs/dados/pos-idade-curvas-fase{}-2026-11-13.parquet``, DECISOES 360) within 1e-9.
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
from core.indicadores_no_tempo import Fontes, notas_dos_sinais  # noqa: E402
from core.informados import insider_score, load_filings, weighted_score  # noqa: E402
from core.inovemetria.data import load  # noqa: E402
from core.inovemetria.signals import convexity_score, patent_score  # noqa: E402
from core.qubo_universo import composite_score  # noqa: E402
from core.sinais import conservadora  # noqa: E402
from core.valor import NotaDoMes  # noqa: E402
from core.views import nota_media_dos_escores  # noqa: E402
from scripts import avaliar_conservadora, avaliar_robustez, avaliar_valor  # noqa: E402
from scripts.avaliar_robustez import FASES, _acoes, referencias  # noqa: E402
from scripts.avaliar_universo_dinamico import ANOS_DE_CDI, CFG_BT  # noqa: E402
from scripts.fermionica_grade_avaliar import memoized  # noqa: E402
from scripts.inovemetria_avaliar import CURVE_TOL, PASTA_CVM, picks_profile  # noqa: E402
from scripts.qpo_montar import FOLDER, LAST_SESSION, RUN_DATE  # noqa: E402
from scripts.qubo_universo_avaliar import with_retries  # noqa: E402
from scripts.xquad_avaliar import COMPOSITE, DSR_MIN, P_MIN, metricas  # noqa: E402

N_DSR = 133                                       # the spec: 132 rows before, this arm
PESO = 0.25                                       # fixed before reading (spec)
ARM = "Comp+Extras-0.25"
DATA = FOLDER.parent
ANCORA = "pos-idade-curvas-fase{}-2026-11-13.parquet"
PASTA_MACRO = FOLDER.parent / "2026-10-21-macro"
CURVAS = "sinais-combinados-curvas-fase{}-2026-11-13.parquet"
METRICAS = "sinais-combinados-metricas-fase{}-2026-11-13.csv"


def scores(dados, mensal) -> dict[str, NotaDoMes]:
    """The Composite's score (with the growth age rule) and EXTRAS, the mean normal score of P, X, INS and RET-3y,
    each built as in its own reading."""
    balancos, _ = avaliar_valor.carregar_balancos(PASTA_CVM)
    inovacao, _ = load()
    filings, _ = load_filings()
    cdi = pd.read_parquet(PASTA_MACRO / "cdi_daily.parquet")
    juro = pd.Series(cdi["cdi_pct"].to_numpy() / 100.0, index=pd.to_datetime(cdi["date"]))
    comp = composite_score(lambda mes, candidatos: conservadora(mensal.retornos, mensal.proventos, mes, candidatos),
                           nota_media_dos_escores(*avaliar_valor.notas(balancos, mensal).values()),
                           nota_de_crescimento(balancos.fundamentos))
    ret3 = notas_dos_sinais(Fontes(dados.precos, juro, mensal.retornos, balancos.fundamentos), ["RET-3y"])["RET-3y"]
    extras = nota_media_dos_escores(patent_score(inovacao), convexity_score(mensal.retornos),
                                    insider_score(filings, inovacao.roots), ret3)
    return {"comp": memoized(comp), "extras": memoized(extras)}


def phase(fase: int) -> int:
    inicio = time.perf_counter()
    avaliar_robustez.PASTA_PRECOS = avaliar_robustez.PASTA_UNIVERSO = avaliar_valor.PASTA_UNIVERSO = FOLDER
    dados = avaliar_robustez.carregar()
    mensal = avaliar_conservadora.carregar_mensal(FOLDER)
    notas = scores(dados, mensal)
    selic = market_data.obter_selic_meta(RUN_DATE)
    cdi = with_retries(lambda: market_data.baixar_cdi_diario(ANOS_DE_CDI, hoje=RUN_DATE))
    precos = dados.precos.loc[:LAST_SESSION].iloc[fase:]
    acoes = _acoes(precos)
    salvas = pd.read_parquet(DATA / ANCORA.format(fase)).set_index("date")

    def rodar(estrategia: Estrategia, nome: str) -> ResultadoBacktest:
        return walk_forward_dinamico(acoes, dados.universo, estrategia, CFG_BT, nome, Congelamento())

    replay = rodar(estrategia_por_nota(notas["comp"], dados.setores), COMPOSITE)
    indice = replay.valores.index
    desvio = float((replay.valores / salvas["Composite"].reindex(indice) - 1.0).abs().max())
    if not desvio < CURVE_TOL:
        raise SystemExit(f"phase {fase}: the Composite's replay is {desvio:.2e} off its saved curve; the arm was not read")
    print(f"=== Phase {fase}: {indice[0]:%d/%m/%Y} to {indice[-1]:%d/%m/%Y}; DSR N {N_DSR}; anchor: Composite "
          f"{desvio:.1e} off its saved curve ===", flush=True)
    print(f"  anchor ok at {time.perf_counter() - inicio:.0f} s", file=sys.stderr, flush=True)

    braco = rodar(estrategia_por_nota(weighted_score(notas["comp"], notas["extras"], lambda mes: PESO), dados.setores), ARM)
    if not braco.valores.index.equals(indice):
        raise SystemExit(f"phase {fase}: {ARM} does not share the Composite's dates")
    curvas = {COMPOSITE: replay.valores, ARM: braco.valores,
              "BOVA11": referencias(precos, CFG_BT)["BOVA11"].valores.reindex(indice)}
    rf = market_data.serie_rf_diaria(indice, cdi, selic.valor)
    curvas["CDI"] = pd.Series(np.cumprod(1.0 + rf.to_numpy()), index=indice)
    tabela = metricas(curvas, rf, N_DSR)
    tabela.loc[ARM, "shared"] = picks_profile(braco.pesos_alvo, replay.pesos_alvo)
    tabela.loc[ARM, "names"] = float((braco.pesos_alvo > 0).sum(axis=1).mean())
    tabela["turnover"] = pd.Series({COMPOSITE: replay.turnover_anual_medio, ARM: braco.turnover_anual_medio})
    tabela.loc[ARM, "refusals"] = len(braco.recusas)
    print(tabela.to_string(float_format=lambda x: f"{x:.3f}"))
    pd.DataFrame(curvas).assign(fase=fase).rename_axis("date").reset_index().to_parquet(DATA / CURVAS.format(fase),
                                                                                        index=False)
    tabela.rename_axis("arm").reset_index().assign(fase=fase).to_csv(DATA / METRICAS.format(fase), index=False,
                                                                     lineterminator="\n")
    print(f"(phase {fase} done in {time.perf_counter() - inicio:.0f} s)", flush=True)
    return 0


def summary() -> int:
    todas = pd.concat(pd.read_csv(DATA / METRICAS.format(fase)) for fase in FASES)
    for coluna, formato in (("CAGR", "{:.1%}"), ("Vol", "{:.1%}"), ("MDD", "{:.1%}"), ("Sharpe", "{:.2f}"),
                            ("P[> Comp]", "{:.2f}"), ("DSR", "{:.3f}"), ("alpha vs Comp", "{:+.1%}"),
                            ("t vs Comp", "{:+.2f}"), ("shared", "{:.1f}"), ("turnover", "{:.1f}")):
        if coluna in todas.columns:
            print(f"\n{coluna} (phases {' / '.join(map(str, FASES))}):")
            print(todas.pivot(index="arm", columns="fase", values=coluna).to_string(float_format=formato.format,
                                                                                   na_rep="-"))
    linhas = todas[todas["arm"] == ARM]
    leituras = list(zip(linhas["P[> Comp]"].round(3), linhas["DSR"].round(3)))
    passou = all(p >= P_MIN and d >= DSR_MIN for p, d in leituras)
    print(f"\nGate (P[> Composite] >= {P_MIN} and DSR >= {DSR_MIN} in every phase): {ARM} {leituras} -> "
          f"{'PASSES (forward fictitious account only)' if passou else 'fails'}")
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in ("resumo", *map(str, FASES)):
        raise SystemExit(f"usage: sinais_combinados_avaliar.py [{' | '.join(map(str, FASES))} | resumo]")
    raise SystemExit(summary() if sys.argv[1] == "resumo" else phase(int(sys.argv[1])))
