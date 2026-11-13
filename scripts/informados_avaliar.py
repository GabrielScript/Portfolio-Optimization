"""
Stage 2 of ``docs/superpowers/specs/2026-11-12-informados-design.md`` (committed in ``fd57a77`` before any
return-based number; stage 1 read in ``docs/informados-ic-2026-11-12.txt``: INS passes): three arms against the
15-name Composite, read once. One phase per process, then the summary:

    python -X utf8 scripts/informados_avaliar.py 0 > docs/informados-fase0-2026-11-12.txt
    ... 21 > docs/informados-fase21-2026-11-12.txt        ... 42 > docs/informados-fase42-2026-11-12.txt
    ... scripts/informados_avaliar.py resumo > docs/informados-2026-11-12.txt

Anchors per phase, before any arm is read: the Composite's replay and LogDet-15 t0.5 reproduce their curves saved
by the size grid of DECISOES 350 (1e-9).
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
from core.fermionica import estrategia_fermionica  # noqa: E402
from core.informados import earned_weight, insider_score, load_filings, mean_of, weighted_score  # noqa: E402
from core.inovemetria.data import load  # noqa: E402
from core.inovemetria.signals import patent_score  # noqa: E402
from core.qubo_universo import composite_score  # noqa: E402
from core.sinais import conservadora  # noqa: E402
from core.valor import NotaDoMes  # noqa: E402
from core.views import nota_media_dos_escores  # noqa: E402
from scripts import avaliar_conservadora, avaliar_robustez, avaliar_valor  # noqa: E402
from scripts.avaliar_robustez import FASES, _acoes, referencias  # noqa: E402
from scripts.avaliar_universo_dinamico import ANOS_DE_CDI, CFG_BT  # noqa: E402
from scripts.fermionica_grade_avaliar import memoized  # noqa: E402
from scripts.inovemetria_avaliar import CURVE_TOL, DATA, GRID, PASTA_CVM, picks_profile  # noqa: E402
from scripts.qpo_montar import FOLDER, LAST_SESSION, RUN_DATE  # noqa: E402
from scripts.qubo_universo_avaliar import with_retries  # noqa: E402
from scripts.xquad_avaliar import COMPOSITE, DSR_MIN, P_MIN, metricas  # noqa: E402

N_DSR = 126                           # the spec's: 121 after DRL v3, the two MARS-R rows, these three
THETAS = (0.25, 0.5, 0.75, 1.0)       # the ladder of DECISOES 350
ANCHOR = "LogDet-15 t0.5"
PAT, MEAN, INS = "Comp+Pat-w", "Fermi-mean", "Comp+Ins-w"
EXTRAS = {PAT: "patents", INS: "insiders"}


def scores(dados, mensal) -> dict[str, NotaDoMes]:
    """The Composite's score (attempt 51), the patent family and the insider ratio, each memoized."""
    balancos, _ = avaliar_valor.carregar_balancos(PASTA_CVM)
    inovacao, _ = load()
    filings, _ = load_filings()
    cons = lambda mes, candidatos: conservadora(mensal.retornos, mensal.proventos, mes, candidatos)  # noqa: E731
    comp = composite_score(cons, nota_media_dos_escores(*avaliar_valor.notas(balancos, mensal).values()),
                           nota_de_crescimento(balancos.fundamentos))
    return {"comp": memoized(comp), "patents": memoized(patent_score(inovacao)),
            "insiders": memoized(insider_score(filings, inovacao.roots))}


def weights(notas: dict[str, NotaDoMes], dados, mensal) -> dict[str, pd.Series]:
    """lambda of each decision month for each extra score: from the monthly ICs of the months before it."""
    ics = {nome: avaliar_valor.ics_e_cobertura(nota, mensal, dados.universo)[0] for nome, nota in notas.items()}
    meses = pd.date_range(avaliar_valor.PERIODO_DO_IC[0], LAST_SESSION, freq="MS")
    return {nome: pd.Series({mes: earned_weight(ics[nome], ics["comp"], mes) for mes in meses})
            for nome in EXTRAS.values()}


def arms(notas: dict, pesos: dict[str, pd.Series], setores: dict) -> dict[str, Estrategia]:
    """The anchor first, then the spec's arms in order."""
    def com_peso(extra: str) -> Estrategia:
        return estrategia_por_nota(weighted_score(notas["comp"], notas[extra], lambda mes: float(pesos[extra].get(mes, 0.0))),
                                   setores)
    escada = [estrategia_fermionica(notas["comp"], setores, theta) for theta in THETAS]
    return {ANCHOR: escada[THETAS.index(0.5)], PAT: com_peso("patents"), MEAN: mean_of(escada), INS: com_peso("insiders")}


def phase(fase: int) -> int:
    inicio = time.perf_counter()
    avaliar_robustez.PASTA_PRECOS = avaliar_robustez.PASTA_UNIVERSO = avaliar_valor.PASTA_UNIVERSO = FOLDER
    dados = avaliar_robustez.carregar()
    mensal = avaliar_conservadora.carregar_mensal(FOLDER)
    notas = scores(dados, mensal)
    pesos = weights(notas, dados, mensal)
    selic = market_data.obter_selic_meta(RUN_DATE)
    cdi = with_retries(lambda: market_data.baixar_cdi_diario(ANOS_DE_CDI, hoje=RUN_DATE))
    precos = dados.precos.loc[:LAST_SESSION].iloc[fase:]
    acoes = _acoes(precos)
    salvas = pd.read_parquet(DATA / GRID.format(fase)).set_index("date")

    def rodar(estrategia: Estrategia, nome: str) -> ResultadoBacktest:
        return walk_forward_dinamico(acoes, dados.universo, estrategia, CFG_BT, nome, Congelamento())

    def ancora(resultado: ResultadoBacktest, coluna: str) -> float:
        desvio = float((resultado.valores / salvas[coluna].reindex(resultado.valores.index) - 1.0).abs().max())
        if not desvio < CURVE_TOL:
            raise SystemExit(f"phase {fase}: anchor {coluna} failed, {desvio:.2e} off its saved curve")
        return desvio

    replay = rodar(estrategia_por_nota(notas["comp"], dados.setores), COMPOSITE)
    indice = replay.valores.index
    desvio_comp = ancora(replay, COMPOSITE)
    bracos = arms(notas, pesos, dados.setores)
    curvas, giros, recusas = {COMPOSITE: replay.valores}, {COMPOSITE: replay.turnover_anual_medio}, {}
    alvos = {COMPOSITE: replay.pesos_alvo}
    for posicao, (nome, estrategia) in enumerate(bracos.items()):
        resultado = rodar(estrategia, nome)
        if not resultado.valores.index.equals(indice):
            raise SystemExit(f"phase {fase}: {nome} does not share the Composite's dates")
        if posicao == 0:
            desvio = ancora(resultado, ANCHOR)
            print(f"=== Phase {fase}: {indice[0]:%d/%m/%Y} to {indice[-1]:%d/%m/%Y}; DSR N {N_DSR}; anchors: "
                  f"Composite {desvio_comp:.1e}, {ANCHOR} {desvio:.1e} off their saved curves ===", flush=True)
        curvas[nome], giros[nome], recusas[nome] = resultado.valores, resultado.turnover_anual_medio, len(resultado.recusas)
        alvos[nome] = resultado.pesos_alvo
        print(f"  [{posicao + 1}/{len(bracos)}] {nome}: {time.perf_counter() - inicio:.0f} s", file=sys.stderr, flush=True)

    curvas["BOVA11"] = referencias(precos, CFG_BT)["BOVA11"].valores.reindex(indice)
    rf = market_data.serie_rf_diaria(indice, cdi, selic.valor)
    curvas["CDI"] = pd.Series(np.cumprod(1.0 + rf.to_numpy()), index=indice)
    tabela = metricas(curvas, rf, N_DSR)
    for nome in bracos:
        tabela.loc[nome, "shared"] = picks_profile(alvos[nome], alvos[COMPOSITE])
        tabela.loc[nome, "names"] = float((alvos[nome] > 0).sum(axis=1).mean())
    tabela["turnover"], tabela["refusals"] = pd.Series(giros), pd.Series(recusas)
    print(tabela.to_string(float_format=lambda x: f"{x:.3f}"))
    for braco, extra in EXTRAS.items():
        serie = pesos[extra]
        ativo = serie[serie > 0]
        print(f"{braco}: lambda > 0 in {len(ativo)} of {len(serie)} months"
              + (f", from {ativo.index[0]:%Y-%m}; mean {ativo.mean():.2f}, min {ativo.min():.2f}, max "
                 f"{ativo.max():.2f}, last {serie.iloc[-1]:.2f}" if len(ativo) else ""))
    pd.DataFrame(curvas).assign(fase=fase).rename_axis("date").reset_index().to_parquet(
        DATA / f"informados-curvas-fase{fase}-2026-11-12.parquet", index=False)
    tabela.rename_axis("arm").reset_index().assign(fase=fase).to_csv(
        DATA / f"informados-metricas-fase{fase}-2026-11-12.csv", index=False, lineterminator="\n")
    print(f"(phase {fase} done in {time.perf_counter() - inicio:.0f} s)", flush=True)
    return 0


def summary() -> int:
    todas = pd.concat(pd.read_csv(DATA / f"informados-metricas-fase{fase}-2026-11-12.csv") for fase in FASES)
    for coluna, formato in (("CAGR", "{:.1%}"), ("Vol", "{:.1%}"), ("MDD", "{:.1%}"), ("Sharpe", "{:.2f}"),
                            ("P[> Comp]", "{:.2f}"), ("DSR", "{:.3f}"), ("alpha vs Comp", "{:+.1%}"),
                            ("t vs Comp", "{:+.2f}"), ("shared", "{:.1f}"), ("names", "{:.1f}"),
                            ("turnover", "{:.1f}")):
        print(f"\n{coluna} (phases {' / '.join(map(str, FASES))}):")
        print(todas.pivot(index="arm", columns="fase", values=coluna)
              .loc[lambda t: [a for a in todas["arm"].unique() if a in t.index]]
              .to_string(float_format=formato.format, na_rep="-"))
    print(f"\nGate per arm (P[> Composite] >= {P_MIN} and DSR >= {DSR_MIN} in every phase):")
    passaram = []
    for nome, linhas in todas.groupby("arm", sort=False):
        if nome in (COMPOSITE, "BOVA11", "CDI", ANCHOR):
            continue
        leituras = list(zip(linhas["P[> Comp]"].round(3), linhas["DSR"].round(3)))
        if all(p >= P_MIN and d >= DSR_MIN for p, d in leituras):
            passaram.append(nome)
        print(f"  {nome}: {leituras} -> {'CANDIDATE (forward fictitious account only)' if nome in passaram else 'fails'}")
    print(f"\nArms passing the gate: {passaram or 'none'}")
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in ("resumo", *map(str, FASES)):
        raise SystemExit(f"usage: informados_avaliar.py [{' | '.join(map(str, FASES))} | resumo]")
    raise SystemExit(summary() if sys.argv[1] == "resumo" else phase(int(sys.argv[1])))
