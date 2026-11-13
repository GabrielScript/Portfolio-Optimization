"""
Stage 2 of the Inovemetria reading (``docs/superpowers/specs/2026-11-12-inovemetria-design.md``, committed in
``b7aad73`` before any return-based number; stage 1 read in ``docs/inovemetria-ic-2026-11-12.txt``): eight arms
against the 15-name Composite, read once. One phase per process, then the summary:

    python -X utf8 scripts/inovemetria_avaliar.py 0 > docs/inovemetria-fase0-2026-11-12.txt
    ... 21 > docs/inovemetria-fase21-2026-11-12.txt        ... 42 > docs/inovemetria-fase42-2026-11-12.txt
    ... scripts/inovemetria_avaliar.py resumo > docs/inovemetria-2026-11-12.txt

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
from core.diversidade import von_neumann  # noqa: E402
from core.fermionica import estrategia_fermionica  # noqa: E402
from core.inovemetria.data import load  # noqa: E402
from core.inovemetria.kernel import mixed_correlation  # noqa: E402
from core.inovemetria.signals import WIPO_FIELDS, families, field_counts  # noqa: E402
from core.metrics import cagr, comparar_benchmark  # noqa: E402
from core.qubo_universo import composite_score  # noqa: E402
from core.sinais import conservadora  # noqa: E402
from core.valor import NotaDoMes  # noqa: E402
from core.views import nota_media_dos_escores  # noqa: E402
from scripts import avaliar_conservadora, avaliar_robustez, avaliar_valor  # noqa: E402
from scripts.avaliar_robustez import FASES, _acoes, contar_tentativas, referencias  # noqa: E402
from scripts.avaliar_universo_dinamico import ANOS_DE_CDI, CFG_BT  # noqa: E402
from scripts.fermionica_grade_avaliar import memoized  # noqa: E402
from scripts.qpo_montar import FOLDER, LAST_SESSION, RUN_DATE  # noqa: E402
from scripts.qubo_universo_avaliar import with_retries  # noqa: E402
from scripts.xquad_avaliar import COMPOSITE, CURVAS_51, DSR_MIN, P_MIN, metricas  # noqa: E402

PASSING: tuple[str, ...] = ("P", "X")   # passed stage 1 (docs/inovemetria-ic-2026-11-12.txt, committed before stage 2)
THETA = 0.5
ANCHOR = "LogDet-15 t0.5"
ROWS = 8                              # arm 3 counts in N whether or not it runs (spec)
DATA = CURVAS_51.parent
PASTA_CVM = FOLDER.parent / "2026-10-24-cvm"
GRID = "fermionica-grade-curvas-fase{}-2026-11-11.parquet"
CURVE_TOL = 1e-9
INOV, COMP_INOV, COMP_GATE = "Inov-15", "Comp+Inov-15", "Comp+InovGate-15"
FERMI_INOV, VN_INOV, FERMI_TECH, FERMI_INOV_TECH, FERMI_HALF = (
    "Fermi-Inov t0.5", "vonNeumann-Inov t0.5", "Fermi-Tech t0.5", "Fermi-Inov-Tech t0.5", "Fermi-Half t0.5")


def scores(dados, mensal) -> dict[str, NotaDoMes]:
    """The Composite's score (attempt 51), INOV, COMP+INOV and COMP+INOV-gate, each memoized."""
    balancos, _ = avaliar_valor.carregar_balancos(PASTA_CVM)
    inovacao, _ = load()
    sinais = families(inovacao, dados.setores, mensal.retornos)
    cons = lambda mes, candidatos: conservadora(mensal.retornos, mensal.proventos, mes, candidatos)  # noqa: E731
    componentes = [nota_media_dos_escores(*avaliar_valor.notas(balancos, mensal).values()),
                   nota_de_crescimento(balancos.fundamentos)]
    inov = memoized(nota_media_dos_escores(*sinais.values()))
    out = {"comp": memoized(composite_score(cons, *componentes)), "inov": inov,
           "comp+inov": memoized(composite_score(cons, *componentes, inov))}
    if 0 < len(PASSING) < len(sinais):
        out["comp+gate"] = memoized(composite_score(cons, *componentes,
                                                    nota_media_dos_escores(*(sinais[f] for f in PASSING))))
    campos: dict = {}

    def fields_of_month(mes: pd.Timestamp) -> pd.DataFrame:
        if mes not in campos:
            campos[mes] = field_counts(inovacao, mes)
        return campos[mes]
    out["fields"] = fields_of_month
    return out


def arms(notas: dict, setores: dict, registros: dict) -> dict[str, Estrategia]:
    """The anchor first, then the spec's arms in order (arm 3 only when 0 < passing families < 5)."""
    tech = mixed_correlation(notas["fields"])
    half = mixed_correlation(lambda mes: pd.DataFrame(columns=WIPO_FIELDS, dtype=float))

    def fermi(nota, nome, **kw) -> Estrategia:
        registros[nome] = {}
        return estrategia_fermionica(nota, setores, THETA, registro=registros[nome], **kw)

    bracos = {ANCHOR: fermi(notas["comp"], ANCHOR), INOV: estrategia_por_nota(notas["inov"], setores),
              COMP_INOV: estrategia_por_nota(notas["comp+inov"], setores)}
    if "comp+gate" in notas:
        bracos[COMP_GATE] = estrategia_por_nota(notas["comp+gate"], setores)
    bracos[FERMI_INOV] = fermi(notas["comp+inov"], FERMI_INOV)
    bracos[VN_INOV] = fermi(notas["comp+inov"], VN_INOV, diversity=von_neumann)
    bracos[FERMI_TECH] = fermi(notas["comp"], FERMI_TECH, correlation=tech)
    bracos[FERMI_INOV_TECH] = fermi(notas["comp+inov"], FERMI_INOV_TECH, correlation=tech)
    bracos[FERMI_HALF] = fermi(notas["comp"], FERMI_HALF, correlation=half)
    return bracos


def changed_decisions(registros: dict, braco: str, base: str) -> str:
    """Decisions whose picks differ between two Fermiônica arms on the same score."""
    a, b = registros[braco], registros[base]
    comuns = sorted(set(a) & set(b))
    diferentes = sum(1 for d in comuns if {a[d][0][i] for i in a[d][1]} != {b[d][0][i] for i in b[d][1]})
    return f"{diferentes} of {len(comuns)}"


def picks_profile(pesos: pd.DataFrame, composite: pd.DataFrame) -> float:
    """Mean names shared with the Composite over the rebalances."""
    datas = pesos.index.intersection(composite.index)
    return float(np.mean([((pesos.loc[d] > 0) & (composite.loc[d] > 0)).sum() for d in datas])) if len(datas) else np.nan


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
    n = contar_tentativas() + ROWS
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
    registros: dict = {}
    bracos = arms(notas, dados.setores, registros)
    curvas, giros, recusas = {COMPOSITE: replay.valores}, {COMPOSITE: replay.turnover_anual_medio}, {}
    pesos = {COMPOSITE: replay.pesos_alvo}
    for posicao, (nome, estrategia) in enumerate(bracos.items()):
        resultado = rodar(estrategia, nome)
        if not resultado.valores.index.equals(indice):
            raise SystemExit(f"phase {fase}: {nome} does not share the Composite's dates")
        if posicao == 0:
            desvio = ancora(resultado, ANCHOR)
            print(f"=== Phase {fase}: {indice[0]:%d/%m/%Y} to {indice[-1]:%d/%m/%Y}; DSR N {n} (with these {ROWS} "
                  f"rows); anchors: Composite {desvio_comp:.1e}, {ANCHOR} {desvio:.1e} off their saved curves; "
                  f"stage-1 families passing: {list(PASSING) or 'none'} ===", flush=True)
        curvas[nome], giros[nome], recusas[nome] = resultado.valores, resultado.turnover_anual_medio, len(resultado.recusas)
        pesos[nome] = resultado.pesos_alvo
        print(f"  [{posicao + 1}/{len(bracos)}] {nome}: {time.perf_counter() - inicio:.0f} s", file=sys.stderr, flush=True)

    curvas["BOVA11"] = referencias(precos, CFG_BT)["BOVA11"].valores.reindex(indice)
    rf = market_data.serie_rf_diaria(indice, cdi, selic.valor)
    curvas["CDI"] = pd.Series(np.cumprod(1.0 + rf.to_numpy()), index=indice)
    tabela = metricas(curvas, rf, n)
    for nome in bracos:                                   # metricas already has alpha, t and P against the Composite
        tabela.loc[nome, "shared"] = picks_profile(pesos[nome], pesos[COMPOSITE])
    tabela["turnover"], tabela["refusals"] = pd.Series(giros), pd.Series(recusas)
    print(tabela.to_string(float_format=lambda x: f"{x:.3f}"))
    controle = comparar_benchmark(curvas[FERMI_TECH], curvas[FERMI_HALF], rf)
    print(f"{FERMI_TECH} against its control {FERMI_HALF}: alpha {controle.alfa_anual:+.2%}, t "
          f"{controle.t_stat_alfa:+.2f}")
    print(f"Decisions the kernel changes: {FERMI_TECH} vs {ANCHOR} {changed_decisions(registros, FERMI_TECH, ANCHOR)}; "
          f"{FERMI_INOV_TECH} vs {FERMI_INOV} {changed_decisions(registros, FERMI_INOV_TECH, FERMI_INOV)}; "
          f"{FERMI_HALF} vs {ANCHOR} {changed_decisions(registros, FERMI_HALF, ANCHOR)}")
    pd.DataFrame(curvas).assign(fase=fase).rename_axis("date").reset_index().to_parquet(
        DATA / f"inovemetria-curvas-fase{fase}-2026-11-12.parquet", index=False)
    tabela.rename_axis("arm").reset_index().assign(fase=fase).to_csv(
        DATA / f"inovemetria-metricas-fase{fase}-2026-11-12.csv", index=False, lineterminator="\n")
    print(f"(phase {fase} done in {time.perf_counter() - inicio:.0f} s)", flush=True)
    return 0


def summary() -> int:
    todas = pd.concat(pd.read_csv(DATA / f"inovemetria-metricas-fase{fase}-2026-11-12.csv") for fase in FASES)
    for coluna, formato in (("CAGR", "{:.1%}"), ("Vol", "{:.1%}"), ("MDD", "{:.1%}"), ("Sharpe", "{:.2f}"),
                            ("P[> Comp]", "{:.2f}"), ("DSR", "{:.3f}"), ("alpha vs Comp", "{:+.1%}"),
                            ("t vs Comp", "{:+.2f}"), ("shared", "{:.1f}"), ("turnover", "{:.1f}")):
        if coluna in todas.columns:
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
        raise SystemExit(f"usage: inovemetria_avaliar.py [{' | '.join(map(str, FASES))} | resumo]")
    raise SystemExit(summary() if sys.argv[1] == "resumo" else phase(int(sys.argv[1])))
