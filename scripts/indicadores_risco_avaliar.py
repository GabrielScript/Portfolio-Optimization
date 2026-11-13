"""
Stage 2 of ``docs/superpowers/specs/2026-11-13-indicadores-risco-design.md`` (amendment 1, committed before any
stage 2 number; stage 1 read in ``docs/indicadores-risco-ic-2026-11-13.txt``): the four stage 2 candidates as
earned-weight components of the 15-name Composite (the rule of DECISOES 354), read once. One phase per process, then
the summary:

    python -X utf8 scripts/indicadores_risco_avaliar.py 0 > docs/indicadores-risco-fase0-2026-11-13.txt
    ... 21 > docs/indicadores-risco-fase21-2026-11-13.txt      ... 42 > docs/indicadores-risco-fase42-2026-11-13.txt
    ... scripts/indicadores_risco_avaliar.py resumo > docs/indicadores-risco-2026-11-13.txt

Anchor per phase, before any arm is read: the Composite's replay reproduces its curve saved by the size grid of
DECISOES 350 (1e-9).
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core import market_data
from core.backtest import (
    Congelamento,
    Estrategia,
    ResultadoBacktest,
    walk_forward_dinamico,
)
from core.composto import estrategia_por_nota
from core.crescimento_lucro import nota_de_crescimento
from core.indicadores_no_tempo import Fontes, notas_dos_sinais
from core.informados import earned_weight, weighted_score
from core.qubo_universo import composite_score
from core.sinais import conservadora
from core.valor import NotaDoMes
from core.views import nota_media_dos_escores
from scripts import avaliar_conservadora, avaliar_robustez, avaliar_valor
from scripts.avaliar_robustez import FASES, _acoes, referencias
from scripts.avaliar_universo_dinamico import ANOS_DE_CDI, CFG_BT
from scripts.fermionica_grade_avaliar import memoized
from scripts.inovemetria_avaliar import CURVE_TOL, DATA, GRID, PASTA_CVM, picks_profile
from scripts.qpo_montar import FOLDER, LAST_SESSION, RUN_DATE
from scripts.qubo_universo_avaliar import with_retries
from scripts.xquad_avaliar import COMPOSITE, DSR_MIN, P_MIN, metricas

N_DSR = 132                                     # amendment 1: 128 rows before, these four
PASTA_MACRO = FOLDER.parent / "2026-10-21-macro"
ARMS = {"Comp+RET3y-w": "RET-3y", "Comp+CVAR-w": "CVAR", "Comp+ALPHA-w": "ALPHA", "Comp+RISKALL-w": "RISK-ALL"}
CURVAS = "indicadores-risco-curvas-fase{}-2026-11-13.parquet"
METRICAS = "indicadores-risco-metricas-fase{}-2026-11-13.csv"


def scores(mensal) -> dict[str, NotaDoMes]:
    """The Composite's score (attempt 51) and the four signals of the table at a month, each memoized."""
    balancos, _ = avaliar_valor.carregar_balancos(PASTA_CVM)
    cdi = pd.read_parquet(PASTA_MACRO / "cdi_daily.parquet")          # the frozen CDI of stage 1: the same signals
    juro = pd.Series(cdi["cdi_pct"].to_numpy() / 100.0, index=pd.to_datetime(cdi["date"]))
    dados = avaliar_robustez.carregar()
    comp = composite_score(lambda mes, candidatos: conservadora(mensal.retornos, mensal.proventos, mes, candidatos),
                           nota_media_dos_escores(*avaliar_valor.notas(balancos, mensal).values()),
                           nota_de_crescimento(balancos.fundamentos))
    sinais = notas_dos_sinais(Fontes(dados.precos, juro, mensal.retornos, balancos.fundamentos), ARMS.values())
    return {"comp": memoized(comp), **sinais}


def weights(notas: dict[str, NotaDoMes], dados, mensal) -> dict[str, pd.Series]:
    """lambda of each decision month for each signal: from the monthly ICs of the months before it."""
    ics = {nome: avaliar_valor.ics_e_cobertura(nota, mensal, dados.universo)[0] for nome, nota in notas.items()}
    meses = pd.date_range(avaliar_valor.PERIODO_DO_IC[0], LAST_SESSION, freq="MS")
    return {sinal: pd.Series({mes: earned_weight(ics[sinal], ics["comp"], mes) for mes in meses})
            for sinal in ARMS.values()}


def arms(notas: dict[str, NotaDoMes], pesos: dict[str, pd.Series], setores: dict) -> dict[str, Estrategia]:
    def com_peso(sinal: str) -> Estrategia:
        return estrategia_por_nota(
            weighted_score(notas["comp"], notas[sinal], lambda mes: float(pesos[sinal].get(mes, 0.0))), setores)
    return {braco: com_peso(sinal) for braco, sinal in ARMS.items()}


def phase(fase: int) -> int:
    inicio = time.perf_counter()
    avaliar_robustez.PASTA_PRECOS = avaliar_robustez.PASTA_UNIVERSO = avaliar_valor.PASTA_UNIVERSO = FOLDER
    dados = avaliar_robustez.carregar()
    mensal = avaliar_conservadora.carregar_mensal(FOLDER)
    notas = scores(mensal)
    selic = market_data.obter_selic_meta(RUN_DATE)
    cdi = with_retries(lambda: market_data.baixar_cdi_diario(ANOS_DE_CDI, hoje=RUN_DATE))
    precos = dados.precos.loc[:LAST_SESSION].iloc[fase:]
    acoes = _acoes(precos)
    salvas = pd.read_parquet(DATA / GRID.format(fase)).set_index("date")

    def rodar(estrategia: Estrategia, nome: str) -> ResultadoBacktest:
        return walk_forward_dinamico(acoes, dados.universo, estrategia, CFG_BT, nome, Congelamento())

    replay = rodar(estrategia_por_nota(notas["comp"], dados.setores), COMPOSITE)
    indice = replay.valores.index
    desvio = float((replay.valores / salvas[COMPOSITE].reindex(indice) - 1.0).abs().max())
    if not desvio < CURVE_TOL:
        raise SystemExit(f"phase {fase}: the Composite's replay is {desvio:.2e} off its saved curve; no arm was read")
    print(f"=== Phase {fase}: {indice[0]:%d/%m/%Y} to {indice[-1]:%d/%m/%Y}; DSR N {N_DSR}; anchor: Composite "
          f"{desvio:.1e} off its saved curve ===", flush=True)
    print(f"  anchor ok at {time.perf_counter() - inicio:.0f} s", file=sys.stderr, flush=True)

    pesos = weights(notas, dados, mensal)
    bracos = arms(notas, pesos, dados.setores)
    curvas, giros, recusas = {COMPOSITE: replay.valores}, {COMPOSITE: replay.turnover_anual_medio}, {}
    alvos = {COMPOSITE: replay.pesos_alvo}
    for posicao, (nome, estrategia) in enumerate(bracos.items()):
        resultado = rodar(estrategia, nome)
        if not resultado.valores.index.equals(indice):
            raise SystemExit(f"phase {fase}: {nome} does not share the Composite's dates")
        curvas[nome], giros[nome], recusas[nome] = (resultado.valores, resultado.turnover_anual_medio,
                                                    len(resultado.recusas))
        alvos[nome] = resultado.pesos_alvo
        print(f"  [{posicao + 1}/{len(bracos)}] {nome}: {time.perf_counter() - inicio:.0f} s", file=sys.stderr,
              flush=True)

    curvas["BOVA11"] = referencias(precos, CFG_BT)["BOVA11"].valores.reindex(indice)
    rf = market_data.serie_rf_diaria(indice, cdi, selic.valor)
    curvas["CDI"] = pd.Series(np.cumprod(1.0 + rf.to_numpy()), index=indice)
    tabela = metricas(curvas, rf, N_DSR)
    for nome in bracos:
        tabela.loc[nome, "shared"] = picks_profile(alvos[nome], alvos[COMPOSITE])
        tabela.loc[nome, "names"] = float((alvos[nome] > 0).sum(axis=1).mean())
    tabela["turnover"], tabela["refusals"] = pd.Series(giros), pd.Series(recusas)
    print(tabela.to_string(float_format=lambda x: f"{x:.3f}"))
    for braco, sinal in ARMS.items():
        serie = pesos[sinal]
        ativo = serie[serie > 0]
        print(f"{braco}: lambda > 0 in {len(ativo)} of {len(serie)} months"
              + (f", from {ativo.index[0]:%Y-%m}; mean {ativo.mean():.2f}, min {ativo.min():.2f}, max "
                 f"{ativo.max():.2f}, last {serie.iloc[-1]:.2f}" if len(ativo) else ""))
    pd.DataFrame(curvas).assign(fase=fase).rename_axis("date").reset_index().to_parquet(
        DATA / CURVAS.format(fase), index=False)
    tabela.rename_axis("arm").reset_index().assign(fase=fase).to_csv(
        DATA / METRICAS.format(fase), index=False, lineterminator="\n")
    print(f"(phase {fase} done in {time.perf_counter() - inicio:.0f} s)", flush=True)
    return 0


def summary() -> int:
    todas = pd.concat(pd.read_csv(DATA / METRICAS.format(fase)) for fase in FASES)
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
        if nome not in ARMS:
            continue
        leituras = list(zip(linhas["P[> Comp]"].round(3), linhas["DSR"].round(3)))
        if all(p >= P_MIN and d >= DSR_MIN for p, d in leituras):
            passaram.append(nome)
        print(f"  {nome}: {leituras} -> {'PASSES (forward fictitious account only)' if nome in passaram else 'fails'}")
    print(f"\nArms passing the gate: {passaram or 'none'}")
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in ("resumo", *map(str, FASES)):
        raise SystemExit(f"usage: indicadores_risco_avaliar.py [{' | '.join(map(str, FASES))} | resumo]")
    raise SystemExit(summary() if sys.argv[1] == "resumo" else phase(int(sys.argv[1])))
