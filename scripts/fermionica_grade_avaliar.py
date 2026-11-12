"""
The one reading of the size x theta x functional grid, the grand-canonical selection and the EIC weights
(``docs/superpowers/specs/2026-11-11-fermionic-size-grid-design.md``, committed before any number).

One phase per process (the three run side by side), then the summary:

    ../Portfolio-Optimization/.venv/Scripts/python -X utf8 scripts/fermionica_grade_avaliar.py 0 > docs/fermionica-grade-fase0-2026-11-11.txt
    ... 21 > docs/fermionica-grade-fase21-2026-11-11.txt        ... 42 > docs/fermionica-grade-fase42-2026-11-11.txt
    ... scripts/fermionica_grade_avaliar.py resumo > docs/fermionica-grade-2026-11-11.txt

Per phase, two anchors first (the Composite's replay has the CAGR measured after the refiled-statements fix,
``docs/duplicados-2026-11-11.txt``; LogDet-15 t0.5 reproduces its saved curve, exactly in phase 0 where the fix
changes nothing and by CAGR elsewhere), then every arm next to the Composite, BOVA11 and the CDI.
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core import market_data  # noqa: E402
from core.backtest import Congelamento, Estrategia, ResultadoBacktest, walk_forward_dinamico  # noqa: E402
from core.composto import estrategia_por_nota  # noqa: E402
from core.diversidade import Diversity, log_det, von_neumann  # noqa: E402
from core.fermionica import com_pesos_eic, estrategia_fermionica, estrategia_grande_canonica, names_per_sector  # noqa: E402
from core.metrics import cagr, comparar_benchmark  # noqa: E402
from core.valor import NotaDoMes  # noqa: E402
from scripts import avaliar_conservadora, avaliar_robustez  # noqa: E402
from scripts.avaliar_robustez import FASES, _acoes, contar_tentativas, referencias  # noqa: E402
from scripts.avaliar_universo_dinamico import ANOS_DE_CDI, CFG_BT  # noqa: E402
from scripts.qpo_montar import FOLDER, LAST_SESSION, RUN_DATE  # noqa: E402
from scripts.qubo_universo_avaliar import _value_score, with_retries  # noqa: E402
from scripts.xquad_avaliar import COMPOSITE, CURVAS_51, DSR_MIN, P_MIN, metricas  # noqa: E402

SIZES = (5, 10, 20, 30)
THETAS = (0.25, 0.5, 0.75, 1.0)
FUNCTIONALS: dict[str, Diversity] = {"LogDet": log_det, "vonNeumann": von_neumann}
READ_SIZE, MISSING_RUNG, FOLLOWED = 15, 0.75, 0.5
GRAND_CANONICAL, COMPOSITE_EIC, FOLLOWED_EIC = "GrandCanonical t0.5", "Composite-EIC", "LogDet-15 t0.5-EIC"
CAGR_AFTER_FIX = {0: 0.214006, 21: 0.200043, 42: 0.190202}     # docs/duplicados-2026-11-11.txt
CAGR_344 = {0: 0.227, 21: 0.204, 42: 0.208}                    # LogDet-15 t0.5 in DECISOES 344, three decimals
CAGR_TOL, CURVE_TOL = 5e-7, 1e-9
ROWS = len(SIZES) * (1 + len(FUNCTIONALS) * len(THETAS)) + len(FUNCTIONALS) + 3
DATA = CURVAS_51.parent
SAVED_344 = DATA / "fermionica-v2-curvas-2026-11-11.parquet"


@dataclass(frozen=True)
class Arm:
    estrategia: Estrategia
    size: int | None            # None: the size is the arm's own (the grand-canonical selection)


def arms(nota: NotaDoMes, setores: dict[str, str], tamanhos_gc: dict) -> dict[str, Arm]:
    """Every row of the spec, the code anchor (LogDet-15 t0.5) first."""
    def fermionic(funcional: Diversity, theta: float, k: int) -> Estrategia:
        return estrategia_fermionica(nota, setores, theta, nomes=k, por_setor=names_per_sector(k), diversity=funcional)

    bracos = {f"LogDet-{READ_SIZE} t{FOLLOWED:g}": Arm(fermionic(log_det, FOLLOWED, READ_SIZE), READ_SIZE)}
    for k in SIZES:
        bracos[f"Composite-{k}"] = Arm(estrategia_por_nota(nota, setores, nomes=k, por_setor=names_per_sector(k)), k)
        for nome, funcional in FUNCTIONALS.items():
            for theta in THETAS:
                bracos[f"{nome}-{k} t{theta:g}"] = Arm(fermionic(funcional, theta, k), k)
    for nome, funcional in FUNCTIONALS.items():
        bracos[f"{nome}-{READ_SIZE} t{MISSING_RUNG:g}"] = Arm(fermionic(funcional, MISSING_RUNG, READ_SIZE), READ_SIZE)
    bracos[GRAND_CANONICAL] = Arm(estrategia_grande_canonica(nota, setores, FOLLOWED, registro=tamanhos_gc), None)
    bracos[COMPOSITE_EIC] = Arm(com_pesos_eic(estrategia_por_nota(nota, setores), setores), READ_SIZE)
    bracos[FOLLOWED_EIC] = Arm(com_pesos_eic(fermionic(log_det, FOLLOWED, READ_SIZE), setores), READ_SIZE)
    return bracos


def memoized(nota: NotaDoMes) -> NotaDoMes:
    """``nota`` computed once per (month, candidates): the 42 arms of a phase ask for the same score."""
    feitas: dict = {}

    def _nota(mes: pd.Timestamp, candidatos) -> pd.Series:
        chave = (mes, tuple(candidatos))
        if chave not in feitas:
            feitas[chave] = nota(mes, candidatos)
        return feitas[chave].copy()
    return _nota


def same_size_baseline(nome: str, braco: Arm) -> str:
    """The Composite of the arm's own size (the 15-name Composite at the read size and for a free size)."""
    return COMPOSITE if braco.size in (None, READ_SIZE) else f"Composite-{braco.size}"


def phase(fase: int) -> int:
    inicio = time.perf_counter()
    avaliar_robustez.PASTA_PRECOS = avaliar_robustez.PASTA_UNIVERSO = FOLDER
    dados = avaliar_robustez.carregar()
    nota = memoized(_value_score(avaliar_conservadora.carregar_mensal(FOLDER), com_crescimento=True)[0])
    selic = market_data.obter_selic_meta(RUN_DATE)
    cdi = with_retries(lambda: market_data.baixar_cdi_diario(ANOS_DE_CDI, hoje=RUN_DATE))
    precos = dados.precos.loc[:LAST_SESSION].iloc[fase:]
    acoes = _acoes(precos)
    n = contar_tentativas() + ROWS

    def rodar(estrategia: Estrategia, nome: str) -> ResultadoBacktest:
        return walk_forward_dinamico(acoes, dados.universo, estrategia, CFG_BT, nome, Congelamento())

    replay = rodar(estrategia_por_nota(nota, dados.setores), COMPOSITE)
    indice = replay.valores.index
    if abs(cagr(replay.valores) - CAGR_AFTER_FIX[fase]) > CAGR_TOL:
        raise SystemExit(f"phase {fase}: anchor 1 failed, the Composite's CAGR is {cagr(replay.valores):.6f}")
    tamanhos_gc: dict = {}
    bracos = arms(nota, dados.setores, tamanhos_gc)
    print(f"=== Phase {fase}: {indice[0]:%d/%m/%Y} to {indice[-1]:%d/%m/%Y}; DSR N {n} (with these {ROWS} rows); "
          f"anchor 1: Composite CAGR {cagr(replay.valores):.4%} ===", flush=True)

    curvas, giros, recusas = {COMPOSITE: replay.valores}, {COMPOSITE: replay.turnover_anual_medio}, {}
    for posicao, (nome, braco) in enumerate(bracos.items()):
        resultado = rodar(braco.estrategia, nome)
        if not resultado.valores.index.equals(indice):
            raise SystemExit(f"phase {fase}: {nome} does not share the Composite's dates")
        curvas[nome], giros[nome], recusas[nome] = resultado.valores, resultado.turnover_anual_medio, len(resultado.recusas)
        if posicao == 0:
            salva = pd.read_parquet(SAVED_344)
            salva = salva.loc[salva["fase"] == fase].set_index("date")["LogDet t0.5"].reindex(indice)
            desvio = float((resultado.valores / salva - 1.0).abs().max())
            exata = fase == 0 and desvio < CURVE_TOL
            if not exata and abs(round(cagr(resultado.valores), 3) - CAGR_344[fase]) > 0.0015:
                raise SystemExit(f"phase {fase}: anchor 2 failed, {nome} is {desvio:.2e} off its saved curve, "
                                 f"CAGR {cagr(resultado.valores):.4f}")
            print(f"anchor 2: {nome} off its saved curve by {desvio:.1e}, CAGR {cagr(resultado.valores):.4%} "
                  f"(DECISOES 344: {CAGR_344[fase]:.1%})", flush=True)
        print(f"  [{posicao + 1}/{len(bracos)}] {nome}: {time.perf_counter() - inicio:.0f} s", file=sys.stderr, flush=True)

    curvas["BOVA11"] = referencias(precos, CFG_BT)["BOVA11"].valores.reindex(indice)
    rf = market_data.serie_rf_diaria(indice, cdi, selic.valor)
    curvas["CDI"] = pd.Series(np.cumprod(1.0 + rf.to_numpy()), index=indice)
    tabela = metricas(curvas, rf, n)
    for nome, braco in bracos.items():
        base = same_size_baseline(nome, braco)
        contra = comparar_benchmark(curvas[nome], curvas[base], rf)
        tabela.loc[nome, "alpha vs same size"] = contra.alfa_anual
        tabela.loc[nome, "t same size"] = contra.t_stat_alfa
    tabela["turnover"], tabela["refusals"] = pd.Series(giros), pd.Series(recusas)
    print(tabela.to_string(float_format=lambda x: f"{x:.3f}"))
    escolhidos = pd.Series(tamanhos_gc).value_counts().sort_index()
    print(f"{GRAND_CANONICAL}: names chosen per decision {escolhidos.to_dict()}, mean "
          f"{np.mean(list(tamanhos_gc.values())):.1f}")
    pd.DataFrame(curvas).assign(fase=fase).rename_axis("date").reset_index().to_parquet(
        DATA / f"fermionica-grade-curvas-fase{fase}-2026-11-11.parquet", index=False)
    tabela.rename_axis("arm").reset_index().assign(fase=fase).to_csv(
        DATA / f"fermionica-grade-metricas-fase{fase}-2026-11-11.csv", index=False, lineterminator="\n")
    print(f"(phase {fase} done in {time.perf_counter() - inicio:.0f} s)", flush=True)
    return 0


def summary() -> int:
    metricas_das_fases = pd.concat(pd.read_csv(DATA / f"fermionica-grade-metricas-fase{fase}-2026-11-11.csv")
                                   for fase in FASES)
    for coluna, formato in (("CAGR", "{:.1%}"), ("Vol", "{:.1%}"), ("MDD", "{:.1%}"), ("Sharpe", "{:.2f}"),
                            ("P[> Comp]", "{:.2f}"), ("alpha vs same size", "{:+.1%}"), ("t same size", "{:+.2f}"),
                            ("turnover", "{:.1f}")):
        if coluna not in metricas_das_fases.columns:
            continue
        print(f"\n{coluna} (phases {' / '.join(map(str, FASES))}):")
        print(metricas_das_fases.pivot(index="arm", columns="fase", values=coluna)
              .loc[lambda t: [nome for nome in metricas_das_fases["arm"].unique() if nome in t.index]]
              .to_string(float_format=formato.format, na_rep="-"))
    print(f"\nGate per arm (P[> Composite] >= {P_MIN} and DSR >= {DSR_MIN} in every phase), {ROWS} correlated rows:")
    passaram = []
    for nome, linhas in metricas_das_fases.groupby("arm", sort=False):
        if nome in (COMPOSITE, "BOVA11", "CDI"):
            continue
        leituras = list(zip(linhas["P[> Comp]"].round(3), linhas["DSR"].round(3)))
        if all(p >= P_MIN and d >= DSR_MIN for p, d in leituras):
            passaram.append(nome)
        print(f"  {nome}: {leituras} -> {'CANDIDATE (forward fictitious account only)' if nome in passaram else 'fails'}")
    print(f"\nArms passing the gate: {passaram or 'none'}")
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in ("resumo", *map(str, FASES)):
        raise SystemExit(f"usage: fermionica_grade_avaliar.py [{' | '.join(map(str, FASES))} | resumo]")
    raise SystemExit(summary() if sys.argv[1] == "resumo" else phase(int(sys.argv[1])))
