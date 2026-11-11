"""
The reading of XQuad and the Quip network as solvers of the full-universe Composite QUBO
(``docs/superpowers/specs/2026-11-11-xquad-quip-design.md``, committed before any number).

    ../Portfolio-Optimization/.venv/Scripts/python -X utf8 scripts/xquad_avaliar.py > docs/xquad-quip-2026-11-11.txt

Replays in the engine the picks solved in the Python 3.13 container (``../quantum-lab/escolhas/*.json``) with
``core.qpo.recorded_selector`` and attempt 56's max-Sharpe weights, next to attempt 56's own picks (anchor: its CAGR),
the Composite (attempt 51's curve), BOVA11 and the CDI. Prints, per phase, the metrics table and the calendar years.
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core import market_data  # noqa: E402
from core.backtest import Congelamento, walk_forward_dinamico  # noqa: E402
from core.config import taxa_anual_para_aritmetica  # noqa: E402
from core.metrics import cagr, comparar_benchmark, drawdown, retornos_de_valores  # noqa: E402
from core.qpo import CFG_MAX_SHARPE, recorded_selector, selection_then_max_sharpe  # noqa: E402
from core.robustez import psr_da_diferenca  # noqa: E402
from scripts import avaliar_conservadora, avaliar_robustez  # noqa: E402
from scripts.avaliar_robustez import FASES, _acoes, contar_tentativas, referencias, tabela_do_periodo  # noqa: E402
from scripts.avaliar_universo_dinamico import ANOS_DE_CDI, CFG_BT  # noqa: E402
from scripts.qpo_avaliar import _year_table  # noqa: E402
from scripts.qpo_montar import FOLDER, LAST_SESSION, RUN_DATE  # noqa: E402
from scripts.qubo_universo_avaliar import _value_score, with_retries  # noqa: E402

LAB = Path(__file__).resolve().parents[2] / "quantum-lab"
CURVAS_51 = FOLDER.parent / "composto-curvas-2026-10-28.parquet"
CAGR_56 = {0: 0.201, 21: 0.165}                 # attempt 56's published CAGRs (3 decimals), the replay's anchor
CPU, QUIP, A56, COMPOSITE = "XQuad-CPU", "XQuad-Quip", "Attempt 56 (own annealer)", "Composite"
P_MIN = DSR_MIN = 0.95
QUIP_FALLBACK_MAX = 0.20


def picks_from_json(caminho: Path) -> tuple[dict[pd.Timestamp, tuple[str, ...]], dict]:
    registros = json.loads(caminho.read_text())
    return {pd.Timestamp(d): tuple(r["tickers"]) for d, r in registros.items()}, registros


def picks_of_56(fase: int) -> dict[pd.Timestamp, tuple[str, ...]]:
    dados = np.load(LAB / "instancias" / f"fase{fase}.npz")
    return {pd.Timestamp(str(d)): tuple(str(dados[f"t{i}"][j]) for j in dados[f"p{i}"])
            for i, d in enumerate(dados["dates"])}


def solver_report(nome: str, registros: dict) -> str:
    vals = list(registros.values())
    iguais = np.mean([r["same_as_56"] for r in vals])
    reparados = np.mean([r["repaired"] for r in vals])
    piores = np.mean([r["energy"] > r["energy_56"] + 1e-12 for r in vals])
    melhores = np.mean([r["energy"] < r["energy_56"] - 1e-12 for r in vals])
    reserva = np.mean([r["solver"] == "cpu-fallback" for r in vals])
    return (f"{nome}: {len(vals)} decisions; same picks as attempt 56 {iguais:.0%}; lower energy {melhores:.0%}, "
            f"higher energy {piores:.0%}; repaired {reparados:.0%}; CPU fallback {reserva:.0%}; median "
            f"{np.median([r['seconds'] for r in vals]):.1f} s per decision")


def metricas(curvas: dict[str, pd.Series], rf: pd.Series, n: int) -> pd.DataFrame:
    tabela = tabela_do_periodo({k: v for k, v in curvas.items() if k not in ("BOVA11", "CDI")}, curvas["BOVA11"], rf, n)
    linhas = {}
    for nome, valores in curvas.items():
        if nome == "CDI":
            continue
        r = retornos_de_valores(valores)
        ex = r - rf.reindex(r.index).fillna(0.0)
        baixo = ex[ex < 0].std() * np.sqrt(252)
        mdd = drawdown(valores)[1]
        linha = {"CAGR": cagr(valores), "Vol": float(r.std() * np.sqrt(252)),
                 "Sharpe": float(ex.mean() * 252 / (r.std() * np.sqrt(252))),
                 "Sortino": float(ex.mean() * 252 / baixo) if baixo > 0 else np.nan, "MDD": mdd,
                 "Calmar": cagr(valores) / abs(mdd) if mdd < 0 else np.nan}
        if nome != COMPOSITE:
            c = comparar_benchmark(valores, curvas[COMPOSITE], rf)
            linha |= {"alpha vs Comp": c.alfa_anual, "t vs Comp": c.t_stat_alfa,
                      "P[> Comp]": psr_da_diferenca(valores, curvas[COMPOSITE])}
        if nome != "BOVA11":
            b = comparar_benchmark(valores, curvas["BOVA11"], rf)
            linha |= {"alpha vs BOVA": b.alfa_anual, "t vs BOVA": b.t_stat_alfa,
                      "DSR": float(tabela.loc[nome, "DSR"]) if nome in tabela.index else np.nan}
        linhas[nome] = linha
    return pd.DataFrame(linhas).T


def main() -> int:
    inicio = time.perf_counter()
    avaliar_robustez.PASTA_PRECOS = avaliar_robustez.PASTA_UNIVERSO = FOLDER
    dados = avaliar_robustez.carregar()
    mensal = avaliar_conservadora.carregar_mensal(FOLDER)
    nota, _ = _value_score(mensal, com_crescimento=True)
    selic = market_data.obter_selic_meta(RUN_DATE)
    cdi = with_retries(lambda: market_data.baixar_cdi_diario(ANOS_DE_CDI, hoje=RUN_DATE))
    cfg = replace(CFG_MAX_SHARPE, taxa_livre_risco=taxa_anual_para_aritmetica(selic.valor))
    publicadas = pd.read_parquet(CURVAS_51)
    n = contar_tentativas() + 1
    print(f"DSR N {n} (with this row); Selic {selic.valor:.4f} ({selic.fonte}); data to {LAST_SESSION:%d/%m/%Y}")

    gate = []
    for fase in FASES:
        acoes = _acoes(dados.precos.loc[:LAST_SESSION].iloc[fase:])

        def rodar(picks, nome):
            estrategia = selection_then_max_sharpe(recorded_selector(picks), mensal.retornos, mensal.proventos,
                                                   dados.setores, cfg, cdi, nota)
            return walk_forward_dinamico(acoes, dados.universo, estrategia, CFG_BT, nome, Congelamento()).valores

        curvas = {A56: rodar(picks_of_56(fase), A56)}
        if fase in CAGR_56 and abs(round(cagr(curvas[A56]), 3) - CAGR_56[fase]) > 0.0015:
            raise SystemExit(f"phase {fase}: anchor failed, attempt 56 replay CAGR {cagr(curvas[A56]):.4f}")
        relatorios = []
        cpu_picks, cpu_reg = picks_from_json(LAB / "escolhas" / f"cpu-fase{fase}.json")
        curvas[CPU] = rodar(cpu_picks, CPU)
        relatorios.append(solver_report(CPU, cpu_reg))
        quip_arquivo = LAB / "escolhas" / f"quip-fase{fase}.json"
        if quip_arquivo.exists():
            quip_picks, quip_reg = picks_from_json(quip_arquivo)
            reserva = np.mean([r["solver"] == "cpu-fallback" for r in quip_reg.values()])
            relatorios.append(solver_report(QUIP, quip_reg))
            if len(quip_reg) == len(cpu_reg) and reserva <= QUIP_FALLBACK_MAX:
                curvas[QUIP] = rodar(quip_picks, QUIP)
            else:
                relatorios.append(f"{QUIP}: not run as an arm ({len(quip_reg)} of {len(cpu_reg)} decisions, "
                                  f"fallback {reserva:.0%})")
        indice = curvas[CPU].index
        composite = publicadas.loc[publicadas["fase"] == fase].set_index("date")["Composto"]
        curvas[COMPOSITE] = composite.reindex(indice)
        curvas["BOVA11"] = referencias(dados.precos.loc[:LAST_SESSION].iloc[fase:], CFG_BT)["BOVA11"].valores \
            .reindex(indice)
        rf = market_data.serie_rf_diaria(indice, cdi, selic.valor)
        curvas["CDI"] = pd.Series(np.cumprod(1.0 + rf.to_numpy()), index=indice)
        tabela = metricas(curvas, rf, n)
        gate.append((float(tabela.loc[CPU, "P[> Comp]"]), float(tabela.loc[CPU, "DSR"])))
        print(f"\n=== Phase {fase}: {indice[0]:%d/%m/%Y} to {indice[-1]:%d/%m/%Y} ===")
        print("\n".join(relatorios))
        print(tabela.to_string(float_format=lambda x: f"{x:.3f}"))
        print("Calendar years:")
        print(_year_table(curvas).to_string(float_format=lambda x: f"{x:.1%}"))
        print(f"(phase {fase} done in {time.perf_counter() - inicio:.0f} s)", flush=True)

    passa = all(p >= P_MIN and d >= DSR_MIN for p, d in gate)
    print(f"\nGate (P[XQuad-CPU > Composite] >= {P_MIN} and DSR >= {DSR_MIN} in every phase): "
          f"{[(round(p, 3), round(d, 3)) for p, d in gate]} -> "
          + ("CANDIDATE" if passa else "FAILS: the quantum-solver line on the B3 stays closed"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
