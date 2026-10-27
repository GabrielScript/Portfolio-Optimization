"""
qubo_universo_avaliar.py - the one reading of the QPO's QUBO over the full universe
(docs/superpowers/specs/2026-10-27-qubo-universo-design.md). Runs only after the pre-registration commit.

    python scripts/qubo_universo_avaliar.py > docs/qubo-universo-2026-10-27.txt
"""

from __future__ import annotations

import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core import market_data  # noqa: E402
from core.backtest import Congelamento, walk_forward_dinamico  # noqa: E402
from core.config import taxa_anual_para_aritmetica  # noqa: E402
from core.metrics import cagr, drawdown  # noqa: E402
from core.qpo import CFG_MAX_SHARPE, K, QuboInstance, exact_picks, qubo_selector  # noqa: E402
from core.qpo import selection_then_max_sharpe  # noqa: E402
from core.qubo_universo import FULL_UNIVERSE, SECTOR_CAP, AnnealRecord, anneal_solver  # noqa: E402
from core.robustez import psr_da_diferenca  # noqa: E402
from core.sinais import estrategia_conservadora  # noqa: E402
from scripts import avaliar_conservadora, avaliar_robustez  # noqa: E402
from scripts.avaliar_robustez import FASES, _acoes, contar_tentativas, referencias, tabela_do_periodo  # noqa: E402
from scripts.avaliar_universo_dinamico import ANOS_DE_CDI, CFG_BT  # noqa: E402
from scripts.qpo_avaliar import (CONSERVADORA, MATCH_MIN, _year_returns, _year_table, check_aligned,  # noqa: E402
                                 check_plumbing)
from scripts.qpo_montar import FOLDER, LAST_SESSION, RUN_DATE  # noqa: E402

ARM = "QUBO-universo"
TOP20 = "QPO-exact (top-20)"     # the nested design of the first QPO reading: context, already a trial row


P_MIN = 0.95            # P[> Conservadora] and P[> BOVA11], every phase
DSR_MIN = 0.95          # deflated Sharpe of the IR against BOVA11 (N from docs/tentativas.csv), every phase


def strategy_gate(p_vs_conservadora: Sequence[float], p_vs_bova: Sequence[float], dsr: Sequence[float]) -> bool:
    return all(p >= P_MIN for p in (*p_vs_conservadora, *p_vs_bova)) and all(d >= DSR_MIN for d in dsr)


def verdict(strategy_passed: bool, agreement: float) -> str:
    """The reading: the strategy gate and reproducible picks (the annealing runs agree on >= 80% of rebalances)."""
    if not strategy_passed:
        return "not a candidate: the QPO line is closed"
    if agreement < MATCH_MIN - 1e-12:
        return "not a candidate (picks not reproducible): the QPO line is closed"
    return "CANDIDATE: pre-register a forward paper log (paper-v3)"


def years_beating_all(arm: pd.Series, references: Sequence[pd.Series]) -> tuple[int, int]:
    """(calendar years where the arm's return beats every reference's, years): the user's target, reported only."""
    anos = pd.concat([_year_returns(arm), *(_year_returns(ref) for ref in references)], axis=1).dropna()
    return int((anos.iloc[:, 0].to_numpy()[:, None] > anos.iloc[:, 1:].to_numpy()).all(axis=1).sum()), len(anos)


def anneal_status(log: Mapping[pd.Timestamp, AnnealRecord]) -> str:
    """The solver check: the share of rebalances where two independent annealing runs ended on the same energy."""
    concordancia = float(np.mean([registro.agree for registro in log.values()]))
    leitura = (f">= {MATCH_MIN:.0%}: consistent" if concordancia >= MATCH_MIN - 1e-12 else
               f"< {MATCH_MIN:.0%}: NOT consistent")
    return (f"Annealing (quantum-inspired, solver of record): {len(log)} rebalances, "
            f"{sum(not registro.agree for registro in log.values())} where the two independent runs disagreed "
            f"(agreement {concordancia:.0%} {leitura})")


def picks_profile(record: Mapping[pd.Timestamp, tuple[QuboInstance, tuple[int, ...]]],
                  sectors: Mapping[str, str]) -> str:
    candidatos = [len(instancia.tickers) for instancia, _ in record.values()]
    negativos = [float(np.mean(instancia.alpha < 0)) for instancia, _ in record.values()]
    por_setor = [pd.Series([sectors[instancia.tickers[i]] for i in picks]).value_counts().max()
                 for instancia, picks in record.values()]
    return (f"candidates per rebalance: min {min(candidatos)}, median {int(np.median(candidatos))}, max "
            f"{max(candidatos)}; share of negative alpha {np.mean(negativos):.0%}; most picks in one sector: "
            f"median {int(np.median(por_setor))}, max {max(por_setor)}")


def _run_phase(fase: int, dados, mensal, cfg, cdi) -> tuple[dict[str, pd.Series], dict, dict, int]:
    """Every curve of one phase, plus the arm's recorded QUBOs, MIQP log and refusal count."""
    acoes = _acoes(dados.precos.loc[:LAST_SESSION].iloc[fase:])

    def rodar(estrategia, nome):
        return walk_forward_dinamico(acoes, dados.universo, estrategia, CFG_BT, nome, Congelamento())

    def com_max_sharpe(seletor):
        return selection_then_max_sharpe(seletor, mensal.retornos, mensal.proventos, dados.setores, cfg, cdi)

    conservadora = rodar(estrategia_conservadora(mensal.retornos, mensal.proventos, dados.setores), CONSERVADORA)
    check_plumbing(fase, cagr(conservadora.valores.dropna()))          # before the arm: fail fast
    registro, log = {}, {}
    braco = rodar(com_max_sharpe(qubo_selector(dados.setores, FULL_UNIVERSE, K, anneal_solver(dados.setores,
                                                                                               SECTOR_CAP, log),
                                               registro)), ARM)
    curvas = {ARM: braco, TOP20: rodar(com_max_sharpe(qubo_selector(dados.setores, 20, K, exact_picks)), TOP20),
              CONSERVADORA: conservadora, **referencias(dados.precos.loc[:LAST_SESSION].iloc[fase:], CFG_BT)}
    return {nome: curva.valores for nome, curva in curvas.items()}, registro, log, len(braco.recusas)


def main() -> int:
    inicio = time.perf_counter()
    avaliar_robustez.PASTA_PRECOS = avaliar_robustez.PASTA_UNIVERSO = FOLDER
    dados = avaliar_robustez.carregar()
    mensal = avaliar_conservadora.carregar_mensal(FOLDER)
    selic = market_data.obter_selic_meta(RUN_DATE)
    cdi = market_data.baixar_cdi_diario(ANOS_DE_CDI, hoje=RUN_DATE)
    cfg = replace(CFG_MAX_SHARPE, taxa_livre_risco=taxa_anual_para_aritmetica(selic.valor))
    n_tentativas = contar_tentativas()
    print(f"Files (sha256): {dados.sha256 | mensal.sha256}; k {K}, sector cap {SECTOR_CAP}, DSR N {n_tentativas}")
    p_cons, p_bova, dsr, concordancias = [], [], [], []
    for fase in FASES:
        valores, registro, log, recusas = _run_phase(fase, dados, mensal, cfg, cdi)
        check_aligned(valores)
        rf = market_data.serie_rf_diaria(valores[ARM].index, cdi, selic.valor)
        tabela = tabela_do_periodo({nome: valores[nome] for nome in (ARM, TOP20, CONSERVADORA)}, valores["BOVA11"],
                                   rf, n_tentativas)
        for nome in (ARM, TOP20, CONSERVADORA, "BOVA11"):
            tabela.loc[nome, "CAGR"] = cagr(valores[nome].dropna())
            tabela.loc[nome, "MDD"] = drawdown(valores[nome].dropna())[1]
        for nome in (ARM, TOP20):
            tabela.loc[nome, "P[> Conservadora]"] = psr_da_diferenca(valores[nome], valores[CONSERVADORA])
        p_cons.append(float(tabela.loc[ARM, "P[> Conservadora]"]))
        p_bova.append(float(tabela.loc[ARM, "P aritmético"]))
        dsr.append(float(tabela.loc[ARM, "DSR"]))
        concordancias += [registro.agree for registro in log.values()]
        ganhos, anos = years_beating_all(valores[ARM], [valores["BOVA11"], valores[CONSERVADORA]])
        datas = valores[ARM].index
        print(f"\n=== Phase {fase}: {datas[0]:%d/%m/%Y} to {datas[-1]:%d/%m/%Y}; refusals {recusas}; "
              f"P[{ARM} > {TOP20}] {psr_da_diferenca(valores[ARM], valores[TOP20]):.3f} ===")
        print(anneal_status(log))
        print(picks_profile(registro, dados.setores))
        print(tabela.to_string(float_format=lambda x: f"{x:.3f}"))
        print(f"Years beating BOVA11 and the Conservadora (reported, not a gate; the user's target is 90%): "
              f"{ganhos} of {anos} ({ganhos / anos:.0%})")
        print(_year_table({nome: valores[nome] for nome in (ARM, TOP20, CONSERVADORA, "BOVA11")})
              .to_string(float_format=lambda x: f"{x:.1%}"))
        print(f"(phase {fase} done in {time.perf_counter() - inicio:.0f} s)", flush=True)
    print(f"\nAll phases: agreement of the two annealing runs {np.mean(concordancias):.1%} (>= {MATCH_MIN:.0%})")
    passa = strategy_gate(p_cons, p_bova, dsr)
    print(f"Strategy gate: P[> Conservadora] {[round(p, 3) for p in p_cons]}, P[> BOVA11] "
          f"{[round(p, 3) for p in p_bova]}, DSR {[round(d, 3) for d in dsr]} (all >= {P_MIN}): "
          f"{verdict(passa, float(np.mean(concordancias)))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
