"""
qpo_avaliar.py - the one QPO out-of-sample reading (docs/superpowers/specs/2026-10-27-qpo-design.md, sections 5
and 6). Runs only after the pre-registration commit and the Kaggle solve; output to docs/qpo-2026-10-27.txt.

    python scripts/qpo_avaliar.py N_MAX RESULTS_DIR > docs/qpo-2026-10-27.txt
    python scripts/qpo_avaliar.py N_MAX > docs/qpo-exact-2026-10-27.txt     (every arm but QPO-QAOA: no Kaggle needed)
"""

from __future__ import annotations

import json
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core import market_data  # noqa: E402
from core.backtest import Congelamento, ResultadoBacktest, estrategia_otimizada_nas_regras  # noqa: E402
from core.backtest import walk_forward_dinamico  # noqa: E402
from core.config import taxa_anual_para_aritmetica  # noqa: E402
from core.metrics import cagr, drawdown  # noqa: E402
from core.qpo import (CFG_MAX_SHARPE, K, exact_picks, qubo_selector, random_selector,  # noqa: E402
                      recorded_selector, selection_then_max_sharpe)
from core.robustez import psr_da_diferenca  # noqa: E402
from core.sinais import carteira_da_nota, estrategia_conservadora  # noqa: E402
from scripts import avaliar_conservadora, avaliar_robustez  # noqa: E402
from scripts.avaliar_robustez import FASES, _acoes, contar_tentativas, referencias, tabela_do_periodo  # noqa: E402
from scripts.avaliar_universo_dinamico import ANOS_DE_CDI, CFG_BT  # noqa: E402
from scripts.qpo_montar import FOLDER, LAST_SESSION, RUN_DATE, SAIDA  # noqa: E402

P_MIN = 0.95                 # strategy gate, both references, every phase
MATCH_MIN = 0.80             # solver gate: share of rebalances with the exact pick
RATIO_MIN = 0.98             # solver gate: mean approximation ratio
RANDOM_SEEDS = 20
CONSERVADORA = "Conservadora (15, pesos iguais)"
ARMS = ("QPO-exact", "QPO-QAOA", "Conservadora-MS", "Topk-MS", "Random-15 (mediana)")
# docs/conservadora-2026-10-22.txt, the same frozen folder and price cut: equal weights do not use the CDI, so the
# engine must give these CAGRs back (printed with 2 decimals); a mismatch is a wiring bug, not a result
PUBLISHED_CONSERVADORA_CAGR = {0: 0.1741, 21: 0.1641, 42: 0.1690}


def strategy_gate(p_vs_conservadora: Sequence[float], p_vs_bova: Sequence[float]) -> bool:
    return all(p >= P_MIN for p in (*p_vs_conservadora, *p_vs_bova))


def solver_gate(matches: Sequence[bool], ratios: Sequence[float]) -> bool:
    return float(np.mean(matches)) >= MATCH_MIN and float(np.mean(ratios)) >= RATIO_MIN


def choose_p(ratios_by_p: Mapping[int, Sequence[float]]) -> int:
    """The p with the highest mean approximation ratio on phase 0; a tie goes to the smaller p (fewer gates)."""
    return min(ratios_by_p, key=lambda p: (-float(np.mean(ratios_by_p[p])), p))


def _results(results_dir: Path, fase: int, p: int) -> dict[str, dict]:
    return {arquivo.stem.rsplit("_p", 1)[0]: json.loads(arquivo.read_text(encoding="utf-8"))
            for arquivo in sorted((Path(results_dir) / f"fase{fase}").glob(f"*_p{p}.json"))}


def load_qaoa_picks(results_dir: Path, qubos_dir: Path, fase: int, p: int) -> dict[pd.Timestamp, tuple[str, ...]]:
    """Decision date -> QAOA tickers; every result's QUBO hash must match ``SHA256SUMS`` of the exported QUBOs."""
    somas = {}
    for linha in (Path(qubos_dir) / "SHA256SUMS").read_text(encoding="utf-8").splitlines():
        hash_, caminho = linha.split("  ", 1)
        somas[caminho] = hash_
    escolhas = {}
    for data, resultado in _results(results_dir, fase, p).items():
        if somas.get(f"fase{fase}/{data}.npz") != resultado["sha256"]:
            raise ValueError(f"fase {fase} {data}: the QAOA result was not solved on the exported QUBO")
        escolhas[pd.Timestamp(data)] = tuple(resultado["tickers"])
    return escolhas


def check_aligned(valores: Mapping[str, pd.Series]) -> None:
    """Every curve on the same dates (as ``avaliar_conservadora.curvas_da_fase``): misaligned indexes would give NaN
    statistics silently."""
    primeiro = next(iter(valores.values())).index
    desalinhadas = [nome for nome, serie in valores.items() if not serie.index.equals(primeiro)]
    if desalinhadas:
        raise SystemExit(f"curves on different dates: {desalinhadas}")


def check_plumbing(fase: int, cagr_conservadora: float) -> None:
    publicado = PUBLISHED_CONSERVADORA_CAGR[fase]
    if abs(cagr_conservadora - publicado) > 0.00005 + 1e-12:
        raise SystemExit(f"phase {fase}: Conservadora CAGR {cagr_conservadora:.4%} != published {publicado:.2%}: "
                         "data or engine wiring differs from docs/conservadora-2026-10-22.txt")


def _median_curve(curvas: Sequence[ResultadoBacktest]) -> ResultadoBacktest:
    valores = pd.concat([curva.valores for curva in curvas], axis=1).median(axis=1)
    return replace(curvas[0], nome="Random-15 (mediana)", valores=valores)


def _year_table(curvas: Mapping[str, pd.Series]) -> pd.DataFrame:
    return pd.DataFrame({nome: valores.dropna().resample("YE").last().pct_change().dropna()
                         for nome, valores in curvas.items()}).rename(index=lambda d: d.year)


def main(argv: list[str]) -> int:
    n_max = int(argv[0])
    results_dir = Path(argv[1]) if len(argv) > 1 else None       # None: every arm but QPO-QAOA
    arms = ARMS if results_dir is not None else tuple(arm for arm in ARMS if arm != "QPO-QAOA")
    inicio = time.perf_counter()
    avaliar_robustez.PASTA_PRECOS = avaliar_robustez.PASTA_UNIVERSO = FOLDER
    dados = avaliar_robustez.carregar()
    precos = dados.precos.loc[:LAST_SESSION]
    mensal = avaliar_conservadora.carregar_mensal(FOLDER)
    selic = market_data.obter_selic_meta(RUN_DATE)
    cdi = market_data.baixar_cdi_diario(ANOS_DE_CDI, hoje=RUN_DATE)
    cfg = replace(CFG_MAX_SHARPE, taxa_livre_risco=taxa_anual_para_aritmetica(selic.valor))
    n_tentativas = contar_tentativas()
    setores = dados.setores
    p_escolhido = (None if results_dir is None else
                   choose_p({p: [r["ratio"] for r in _results(results_dir, 0, p).values()] for p in (1, 2, 3)}))
    print(f"Files (sha256): {dados.sha256 | mensal.sha256}; N_MAX {n_max}, k {K}, p {p_escolhido}, "
          f"DSR N {n_tentativas}")

    p_cons, p_bova, matches, ratios = [], [], [], []
    for fase in FASES:
        acoes = _acoes(precos.iloc[fase:])

        def rodar(estrategia, nome):
            return walk_forward_dinamico(acoes, dados.universo, estrategia, CFG_BT, nome, Congelamento())

        def com_max_sharpe(seletor):
            return selection_then_max_sharpe(seletor, mensal.retornos, mensal.proventos, setores, cfg, cdi)

        conservadora = rodar(estrategia_conservadora(mensal.retornos, mensal.proventos, setores), CONSERVADORA)
        check_plumbing(fase, cagr(conservadora.valores.dropna()))       # before the arms: fail fast
        curvas = {
            "QPO-exact": rodar(com_max_sharpe(qubo_selector(setores, n_max, K, exact_picks)), "QPO-exact"),
            "Conservadora-MS": rodar(com_max_sharpe(
                lambda data, treino, nota: list(carteira_da_nota(nota, setores, K, 4).index)), "Conservadora-MS"),
            "Topk-MS": rodar(estrategia_otimizada_nas_regras(replace(cfg, n_ativos_max=K), True, setores, cdi),
                             "Topk-MS"),
            "Random-15 (mediana)": _median_curve([rodar(com_max_sharpe(random_selector(setores, n_max, K, s)),
                                                        f"Random {s}") for s in range(RANDOM_SEEDS)]),
            CONSERVADORA: conservadora,
            **referencias(precos.iloc[fase:], CFG_BT),
        }
        if results_dir is not None:
            curvas["QPO-QAOA"] = rodar(com_max_sharpe(recorded_selector(
                load_qaoa_picks(results_dir, SAIDA, fase, p_escolhido))), "QPO-QAOA")
        valores = {nome: curva.valores for nome, curva in curvas.items()}
        check_aligned(valores)
        rf = market_data.serie_rf_diaria(valores["QPO-exact"].index, cdi, selic.valor)
        tabela = tabela_do_periodo({nome: valores[nome] for nome in (*arms, CONSERVADORA)}, valores["BOVA11"], rf,
                                   n_tentativas)
        for nome in (*arms, CONSERVADORA, "BOVA11"):
            tabela.loc[nome, "CAGR"] = cagr(valores[nome].dropna())
            tabela.loc[nome, "MDD"] = drawdown(valores[nome].dropna())[1]
        for nome in arms:
            tabela.loc[nome, "P[> Conservadora]"] = psr_da_diferenca(valores[nome], valores[CONSERVADORA])
            tabela.loc[nome, "P[> Conservadora] log"] = psr_da_diferenca(valores[nome], valores[CONSERVADORA], True)
        p_cons.append(float(tabela.loc["QPO-exact", "P[> Conservadora]"]))
        p_bova.append(float(tabela.loc["QPO-exact", "P aritmético"]))
        selecao = psr_da_diferenca(valores["QPO-exact"], valores["Conservadora-MS"])
        if results_dir is not None:
            resultados = _results(results_dir, fase, p_escolhido)
            matches += [r["match"] for r in resultados.values()]
            ratios += [r["ratio"] for r in resultados.values()]
        print(f"\n=== Phase {fase}: {valores['QPO-exact'].index[0]:%d/%m/%Y} to {valores['QPO-exact'].index[-1]:%d/%m/%Y};"
              f" refusals QPO-exact {len(curvas['QPO-exact'].recusas)}; P[QPO-exact > Conservadora-MS] (selection)"
              f" {selecao:.3f} ===")
        print(tabela.to_string(float_format=lambda x: f"{x:.3f}"))
        print("\nYear by year (reported, not a gate):")
        print(_year_table({nome: valores[nome] for nome in (*arms[:2], CONSERVADORA, "BOVA11")})
              .to_string(float_format=lambda x: f"{x:.1%}"))
        print(f"(phase {fase} done in {time.perf_counter() - inicio:.0f} s)", flush=True)

    estrategia_ok = strategy_gate(p_cons, p_bova)
    if results_dir is None:
        print("\nSolver gate: not read (no QAOA results yet)")
    else:
        solver_ok = solver_gate(matches, ratios)
        print(f"\nSolver gate: matches {np.mean(matches):.1%} (>= {MATCH_MIN:.0%}), mean ratio {np.mean(ratios):.4f}"
              f" (>= {RATIO_MIN}): {'QAOA faithful to the exact QUBO' if solver_ok else 'QAOA NOT faithful'}")
    print(f"Strategy gate: P[QPO-exact > Conservadora] {[round(p, 3) for p in p_cons]}, P[> BOVA11] "
          f"{[round(p, 3) for p in p_bova]} (all >= {P_MIN}): {'CANDIDATE' if estrategia_ok else 'not a candidate'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
