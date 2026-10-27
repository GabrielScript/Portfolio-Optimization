"""QUBO selection over the full universe: exact MIQP (SCIP), simulated annealing, sector caps, random-subset energy."""

import itertools
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("pyscipopt")

from core.qpo import RISK_AVERSION, QuboInstance, build_qubo, energy  # noqa: E402
from core.qubo_universo import GroupCaps, anneal, convex_split, random_subset_energy, solve_miqp  # noqa: E402

QUBOS = Path(__file__).resolve().parent.parent / "docs" / "qpo" / "qubos"


def _recorded_instance(arquivo: Path) -> tuple[QuboInstance, float]:
    dados = np.load(arquivo)
    instancia = QuboInstance(pd.Timestamp(arquivo.stem), tuple(dados["tickers"]), dados["q"], dados["alpha"],
                             dados["cov"], int(dados["k"]))
    return instancia, float(dados["exact_energy"])


@pytest.mark.parametrize("arquivo", sorted(QUBOS.glob("fase*/*.npz"))[::17], ids=lambda p: f"{p.parent.name}-{p.stem}")
def test_miqp_reaches_the_exact_optimum_of_the_recorded_qpo_qubos(arquivo):
    instancia, otimo = _recorded_instance(arquivo)
    solucao = solve_miqp(instancia, RISK_AVERSION)
    x = np.zeros(len(instancia.tickers))
    x[list(solucao.picks)] = 1.0
    assert solucao.optimal
    assert len(solucao.picks) == instancia.k
    assert energy(instancia.q, x) == pytest.approx(otimo, abs=1e-9)
    assert solucao.energy == pytest.approx(energy(instancia.q, x), abs=1e-15)


def _synthetic_instance(n: int, k: int, seed: int) -> QuboInstance:
    """Stock-like inputs: alpha of a few % a year, a one-factor covariance with idiosyncratic noise."""
    rng = np.random.default_rng(seed)
    beta = rng.uniform(0.5, 1.5, n)
    cov = 0.04 * np.outer(beta, beta) + np.diag(rng.uniform(0.02, 0.12, n))
    alpha = rng.normal(0.0, 0.03, n)
    return QuboInstance(pd.Timestamp("2020-01-02"), tuple(f"N{i}" for i in range(n)),
                        build_qubo(alpha, cov, k, RISK_AVERSION), alpha, cov, k)


def _brute_force(instancia: QuboInstance, grupos: tuple[str, ...] | None, teto: int) -> float:
    n = len(instancia.tickers)
    melhor = np.inf
    for combinacao in itertools.combinations(range(n), instancia.k):
        if grupos is not None and max(Counter(grupos[i] for i in combinacao).values()) > teto:
            continue
        melhor = min(melhor, energy(instancia.q, np.isin(np.arange(n), combinacao).astype(float)))
    return melhor


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_miqp_with_group_caps_matches_brute_force_over_the_capped_subsets(seed):
    instancia = _synthetic_instance(12, 5, seed)
    grupos = tuple("ABC"[i % 3] for i in range(12))
    solucao = solve_miqp(instancia, RISK_AVERSION, GroupCaps(grupos, 2))
    assert max(Counter(grupos[i] for i in solucao.picks).values()) <= 2
    assert solucao.energy == pytest.approx(_brute_force(instancia, grupos, 2), abs=1e-12)
    assert solucao.energy > _brute_force(instancia, None, 5) - 1e-12   # the caps bind or cost nothing


def test_miqp_out_of_time_reports_not_optimal_and_returns_the_capped_greedy_incumbent():
    instancia = _synthetic_instance(30, 8, seed=7)
    grupos = tuple("ABCD"[i % 4] for i in range(30))
    solucao = solve_miqp(instancia, RISK_AVERSION, GroupCaps(grupos, 2), time_limit=0.0)
    assert not solucao.optimal
    assert solucao.picks == (0, 1, 2, 3, 4, 5, 6, 7)          # the first k in candidate (score) order, 2 per group


def test_miqp_refuses_a_risk_aversion_that_does_not_rebuild_the_instance_qubo():
    with pytest.raises(ValueError, match="risk aversion"):
        solve_miqp(_synthetic_instance(8, 3, seed=4), 2.0 * RISK_AVERSION)


def test_convex_split_equals_the_qubo_on_every_bitstring_with_the_tightest_identity_shift():
    instancia = _synthetic_instance(8, 3, seed=6)
    quadratica, linear = convex_split(instancia.alpha, instancia.cov, 3, RISK_AVERSION)
    for bits in itertools.product((0.0, 1.0), repeat=8):
        x = np.array(bits)
        assert x @ quadratica @ x + linear @ x == pytest.approx(energy(instancia.q, x), abs=1e-14)
    assert np.linalg.eigvalsh(quadratica).min() == pytest.approx(0.0, abs=1e-12)


def test_random_subset_energy_is_the_mean_energy_over_every_k_subset():
    instancia = _synthetic_instance(9, 4, seed=5)
    todas = [energy(instancia.q, np.isin(np.arange(9), c).astype(float)) for c in itertools.combinations(range(9), 4)]
    assert random_subset_energy(instancia.q, 4) == pytest.approx(np.mean(todas), abs=1e-14)


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_annealing_finds_the_capped_optimum_on_small_instances(seed):
    instancia = _synthetic_instance(12, 5, seed)
    grupos = tuple("ABC"[i % 3] for i in range(12))
    picks = anneal(instancia.q, instancia.k, GroupCaps(grupos, 2), seed=seed)
    assert len(picks) == 5 and max(Counter(grupos[i] for i in picks).values()) <= 2
    assert energy(instancia.q, np.isin(np.arange(12), picks).astype(float)) == pytest.approx(
        _brute_force(instancia, grupos, 2), abs=1e-12)


@pytest.mark.parametrize("arquivo", sorted(QUBOS.glob("fase*/*.npz"))[::17], ids=lambda p: f"{p.parent.name}-{p.stem}")
def test_annealing_reaches_the_exact_optimum_of_the_recorded_qpo_qubos(arquivo):
    instancia, otimo = _recorded_instance(arquivo)
    picks = anneal(instancia.q, instancia.k, seed=20261027)
    assert energy(instancia.q, np.isin(np.arange(len(instancia.tickers)), picks).astype(float)) == pytest.approx(
        otimo, abs=1e-12)


# --- the strategy side: every eligible name is a candidate, sectors capped as the Conservadora's 4 per sector --------

from core.qpo import candidates, qubo_instance  # noqa: E402
from core.qubo_universo import FULL_UNIVERSE, anneal_solver, miqp_solver  # noqa: E402
from core.sinais import conservadora  # noqa: E402
from tests.test_sinais import MES, SETORES_DOS_8, _painel_de_8_papeis, _treino  # noqa: E402

SETORES_EM_PARES = {papel: f"Setor {i % 4}" for i, papel in enumerate("ABCDEFGH")}   # A+E, B+F, C+G, D+H


def test_miqp_solver_takes_every_candidate_caps_the_sectors_and_logs_each_solve():
    retornos, proventos = _painel_de_8_papeis()
    score = conservadora(retornos, proventos, MES, list("ABCDEFGH"))
    instancia = qubo_instance(MES, _treino(0.01), score, SETORES_EM_PARES, n=FULL_UNIVERSE, k=4)
    registro: dict = {}
    picks = miqp_solver(SETORES_EM_PARES, max_per_sector=1, log=registro)(instancia)
    grupos = tuple(SETORES_EM_PARES[papel] for papel in instancia.tickers)
    assert instancia.tickers == tuple(candidates(score, SETORES_EM_PARES, FULL_UNIVERSE))
    assert len(instancia.tickers) == len(score.dropna())
    assert picks == solve_miqp(instancia, RISK_AVERSION, GroupCaps(grupos, 1)).picks
    assert len({grupos[i] for i in picks}) == 4
    assert registro[instancia.date].picks == picks


def _datada(instancia: QuboInstance, data: str) -> QuboInstance:
    return QuboInstance(pd.Timestamp(data), instancia.tickers, instancia.q, instancia.alpha, instancia.cov, instancia.k)


def test_anneal_solver_keeps_the_lower_energy_of_two_seeds_and_logs_their_agreement():
    instancia = _datada(_synthetic_instance(12, 5, seed=1), "2020-03-02")
    setores = {f"N{i}": "ABC"[i % 3] for i in range(12)}
    registro: dict = {}
    picks = anneal_solver(setores, max_per_sector=2, log=registro)(instancia)
    grupos = tuple(setores[papel] for papel in instancia.tickers)
    assert registro[instancia.date].picks == picks
    assert registro[instancia.date].agree
    assert registro[instancia.date].energy == pytest.approx(_brute_force(instancia, grupos, 2), abs=1e-12)


def test_anneal_solver_flags_disagreement_and_takes_the_lower_energy_pick():
    instancia = _datada(_synthetic_instance(12, 5, seed=2), "2020-03-02")
    setores = {f"N{i}": "ABC"[i % 3] for i in range(12)}
    respostas = {instancia.date.toordinal(): (0, 1, 3, 4, 6), instancia.date.toordinal() + 1: (1, 2, 4, 5, 7)}
    registro: dict = {}
    picks = anneal_solver(setores, 2, registro, annealer=lambda q, k, caps, seed: respostas[seed])(instancia)

    def e(p):
        return energy(instancia.q, np.isin(np.arange(12), p).astype(float))

    assert picks == min(respostas.values(), key=e)
    assert not registro[instancia.date].agree


def test_full_universe_arm_weights_only_annealed_picks_within_the_sector_cap():
    from core.qpo import CFG_MAX_SHARPE, qubo_selector, selection_then_max_sharpe

    retornos, proventos = _painel_de_8_papeis()
    registro, log = {}, {}
    # k = 5 in 5 sectors: 5 x 20% (name cap) reaches 100%; fewer sectors cannot under the 30% sector cap
    seletor = qubo_selector(SETORES_DOS_8, FULL_UNIVERSE, 5, anneal_solver(SETORES_DOS_8, 1, log), registro)
    pesos = selection_then_max_sharpe(seletor, retornos, proventos, SETORES_DOS_8, CFG_MAX_SHARPE, None)(
        _treino(0.01))
    instancia, picks = next(iter(registro.values()))
    escolhidos = {instancia.tickers[i] for i in picks}
    assert len(instancia.tickers) == 8 and len(escolhidos) == 5
    assert set(pesos[pesos > 0].index) <= escolhidos
    assert pesos.sum() == pytest.approx(1.0)
    assert log[instancia.date].agree
