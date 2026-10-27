"""QPO (docs/superpowers/specs/2026-10-27-qpo-design.md): QUBO, exact solver, Ising form, candidates and alphas."""

import itertools

import numpy as np
import pandas as pd
import pytest

from core.qpo import (IC0, approximation_ratio, build_qubo, candidates, energy, qubo_alphas, qubo_to_ising,
                      solve_exact)


def _random_qubo(n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    a = rng.normal(size=(n, n))
    return (a + a.T) / 2.0


def _brute_force(q: np.ndarray, k: int) -> tuple[tuple[int, ...], float, float]:
    n = len(q)
    energies = {c: energy(q, np.isin(np.arange(n), c).astype(float)) for c in itertools.combinations(range(n), k)}
    best = min(energies, key=energies.get)
    return best, energies[best], max(energies.values())


def test_ising_form_equals_qubo_energy_on_every_bitstring():
    # x = (1 - z) / 2: z = +1 is x = 0 (|0>), z = -1 is x = 1 (|1>)
    q = _random_qubo(6, seed=1)
    h, j, const = qubo_to_ising(q)
    for bits in itertools.product((0.0, 1.0), repeat=6):
        x = np.array(bits)
        z = 1.0 - 2.0 * x
        assert z @ h + z @ np.triu(j, 1) @ z + const == pytest.approx(energy(q, x))


@pytest.mark.parametrize(("n", "k", "seed"), [(6, 3, 2), (9, 4, 3), (11, 5, 4), (12, 1, 5), (12, 12, 6), (13, 7, 7)])
def test_exact_solver_matches_brute_force(n, k, seed):
    q = _random_qubo(n, seed)
    best, e_best, e_worst = _brute_force(q, k)
    solution = solve_exact(q, k)
    assert (solution.picks, solution.energy, solution.worst) == (best, pytest.approx(e_best), pytest.approx(e_worst))


def test_exact_solver_rejects_impossible_cardinality():
    with pytest.raises(ValueError, match="k"):
        solve_exact(_random_qubo(5, seed=8), 6)


def test_energy_accepts_a_batch_of_bitstrings():
    q = _random_qubo(4, seed=9)
    xs = np.array([[1, 0, 1, 0], [0, 1, 1, 1]], dtype=float)
    assert energy(q, xs) == pytest.approx([energy(q, xs[0]), energy(q, xs[1])])


def test_approximation_ratio_is_one_at_the_optimum_and_zero_at_the_worst():
    assert (approximation_ratio(-3.0, -3.0, 5.0), approximation_ratio(5.0, -3.0, 5.0),
            approximation_ratio(1.0, -3.0, 5.0)) == (1.0, 0.0, pytest.approx(0.5))


def test_qubo_is_negative_equal_weight_mean_variance_utility():
    # E(x) = -alpha'x / k + (lambda / 2) x'Sigma x / k^2: the utility of the equal-weight portfolio w = x / k
    rng = np.random.default_rng(10)
    a = rng.normal(size=(5, 5)) * 0.1
    cov = a @ a.T
    alpha = rng.normal(size=5) * 0.02
    x = np.array([1, 1, 0, 1, 0], dtype=float)
    w = x / 3
    q = build_qubo(alpha, cov, k=3, risk_aversion=1.0)
    assert energy(q, x) == pytest.approx(-(alpha @ w) + 0.5 * w @ cov @ w)
    assert np.allclose(q, q.T)


def test_candidates_follow_the_score_one_class_per_issuer_skip_no_sector_and_no_sector_cap():
    score = pd.Series({"PETR4": 1.9, "PETR3": 1.8, "VALE3": 1.7, "XXXX3": 1.6, "ITUB4": 1.5, "BBDC4": 1.4,
                       "SANB11": 1.3, "BBAS3": 1.2})
    sectors = {"PETR4": "oil", "PETR3": "oil", "VALE3": "mining", "ITUB4": "banks", "BBDC4": "banks",
               "SANB11": "banks", "BBAS3": "banks"}
    assert candidates(score, sectors, n=6) == ["PETR4", "VALE3", "ITUB4", "BBDC4", "SANB11", "BBAS3"]
    assert candidates(score, sectors, n=3) == ["PETR4", "VALE3", "ITUB4"]


def test_qubo_alphas_are_ic_times_vol_times_zscore():
    score = pd.Series({"A": 2.0, "B": 1.0, "C": 0.0})
    cov = pd.DataFrame(np.diag([0.04, 0.09, 0.16]), index=list("ABC"), columns=list("ABC"))
    z = (score - score.mean()) / score.std(ddof=0)
    expected = IC0 * np.sqrt(np.diag(cov.values)) * z.values
    assert IC0 == 0.059
    assert qubo_alphas(score, cov).values == pytest.approx(expected)


def test_qubo_alphas_are_zero_when_every_score_ties():
    score = pd.Series({"A": 1.0, "B": 1.0})
    cov = pd.DataFrame(np.eye(2) * 0.04, index=list("AB"), columns=list("AB"))
    assert qubo_alphas(score, cov).tolist() == [0.0, 0.0]
