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


# --- strategy layer: selection, then the existing max Sharpe under the fixed rules --------------------------------

from core.estimation import estimar_momentos  # noqa: E402
from core.qpo import (CFG_MAX_SHARPE, exact_picks, qubo_instance, qubo_selector, random_selector,  # noqa: E402
                      recorded_selector, selection_then_max_sharpe)
from core.sinais import conservadora  # noqa: E402
from tests.test_sinais import MES, SETORES_DOS_8, _painel_de_8_papeis, _treino  # noqa: E402


def _score():
    retornos, proventos = _painel_de_8_papeis()
    return conservadora(retornos, proventos, MES, list("ABCDEFGH"))


def test_qubo_instance_is_the_qubo_of_the_top_n_candidates_with_the_ledoit_wolf_cov():
    treino, score = _treino(0.01), _score()
    instancia = qubo_instance(MES, treino, score, SETORES_DOS_8, n=6, k=3)
    nomes = candidates(score, SETORES_DOS_8, 6)
    cov = estimar_momentos(treino[nomes], shrink_mu=False).cov
    esperado = build_qubo(qubo_alphas(score[nomes], cov).to_numpy(), cov.to_numpy(), 3, 1.0)
    assert (instancia.tickers, instancia.k) == (tuple(nomes), 3)
    assert instancia.q == pytest.approx(esperado)


def test_qubo_instance_is_none_when_there_are_no_more_candidates_than_k():
    assert qubo_instance(MES, _treino(0.01), _score(), SETORES_DOS_8, n=6, k=6) is None


def test_exact_arm_weights_only_the_exact_picks_and_records_them():
    # k = 5: 5 x 20% name cap and 5 sectors x 30% can reach 100% (k = 3 cannot, by the fixed rules)
    retornos, proventos = _painel_de_8_papeis()
    registro: dict = {}
    estrategia = selection_then_max_sharpe(qubo_selector(SETORES_DOS_8, n=7, k=5, solve=exact_picks, record=registro),
                                           retornos, proventos, SETORES_DOS_8, CFG_MAX_SHARPE, None)
    treino = _treino(0.01)
    pesos = estrategia(treino)
    instancia, picks = registro[treino.index[-1]]
    escolhidos = {instancia.tickers[i] for i in picks}
    assert (len(picks), picks) == (5, solve_exact(instancia.q, 5).picks)
    assert set(pesos[pesos > 0].index) <= escolhidos
    assert pesos.sum() == pytest.approx(1.0)


def test_recorded_selector_replays_the_exact_arm_and_stops_on_a_missing_date():
    retornos, proventos = _painel_de_8_papeis()
    registro: dict = {}
    exata = selection_then_max_sharpe(qubo_selector(SETORES_DOS_8, n=7, k=5, solve=exact_picks, record=registro),
                                      retornos, proventos, SETORES_DOS_8, CFG_MAX_SHARPE, None)
    treino = _treino(0.01)
    pesos = exata(treino)
    instancia, picks = registro[treino.index[-1]]
    gravadas = {treino.index[-1]: tuple(instancia.tickers[i] for i in picks)}
    replay = selection_then_max_sharpe(recorded_selector(gravadas), retornos, proventos, SETORES_DOS_8,
                                       CFG_MAX_SHARPE, None)
    pd.testing.assert_series_equal(replay(treino), pesos)
    with pytest.raises(KeyError, match="no recorded pick"):
        replay(treino.iloc[:-1])


def test_random_selector_is_reproducible_and_draws_k_of_the_candidates():
    treino, score = _treino(0.01), _score()
    sorteio = random_selector(SETORES_DOS_8, n=6, k=3, seed=4)
    primeira = sorteio(MES, treino, score)
    assert primeira == sorteio(MES, treino, score)
    assert len(primeira) == 3 and set(primeira) <= set(candidates(score, SETORES_DOS_8, 6))
