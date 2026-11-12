"""Diversity functionals of a correlation matrix (log det, von Neumann) and the shrunk correlation estimate."""

import itertools

import numpy as np
import pandas as pd
import pytest

from core.diversidade import log_det, shrunk_correlation, von_neumann


def _corr(n: int, seed: int) -> np.ndarray:
    fatores = np.random.default_rng(seed).normal(size=(4 * n, n))
    fatores[:, 1:] += 0.8 * fatores[:, [0]]
    return np.corrcoef(fatores, rowvar=False)


def test_log_det_of_a_pair_is_log_one_minus_rho_squared():
    assert log_det(np.array([[1.0, 0.6], [0.6, 1.0]])) == pytest.approx(np.log(1 - 0.36))


def test_log_det_is_the_sum_of_log_unexplained_variance_in_any_order():
    c = _corr(7, 2)
    for ordem in ((0, 3, 5, 6), (6, 0, 5, 3)):
        soma = 0.0
        for posicao, j in enumerate(ordem):
            antes = list(ordem[:posicao])
            r2 = c[j, antes] @ np.linalg.solve(c[np.ix_(antes, antes)], c[antes, j]) if antes else 0.0
            soma += np.log(1.0 - r2)
        assert log_det(c[np.ix_(ordem, ordem)]) == pytest.approx(soma)


@pytest.mark.parametrize("funcional", [log_det, von_neumann])
def test_diversity_is_zero_without_correlation_and_negative_with_it(funcional):
    c = _corr(9, 1)
    assert funcional(np.eye(5)) == pytest.approx(0.0, abs=1e-12)
    assert all(funcional(c[np.ix_(s, s)]) < 0 for s in itertools.combinations(range(9), 4))


def test_von_neumann_of_one_common_factor_is_minus_k_log_k():
    assert von_neumann(np.ones((6, 6))) == pytest.approx(-6 * np.log(6))
    assert log_det(np.ones((6, 6))) == -np.inf


def test_von_neumann_is_k_times_the_entropy_of_the_density_matrix_minus_log_k():
    qml = pytest.importorskip("pennylane")
    c = _corr(8, 5)                                   # 8 names: a 3-qubit density matrix
    entropia = float(qml.math.vn_entropy(c / 8, indices=[0, 1, 2]))
    assert von_neumann(c) == pytest.approx(8 * (entropia - np.log(8)))


@pytest.mark.parametrize("funcional", [log_det, von_neumann])
def test_functionals_take_a_stack_of_matrices(funcional):
    c = _corr(6, 3)
    pilha = np.stack([c[np.ix_(s, s)] for s in ((0, 1, 2), (3, 4, 5))])
    assert funcional(pilha) == pytest.approx([funcional(pilha[0]), funcional(pilha[1])])


def test_shrunk_correlation_shrinks_every_name_alike_and_keeps_unit_diagonal():
    rng = np.random.default_rng(0)
    retornos = rng.normal(size=(400, 5)) + rng.normal(size=(400, 1))
    retornos *= np.array([0.5, 1.0, 2.0, 4.0, 8.0])            # very different volatilities
    c = shrunk_correlation(pd.DataFrame(retornos))
    amostral = np.corrcoef(retornos, rowvar=False)
    fora = ~np.eye(5, dtype=bool)
    razao = c[fora] / amostral[fora]
    assert np.diag(c) == pytest.approx(1.0)
    assert razao == pytest.approx(razao[0]) and 0.0 < razao[0] <= 1.0


# the state of a weighted set of names and its entropy (docs/superpowers/specs/2026-11-11-fermionic-size-grid-design.md)
def _simplex(k: int, seed: int) -> np.ndarray:
    pesos = np.random.default_rng(seed).uniform(0.2, 1.0, size=k)
    return pesos / pesos.sum()


def _surprisals(corr: np.ndarray, pesos: np.ndarray) -> np.ndarray:
    """-<c_i| log rho |c_i> of each name, with rho = A diag(w) A and A the symmetric square root of C."""
    valores, vetores = np.linalg.eigh(corr)
    raiz = (vetores * np.sqrt(valores)) @ vetores.T
    autovalores, base = np.linalg.eigh(raiz @ np.diag(pesos) @ raiz)
    log_rho = (base * np.log(autovalores)) @ base.T
    return -np.einsum("ji,jk,ki->i", raiz, log_rho, raiz)


def test_state_entropy_is_the_shannon_entropy_of_the_weights_only_without_correlation():
    from core.diversidade import state_entropy
    pesos = _simplex(6, 0)
    shannon = float(-(pesos * np.log(pesos)).sum())
    assert state_entropy(np.eye(6), pesos) == pytest.approx(shannon)
    assert all(state_entropy(_corr(6, seed), pesos) < shannon - 1e-6 for seed in range(5))


def test_state_entropy_at_equal_weights_is_the_von_neumann_functional_of_the_set():
    from core.diversidade import state_entropy
    c = _corr(8, 3)
    assert state_entropy(c, np.full(8, 1 / 8)) == pytest.approx(von_neumann(c) / 8 + np.log(8))


def test_state_entropy_is_concave_in_the_weights():
    from core.diversidade import state_entropy
    c = _corr(7, 4)
    for seed in range(20):
        a, b = _simplex(7, seed), _simplex(7, 100 + seed)
        assert state_entropy(c, (a + b) / 2) >= (state_entropy(c, a) + state_entropy(c, b)) / 2 - 1e-12


def test_eic_weights_make_every_name_as_surprising_as_the_portfolio_and_beat_any_other_weights():
    from core.diversidade import eic_weights, state_entropy
    for seed in range(4):
        c = _corr(8, seed)
        pesos = eic_weights(c, teto=1.0)
        entropia = state_entropy(c, pesos)
        assert pesos.sum() == pytest.approx(1.0) and (pesos > 0).all()
        assert _surprisals(c, pesos) == pytest.approx(entropia, abs=1e-5)
        assert entropia >= state_entropy(c, np.full(8, 1 / 8)) - 1e-10
        assert all(entropia >= state_entropy(c, _simplex(8, outra)) - 1e-10 for outra in range(30))


def test_eic_weights_give_two_near_duplicates_the_weight_of_one_name():
    from core.diversidade import eic_weights
    c = np.array([[1.0, 0.98, 0.0], [0.98, 1.0, 0.0], [0.0, 0.0, 1.0]])
    pesos = eic_weights(c, teto=1.0)
    assert pesos[0] == pytest.approx(pesos[1], abs=1e-5) and 0.45 < pesos[2] < 0.5
    assert eic_weights(np.eye(5), teto=1.0) == pytest.approx(np.full(5, 0.2), abs=1e-6)


def test_eic_weights_respect_the_name_cap_and_the_group_caps():
    from core.diversidade import eic_weights
    c = np.array([[1.0, 0.98, 0.0, 0.0], [0.98, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.1], [0.0, 0.0, 0.1, 1.0]])
    pesos = eic_weights(c, teto=0.30, grupos=[([2, 3], 0.45)])
    assert pesos.sum() == pytest.approx(1.0) and pesos.max() <= 0.30 + 1e-8
    assert pesos[2] + pesos[3] <= 0.45 + 1e-8 and pesos[0] == pytest.approx(0.275, abs=1e-4)
