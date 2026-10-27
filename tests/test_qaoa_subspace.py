"""Exact weight-k subspace QAOA (qpo_kaggle/qaoa_subspace.py) against the PennyLane reference (qpo_kaggle/qpo_qaoa.py)."""

import itertools
import math

import numpy as np
import pytest

qml = pytest.importorskip("pennylane")
torch = pytest.importorskip("torch")

from qpo_kaggle.qaoa_subspace import SubspaceQaoa, solve_qaoa, weight_k_masks  # noqa: E402
from qpo_kaggle.qpo_qaoa import cost_hamiltonian, open_device, qaoa_circuit, qaoa_qnodes  # noqa: E402
from qpo_kaggle.qpo_qaoa import solve_qaoa as solve_reference  # noqa: E402


def _random_qubo(n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    a = rng.normal(size=(n, n))
    return (a + a.T) / 2.0


ANGULOS = np.array([[0.31, -0.72, 0.18], [0.55, 0.12, -0.41]])


def test_weight_k_masks_are_the_ascending_basis_indices_of_weight_k():
    mascaras = weight_k_masks(6, 2)
    assert len(mascaras) == math.comb(6, 2)
    assert list(mascaras) == [i for i in range(64) if bin(i).count("1") == 2]


@pytest.mark.parametrize(("n", "k"), [(6, 3), (8, 3), (7, 5)])
def test_state_equals_pennylane_state_restricted_to_weight_k(n, k):
    # same unitary on its invariant subspace: basis order (wire 0 = MSB), IsingXY sign, ring order and the H_C
    # diagonal (no constant) all show up here, global phase included
    q = _random_qubo(n, seed=n + k)
    circuito = qaoa_circuit(q, k, p=3)

    @qml.qnode(qml.device("default.qubit", wires=n))
    def estado(angles):
        circuito(angles)
        return qml.state()

    cheio = np.asarray(estado(ANGULOS))
    mascaras = weight_k_masks(n, k)
    assert np.allclose(np.delete(cheio, mascaras), 0.0, atol=1e-12)
    sub = SubspaceQaoa(q, k).state(torch.tensor(ANGULOS)).detach().numpy()
    assert sub == pytest.approx(cheio[mascaras], abs=1e-12)


def test_expectation_and_gradient_equal_pennylane_adjoint():
    q, k = _random_qubo(8, seed=21), 3
    custo, _ = qaoa_qnodes(q, k, p=3, device=open_device("lightning.qubit", 8, seed=1), shots=10)
    referencia = torch.tensor(ANGULOS, requires_grad=True)
    valor_ref = custo(referencia)
    valor_ref.backward()
    angulos = torch.tensor(ANGULOS, requires_grad=True)
    valor = SubspaceQaoa(q, k).expval(angulos)
    valor.backward()
    assert float(valor) == pytest.approx(float(valor_ref), abs=1e-12)
    assert angulos.grad.numpy() == pytest.approx(referencia.grad.numpy(), abs=1e-10)


def test_expectation_is_the_scaled_cost_diagonal_of_the_reference():
    q, k = _random_qubo(5, seed=22), 2
    hamiltoniano, _ = cost_hamiltonian(q)
    diagonal = np.real(np.diag(qml.matrix(hamiltoniano, wire_order=range(5))))
    assert SubspaceQaoa(q, k).cost_diagonal.numpy() == pytest.approx(diagonal[weight_k_masks(5, k)], abs=1e-12)


def test_adam_trajectory_matches_the_pennylane_reference():
    q, k = _random_qubo(6, seed=23), 3
    rapido = solve_qaoa(q, k, p=2, steps=40, lr=0.05, shots=500, seed=5)
    lento = solve_reference(q, k, p=2, steps=40, lr=0.05, shots=500, seed=5, device_name="lightning.qubit")
    assert np.asarray(rapido.angles) == pytest.approx(np.asarray(lento.angles), abs=1e-8)
    assert rapido.expval_final == pytest.approx(lento.expval_final, abs=1e-10)


def test_solver_finds_the_optimum_samples_only_weight_k_and_is_reproducible():
    n, k = 9, 4
    q = _random_qubo(n, seed=24)
    energias = {c: float(np.isin(np.arange(n), c) @ q @ np.isin(np.arange(n), c))
                for c in itertools.combinations(range(n), k)}
    otimo = min(energias, key=energias.get)
    a = solve_qaoa(q, k, p=2, steps=60, lr=0.05, shots=2000, seed=11)
    b = solve_qaoa(q, k, p=2, steps=60, lr=0.05, shots=2000, seed=11)
    assert (a.picks, a.n_infeasible, a.energy) == (otimo, 0, pytest.approx(energias[otimo]))
    assert a.expval_final < a.expval_inicial and a.p_optimum > 1.0 / math.comb(n, k)
    assert (a.picks, a.angles, a.p_optimum) == (b.picks, b.angles, b.p_optimum)
