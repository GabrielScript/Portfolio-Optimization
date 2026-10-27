"""QAOA side of the QPO (qpo_kaggle/qpo_qaoa.py). Runs only where PennyLane and torch are installed (.venv-qpo, Kaggle)."""

import itertools
import math

import numpy as np
import pytest

qml = pytest.importorskip("pennylane")
torch = pytest.importorskip("torch")

from qpo_kaggle.qpo_qaoa import (cost_hamiltonian, dicke_state, energy, open_device, qaoa_qnodes,  # noqa: E402
                             qubo_to_ising, solve_qaoa)


def _random_qubo(n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    a = rng.normal(size=(n, n))
    return (a + a.T) / 2.0


def _bits(index: int, n: int) -> np.ndarray:
    # PennyLane: wire 0 is the most significant bit of the basis-state index
    return np.array([(index >> (n - 1 - i)) & 1 for i in range(n)], dtype=float)


def test_dicke_state_is_uniform_over_weight_k_states():
    state = dicke_state(5, 2)
    pesos = np.array([_bits(i, 5).sum() for i in range(32)])
    assert np.linalg.norm(state) == pytest.approx(1.0)
    assert np.allclose(np.abs(state[pesos != 2]), 0.0)
    assert np.allclose(np.abs(state[pesos == 2]) ** 2, 1.0 / math.comb(5, 2))


def test_ising_form_equals_qubo_energy_on_every_bitstring():
    q = _random_qubo(5, seed=1)
    h, j, const = qubo_to_ising(q)
    for bits in itertools.product((0.0, 1.0), repeat=5):
        x = np.array(bits)
        z = 1.0 - 2.0 * x
        assert z @ h + z @ np.triu(j, 1) @ z + const == pytest.approx(energy(q, x))


def test_cost_hamiltonian_diagonal_is_the_normalized_qubo_energy_up_to_a_constant():
    q = _random_qubo(4, seed=2)
    hamiltoniano, escala = cost_hamiltonian(q)
    diagonal = np.real(np.diag(qml.matrix(hamiltoniano, wire_order=range(4))))
    energias = np.array([energy(q, _bits(i, 4)) for i in range(16)])
    assert np.allclose((diagonal - diagonal[0]) * escala, energias - energias[0])


def test_circuit_never_leaves_the_weight_k_subspace():
    q = _random_qubo(6, seed=3)
    _, amostrar = qaoa_qnodes(q, k=3, p=2, device=open_device("lightning.qubit", 6, seed=7), shots=2000)
    angulos = torch.tensor([[0.7, -1.1], [0.4, 0.9]], dtype=torch.float64)
    amostras = np.asarray(amostrar(angulos))
    assert amostras.shape == (2000, 6)
    assert set(amostras.sum(axis=1).tolist()) == {3}


def test_adjoint_gradient_matches_finite_differences():
    q = _random_qubo(6, seed=4)
    custo, _ = qaoa_qnodes(q, k=3, p=2, device=open_device("lightning.qubit", 6, seed=7), shots=10)
    angulos = torch.tensor([[0.3, -0.6], [0.5, 0.2]], dtype=torch.float64, requires_grad=True)
    custo(angulos).backward()
    passo, numerico = 1e-6, np.zeros((2, 2))
    for i, j in itertools.product(range(2), range(2)):
        mais, menos = angulos.detach().clone(), angulos.detach().clone()
        mais[i, j] += passo
        menos[i, j] -= passo
        numerico[i, j] = (float(custo(mais)) - float(custo(menos))) / (2 * passo)
    assert angulos.grad.numpy() == pytest.approx(numerico, abs=1e-6)


def test_training_lowers_the_energy_and_concentrates_on_the_optimum():
    n, k = 8, 3
    q = _random_qubo(n, seed=5)
    resultado = solve_qaoa(q, k, p=2, steps=60, lr=0.05, shots=2000, seed=11, device_name="lightning.qubit")
    feasible = {c: energy(q, np.isin(np.arange(n), c).astype(float)) for c in itertools.combinations(range(n), k)}
    otimo = min(feasible, key=feasible.get)
    assert resultado.expval_final < resultado.expval_inicial
    assert resultado.picks == otimo
    assert resultado.p_optimum > 1.0 / math.comb(n, k)      # better than the uniform Dicke start
    assert resultado.n_infeasible == 0


def test_gpu_required_never_falls_back_to_cpu():
    if torch.cuda.is_available():
        pytest.skip("this machine has CUDA")
    with pytest.raises(RuntimeError, match="GPU"):
        open_device("lightning.gpu", 4, seed=1, require_gpu=True)
    with pytest.raises(RuntimeError, match="GPU"):
        open_device("lightning.qubit", 4, seed=1, require_gpu=True)
