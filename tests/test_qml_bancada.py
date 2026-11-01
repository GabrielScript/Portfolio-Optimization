"""The solver bench of the hybrid QPO (docs/superpowers/specs/2026-11-01-qpo-hibrido-design.md, section 6). Run in .venv-qpo."""

import itertools

import numpy as np
import pytest

pytest.importorskip("pennylane")

from qpo_kaggle import qml_bancada as b  # noqa: E402


def _q(n: int = 6, semente: int = 0) -> np.ndarray:
    bruta = np.random.default_rng(semente).normal(size=(n, n))
    return (bruta + bruta.T) / 2.0


def test_the_checker_is_the_definition():
    q = _q()
    x = np.array([[1, 0, 1, 0, 0, 0], [0, 0, 0, 0, 0, 1]], dtype=float)
    assert b.pontuar(q, x) == pytest.approx([x[0] @ q @ x[0], x[1] @ q @ x[1]])


def test_the_answer_key_agrees_with_plain_enumeration():
    q, k = _q(), 3
    energias = [(lambda x: x @ q @ x)(np.isin(range(6), c).astype(float)) for c in itertools.combinations(range(6), k)]
    gabarito = b.exato(q, k)
    assert gabarito.minimo == pytest.approx(min(energias)) and gabarito.maximo == pytest.approx(max(energias))
    assert gabarito.x.sum() == k and b.pontuar(q, gabarito.x[None])[0] == pytest.approx(min(energias))


def test_the_penalty_makes_the_feasible_optimum_the_global_one():
    q, k = _q(), 3
    diagonal = b.energias_com_penalidade(q, k, b.penalidade_de_hodson(q, k, semente=1))
    melhor = b.todos_os_estados(6)[int(np.argmin(diagonal))]
    assert melhor.sum() == k and b.pontuar(q, melhor[None])[0] == pytest.approx(b.exato(q, k).minimo)


def test_soft_qaoa_starts_uniform_and_stays_a_distribution():
    diagonal = b.energias_com_penalidade(_q(4), 2, 1.0)
    uniforme = b.probabilidades_qaoa_soft(diagonal, np.zeros((2, 3)))
    assert uniforme == pytest.approx(np.full(16, 1 / 16))
    assert b.probabilidades_qaoa_soft(diagonal, np.full((2, 3), 0.4)).sum() == pytest.approx(1.0)


def test_soft_qaoa_is_the_textbook_circuit_on_one_qubit():
    # H, then exp(-i g diag(0, 1)), then RX(2 b): P(1) = (1 + sin(g) sin(2 b)) / 2 up to the sign of the phase convention
    g, beta = 0.7, 0.3
    p = b.probabilidades_qaoa_soft(np.array([0.0, 1.0]), np.array([[g], [beta]]))
    assert p[1] == pytest.approx((1.0 - np.sin(g) * np.sin(2 * beta)) / 2.0) or \
        p[1] == pytest.approx((1.0 + np.sin(g) * np.sin(2 * beta)) / 2.0)


def test_soft_qaoa_runs_on_sixteen_qubits():
    # PennyLane's DiagonalQubitUnitary builds the dense 2^16 x 2^16 matrix: 68.7 GB, a crash on the bench's real size
    diagonal = np.random.default_rng(0).normal(size=2 ** 16)
    assert b.probabilidades_qaoa_soft(diagonal, np.full((2, 3), 0.4)).sum() == pytest.approx(1.0)


def _soft_qaoa_de_referencia(n: int, diagonal):
    """The bench's circuit gate by gate in PennyLane, usable only on small n."""
    import pennylane as qml
    import torch

    @qml.qnode(qml.device("default.qubit", wires=n), interface="torch", diff_method="backprop")
    def circuito(angulos):
        for fio in range(n):
            qml.Hadamard(wires=fio)
        for gama, beta in zip(angulos[0], angulos[1], strict=True):
            qml.DiagonalQubitUnitary(torch.exp(-1j * gama * diagonal), wires=range(n))
            for fio in range(n):
                qml.RX(2.0 * beta, wires=fio)
        return qml.probs(wires=range(n))
    return circuito


def test_soft_qaoa_gives_the_probabilities_and_gradient_of_pennylanes_circuit():
    import torch
    n = 5
    diagonal = torch.tensor(np.random.default_rng(1).normal(size=2 ** n), dtype=torch.float64)
    custo = diagonal - diagonal.mean()
    inicio = np.random.default_rng(2).normal(size=(2, 3))
    resultados = []
    for no in (b._no_qaoa_soft(n, diagonal), _soft_qaoa_de_referencia(n, diagonal)):
        angulos = torch.tensor(inicio, dtype=torch.float64, requires_grad=True)
        probabilidades = no(angulos)
        (probabilidades * custo).sum().backward()
        resultados.append((probabilidades.detach().numpy(), angulos.grad.numpy()))
    (p_nosso, g_nosso), (p_ref, g_ref) = resultados
    assert p_nosso == pytest.approx(p_ref, abs=1e-12)
    assert g_nosso == pytest.approx(g_ref, abs=1e-12)


def test_vqe_at_zero_angles_is_the_zero_state_and_training_lowers_the_energy():
    diagonal = b.energias_com_penalidade(_q(4), 2, 1.0)
    assert b.probabilidades_vqe(np.zeros((4, 4)))[0] == pytest.approx(1.0)
    treinada = b.treinar_vqe(diagonal, camadas=3, passos=60, taxa=0.1, semente=0)
    aleatoria = b.probabilidades_vqe(np.random.default_rng(0).normal(0.0, 0.1, (4, 4)))
    assert treinada @ diagonal < aleatoria @ diagonal


def test_training_soft_qaoa_lowers_the_energy_from_uniform():
    diagonal = b.energias_com_penalidade(_q(4), 2, 1.0)
    assert b.treinar_qaoa_soft(diagonal, p=3, passos=60, taxa=0.05) @ diagonal < diagonal.mean()


def test_hard_qaoa_only_ever_gives_the_right_size():
    q, k = _q(), 3
    probabilidades, estados = b.treinar_qaoa_hard(q, k, p=3, passos=30, taxa=0.05)
    assert probabilidades.sum() == pytest.approx(1.0) and (estados.sum(axis=1) == k).all()


def test_samples_are_reproducible_and_follow_the_distribution():
    estados = b.todos_os_estados(2)
    probabilidades = np.array([0.0, 1.0, 0.0, 0.0])
    assert (b.amostrar(probabilidades, estados, 5, semente=0) == estados[1]).all()
    uniforme = np.full(4, 0.25)
    assert np.array_equal(b.amostrar(uniforme, estados, 50, semente=3), b.amostrar(uniforme, estados, 50, semente=3))


def test_uniform_feasible_samples_have_the_right_size():
    amostras = b.uniforme_viavel(6, 3, 200, semente=0)
    assert amostras.shape == (200, 6) and (amostras.sum(axis=1) == 3).all() and len(np.unique(amostras, axis=0)) > 5


def test_repair_reaches_the_size_greedily_and_leaves_a_feasible_sample_alone():
    q, k = _q(), 3
    feita = b.reparar(q, np.array([[1, 1, 1, 1, 1, 0], [0, 0, 0, 0, 0, 0], [1, 0, 1, 0, 1, 0]], dtype=float), k)
    assert (feita.sum(axis=1) == k).all() and np.array_equal(feita[2], [1, 0, 1, 0, 1, 0])
    # the first removal is the one that lowers the energy most
    x = np.array([1, 1, 1, 1, 0, 0], dtype=float)
    candidatas = [(lambda y: y @ q @ y)(x - np.eye(6)[i]) for i in range(4)]
    assert b.pontuar(q, b.reparar(q, x[None], 3))[0] == pytest.approx(min(candidatas))


def test_vqe_constrained_takes_the_best_with_turnover_among_the_ten_lowest_without_it():
    n, k = 6, 3
    q = _q()
    amostras = b.uniforme_viavel(n, k, 500, semente=0)
    giro = np.diag(np.array([-100.0, 0, 0, 0, 0, 0]))     # turnover term that wants name 0
    escolha = b.vqe_restrito(amostras, q, q + giro, k)
    unicos = np.unique(amostras, axis=0)
    dez = unicos[np.argsort(b.pontuar(q, unicos), kind="stable")[:10]]
    assert b.pontuar(q + giro, escolha[None])[0] == pytest.approx(b.pontuar(q + giro, dez).min())


def test_the_measures_count_size_and_optimum_on_shots():
    q, k = _q(), 3
    gabarito = b.exato(q, k)
    otimo = gabarito.x
    fora = np.array([1, 1, 1, 1, 0, 0], dtype=float)
    medida = b.medir(np.vstack([otimo, otimo, otimo, fora]), q, k, gabarito)
    assert medida["viavel"] == pytest.approx(0.75) and medida["p_otimo"] == pytest.approx(0.75)
    assert medida["razao_melhor"] == pytest.approx(1.0) and medida["razao_media"] == pytest.approx(1.0)
