"""
qpo_qaoa.py - PennyLane reference implementation of the QPO's QAOA (design doc, section 7).

The production solver is ``qaoa_subspace.py`` (the same circuit simulated exactly on its weight-k subspace); this
module is the full-statevector reference it is tested against, and what the Kaggle T4 run ``qpo-solve-0`` used.

Ansatz: Dicke start |D(n, k)> and an XY-ring mixer (IsingXY on the ring edges), so every state stays in the
|x| = k subspace and no penalty is needed (Hadfield et al. 2019). Cost layer: RZ(2 gamma h_i) and
IsingZZ(2 gamma J_ij) = exp(-i gamma H_C) with H_C the Ising form scaled to max |coefficient| = 1, which moves no
minimum. Readout: the best feasible sampled bitstring by the classical x'Qx, never the lowest mean energy.
"""

from __future__ import annotations

import math
import time
from collections.abc import Iterator
from contextlib import contextmanager

import numpy as np
import pennylane as qml
import torch

from qpo_kaggle.qaoa_subspace import QaoaResult, energy, ising_scale, linear_ramp, qubo_to_ising, ring_edges

__all__ = ["QaoaResult", "cost_hamiltonian", "dicke_state", "energy", "float64_default", "open_device",
           "qaoa_circuit", "qaoa_qnodes", "qubo_to_ising", "solve_qaoa"]


@contextmanager
def float64_default() -> Iterator[None]:
    """torch's default dtype at float64 inside the block, restored after. PennyLane's torch interface returns
    results (and the adjoint gradient) in the default dtype; float32 would round the cost below the gaps between
    neighbouring subsets. Wrap both the QNode calls and ``backward()``."""
    anterior = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    try:
        yield
    finally:
        torch.set_default_dtype(anterior)


def dicke_state(n: int, k: int) -> np.ndarray:
    """Uniform superposition of the C(n, k) basis states of Hamming weight k (2^n complex128 amplitudes)."""
    pesos = np.bitwise_count(np.arange(2 ** n, dtype=np.uint32))
    estado = np.zeros(2 ** n, dtype=np.complex128)
    estado[pesos == k] = 1.0 / math.sqrt(math.comb(n, k))
    return estado


def _scaled_ising(q: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    h, j, _ = qubo_to_ising(q)
    escala = ising_scale(q)
    return h / escala, j / escala, escala


def cost_hamiltonian(q: np.ndarray) -> tuple[qml.operation.Operator, float]:
    """H_C = (h'Z + sum_{i<j} J_ij Z_i Z_j) / scale and the scale: scale x H_C = x'Qx - const on the diagonal."""
    h, j, escala = _scaled_ising(q)
    n = len(q)
    coeficientes = [float(h[i]) for i in range(n)]
    operadores = [qml.Z(i) for i in range(n)]
    for a in range(n):
        for b in range(a + 1, n):
            if j[a, b] != 0.0:
                coeficientes.append(float(j[a, b]))
                operadores.append(qml.Z(a) @ qml.Z(b))
    return qml.Hamiltonian(coeficientes, operadores), escala


def open_device(name: str, n: int, seed: int, require_gpu: bool = False):
    """The simulator; with ``require_gpu`` anything but lightning.gpu on a visible CUDA device is an error, never a
    silent CPU fallback."""
    if require_gpu and name != "lightning.gpu":
        raise RuntimeError(f"GPU required but the device is {name}")
    if require_gpu and not torch.cuda.is_available():
        raise RuntimeError("GPU required but CUDA is not available")
    device = qml.device(name, wires=n, seed=seed)
    if require_gpu and device.name != "lightning.gpu":
        raise RuntimeError(f"GPU required but PennyLane opened {device.name}")
    return device


def qaoa_circuit(q: np.ndarray, k: int, p: int):
    """The quantum function of the ansatz; ``angles`` has shape (2, p): row 0 the gammas, row 1 the betas."""
    n = len(q)
    h, j, _ = _scaled_ising(q)
    inicio = dicke_state(n, k)
    pares = [(a, b) for a in range(n) for b in range(a + 1, n) if j[a, b] != 0.0]
    anel = ring_edges(n)

    def _camadas(angles):
        qml.StatePrep(inicio, wires=range(n))
        for camada in range(p):
            gamma, beta = angles[0, camada], angles[1, camada]
            for i in range(n):
                qml.RZ(2.0 * gamma * h[i], wires=i)
            for a, b in pares:
                qml.IsingZZ(2.0 * gamma * j[a, b], wires=[a, b])
            for a, b in anel:
                qml.IsingXY(2.0 * beta, wires=[a, b])
    return _camadas


def qaoa_qnodes(q: np.ndarray, k: int, p: int, device, shots: int):
    """(cost, sample): cost(angles) = <H_C> (torch, adjoint gradient); sample(angles) = ``shots`` bitstrings
    (shots, n). ``angles`` has shape (2, p): row 0 the gammas, row 1 the betas."""
    n = len(q)
    hamiltoniano, _ = cost_hamiltonian(q)
    _camadas = qaoa_circuit(q, k, p)

    @qml.qnode(device, interface="torch", diff_method="adjoint")
    def custo(angles):
        _camadas(angles)
        return qml.expval(hamiltoniano)

    @qml.set_shots(shots)
    @qml.qnode(device, interface="torch", diff_method=None)
    def amostrar(angles):
        _camadas(angles)
        return qml.sample(wires=range(n))

    return custo, amostrar


def solve_qaoa(q: np.ndarray, k: int, p: int, steps: int, lr: float, shots: int, seed: int, device_name: str,
               init: np.ndarray | None = None, require_gpu: bool = False) -> QaoaResult:
    """Adam on <H_C> from ``init`` (default: a linear ramp), then ``shots`` samples scored by x'Qx. The device's
    seed drives the sampler; nothing global is seeded, and torch's default dtype is restored on return."""
    inicio_relogio = time.perf_counter()
    device = open_device(device_name, len(q), seed, require_gpu)
    custo, amostrar = qaoa_qnodes(q, k, p, device, shots)
    angulos = torch.tensor(linear_ramp(p) if init is None else np.asarray(init, dtype=float),
                           dtype=torch.float64, requires_grad=True)
    otimizador = torch.optim.Adam([angulos], lr=lr)
    with float64_default():
        expval_inicial = float(custo(angulos.detach()))
        for _ in range(steps):
            otimizador.zero_grad(set_to_none=True)
            perda = custo(angulos)
            perda.backward()
            otimizador.step()
        finais = angulos.detach()
        expval_final = float(custo(finais))
        amostras = np.asarray(amostrar(finais), dtype=np.int64).reshape(shots, len(q))
    viaveis = amostras[amostras.sum(axis=1) == k]
    if len(viaveis) == 0:
        raise RuntimeError("no feasible sample (|x| = k): the XY mixer should never allow this")
    unicos, contagens = np.unique(viaveis, axis=0, return_counts=True)
    energias = energy(q, unicos.astype(float))
    melhor = int(np.argmin(energias))
    return QaoaResult(tuple(int(i) for i in np.flatnonzero(unicos[melhor])), float(energias[melhor]),
                      expval_inicial, expval_final, float(contagens[melhor] / shots), int(shots - len(viaveis)),
                      finais.tolist(), time.perf_counter() - inicio_relogio, device.name)
