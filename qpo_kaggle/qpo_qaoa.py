"""
qpo_qaoa.py - QAOA solver of the QPO (docs/superpowers/specs/2026-10-27-qpo-design.md, section 7).

Self-contained (numpy, PennyLane 0.45.1, torch) so it runs alone on Kaggle; the numpy helpers repeat the ones in
``core/qpo.py`` and the tests check them against x'Qx directly.

Ansatz: Dicke start |D(n, k)> and an XY-ring mixer (IsingXY on the ring edges), so every state stays in the
|x| = k subspace and no penalty is needed (Hadfield et al. 2019). Cost layer: RZ(2 gamma h_i) and
IsingZZ(2 gamma J_ij) = exp(-i gamma H_C) with H_C the Ising form scaled to max |coefficient| = 1, which moves no
minimum. Readout: the best feasible sampled bitstring by the classical x'Qx, never the lowest mean energy.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass

import numpy as np
import pennylane as qml
import torch

# PennyLane's torch interface returns results in torch's default dtype; float32 would round the cost and its
# gradient (the simulators work in complex128). Process-wide on purpose: this module is the whole Kaggle job.
torch.set_default_dtype(torch.float64)


@dataclass(frozen=True)
class QaoaResult:
    picks: tuple[int, ...]       # indices of the chosen names (wire i = name i), ascending
    energy: float                # x'Qx of the picks
    expval_inicial: float        # <H_C> (scaled) at the starting angles
    expval_final: float          # <H_C> (scaled) at the trained angles
    p_optimum: float             # share of the shots on the returned pick
    n_infeasible: int            # shots with |x| != k (0 by construction; checked)
    angles: list[list[float]]    # [gammas, betas]
    seconds: float
    device: str


def energy(q: np.ndarray, x: np.ndarray) -> float | np.ndarray:
    """x'Qx for one bitstring (n,) or a batch (m, n)."""
    valores = np.einsum("...i,ij,...j->...", x, q, x)
    return float(valores) if np.ndim(valores) == 0 else valores


def qubo_to_ising(q: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """(h, J, const) with x'Qx = h'z + sum_{i<j} J_ij z_i z_j + const for x = (1 - z) / 2 (z = +1 is |0>)."""
    s = (q + q.T) / 2.0
    diagonal = np.diag(s)
    fora = s - np.diag(diagonal)
    h = -diagonal / 2.0 - fora.sum(axis=1) / 2.0
    return h, fora / 2.0, float(diagonal.sum() / 2.0 + fora.sum() / 4.0)


def dicke_state(n: int, k: int) -> np.ndarray:
    """Uniform superposition of the C(n, k) basis states of Hamming weight k (2^n complex128 amplitudes)."""
    pesos = np.bitwise_count(np.arange(2 ** n, dtype=np.uint32))
    estado = np.zeros(2 ** n, dtype=np.complex128)
    estado[pesos == k] = 1.0 / math.sqrt(math.comb(n, k))
    return estado


def _scaled_ising(q: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    h, j, _ = qubo_to_ising(q)
    escala = float(max(np.abs(h).max(), np.abs(j).max()))
    escala = escala if escala > 0 else 1.0
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


def _ring(n: int) -> list[tuple[int, int]]:
    return [(0, 1)] if n == 2 else [(i, (i + 1) % n) for i in range(n)]


def qaoa_circuit(q: np.ndarray, k: int, p: int):
    """The quantum function of the ansatz; ``angles`` has shape (2, p): row 0 the gammas, row 1 the betas."""
    n = len(q)
    h, j, _ = _scaled_ising(q)
    inicio = dicke_state(n, k)
    pares = [(a, b) for a in range(n) for b in range(a + 1, n) if j[a, b] != 0.0]
    anel = _ring(n)

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


def _linear_ramp(p: int, passo: float = 0.8) -> np.ndarray:
    t = (np.arange(p) + 0.5) / p
    return np.vstack([t * passo, (1.0 - t) * passo])


def solve_qaoa(q: np.ndarray, k: int, p: int, steps: int, lr: float, shots: int, seed: int, device_name: str,
               init: np.ndarray | None = None, require_gpu: bool = False) -> QaoaResult:
    """Adam on <H_C> from ``init`` (default: a linear ramp), then ``shots`` samples scored by x'Qx."""
    inicio_relogio = time.perf_counter()
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = open_device(device_name, len(q), seed, require_gpu)
    custo, amostrar = qaoa_qnodes(q, k, p, device, shots)
    angulos = torch.tensor(_linear_ramp(p) if init is None else np.asarray(init, dtype=float),
                           dtype=torch.float64, requires_grad=True)
    otimizador = torch.optim.Adam([angulos], lr=lr)
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
