"""
qaoa_subspace.py - production QAOA solver of the QPO: the exact statevector restricted to the weight-k subspace.

The ansatz of ``qpo_qaoa.py`` (Dicke start, cost layer exp(-i gamma H_C), XY-ring mixer of IsingXY gates) never
leaves span{|x> : |x| = k}: the Dicke state lives there, H_C is diagonal and each IsingXY only mixes |01> and |10>
of its pair. So the C(n, k) amplitudes of that subspace carry the whole state (15,504 at n = 20, k = 15, against
2^20 = 1,048,576), the cost layer is one elementwise phase and each mixer gate a 2x2 rotation on precomputed index
pairs. Same unitary and same gradient (torch autograd instead of the adjoint method): tests/test_qaoa_subspace.py
checks it amplitude by amplitude against the PennyLane reference. Numpy and torch only, CPU or CUDA, complex128.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass

import numpy as np
import torch


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


def ising_scale(q: np.ndarray) -> float:
    """max |h_i|, |J_ij|: H_C is the Ising form divided by it, so gamma is O(1) whatever the units of Q."""
    h, j, _ = qubo_to_ising(q)
    escala = float(max(np.abs(h).max(), np.abs(j).max()))
    return escala if escala > 0 else 1.0


def linear_ramp(p: int, passo: float = 0.8) -> np.ndarray:
    """Starting angles (2, p): gammas rise and betas fall linearly (annealing-like)."""
    t = (np.arange(p) + 0.5) / p
    return np.vstack([t * passo, (1.0 - t) * passo])


def weight_k_masks(n: int, k: int) -> np.ndarray:
    """Basis indices of Hamming weight k, ascending (PennyLane's order: wire i is bit n - 1 - i of the index)."""
    return np.flatnonzero(np.bitwise_count(np.arange(2 ** n, dtype=np.uint32)) == k)


def ring_edges(n: int) -> list[tuple[int, int]]:
    """The XY mixer's edges in gate order: (0, 1), (1, 2), ..., (n - 1, 0)."""
    return [(0, 1)] if n == 2 else [(i, (i + 1) % n) for i in range(n)]


class SubspaceQaoa:
    """The QAOA state of one QUBO on its weight-k subspace; ``angles`` has shape (2, p): gammas, then betas."""

    def __init__(self, q: np.ndarray, k: int, device: str | torch.device = "cpu") -> None:
        n = len(q)
        self.k = k
        self.device = torch.device(device)
        self.masks = weight_k_masks(n, k)
        self.bits = (self.masks[:, None] >> (n - 1 - np.arange(n))) & 1          # (m, n): column i = name i
        self.energies = energy(q, self.bits.astype(float))                        # x'Qx of every basis state
        _, _, constante = qubo_to_ising(q)
        diagonal = (self.energies - constante) / ising_scale(q)                   # H_C on the subspace
        self.cost_diagonal = torch.tensor(diagonal, dtype=torch.float64, device=self.device)
        self._pares = []                                                          # per ring edge, in gate order
        for a, b in ring_edges(n):
            bit_a, bit_b = 1 << (n - 1 - a), 1 << (n - 1 - b)
            ativos = np.flatnonzero(((self.masks & bit_a) != 0) != ((self.masks & bit_b) != 0))
            parceiros = np.searchsorted(self.masks, self.masks[ativos] ^ (bit_a | bit_b))
            self._pares.append((torch.tensor(ativos, device=self.device),
                                torch.tensor(parceiros, device=self.device)))

    def state(self, angles: torch.Tensor) -> torch.Tensor:
        """Amplitudes (C(n, k),) complex128: Dicke start, then per layer exp(-i gamma H_C) and IsingXY(2 beta) on
        the ring edges in order, whose (|01>, |10>) block is cos(beta) I + i sin(beta) X."""
        angles = angles.to(self.device)
        m = len(self.masks)
        psi = torch.full((m,), 1.0 / math.sqrt(m), dtype=torch.complex128, device=self.device)
        for gamma, beta in zip(angles[0], angles[1]):
            psi = psi * torch.exp(-1j * gamma * self.cost_diagonal)
            cosseno, seno = torch.cos(beta), torch.sin(beta)
            for ativos, parceiros in self._pares:
                psi = psi.index_copy(0, ativos, cosseno * psi[ativos] + 1j * seno * psi[parceiros])
        return psi

    def probabilities(self, angles: torch.Tensor) -> torch.Tensor:
        psi = self.state(angles)
        return psi.real ** 2 + psi.imag ** 2          # not abs(): its gradient is NaN at an exact zero amplitude

    def expval(self, angles: torch.Tensor) -> torch.Tensor:
        """<H_C> (scaled), differentiable in ``angles``."""
        return (self.probabilities(angles) * self.cost_diagonal).sum()


def solve_qaoa(q: np.ndarray, k: int, p: int, steps: int, lr: float, shots: int, seed: int,
               init: np.ndarray | None = None, device: str = "cpu") -> QaoaResult:
    """Adam on <H_C> from ``init`` (default: a linear ramp), then ``shots`` samples of the trained state scored by
    x'Qx; the best sampled subset wins (ties: the lowest basis index). The sampler's generator is seeded by
    ``seed``; nothing global is touched."""
    inicio = time.perf_counter()
    simulador = SubspaceQaoa(q, k, device)
    angulos = torch.tensor(linear_ramp(p) if init is None else np.asarray(init, dtype=float),
                           dtype=torch.float64, device=simulador.device, requires_grad=True)
    otimizador = torch.optim.Adam([angulos], lr=lr)
    with torch.no_grad():
        expval_inicial = float(simulador.expval(angulos))
    for _ in range(steps):
        otimizador.zero_grad(set_to_none=True)
        simulador.expval(angulos).backward()
        otimizador.step()
    finais = angulos.detach()
    with torch.no_grad():
        probabilidades = simulador.probabilities(finais)
        expval_final = float((probabilidades * simulador.cost_diagonal).sum())
    gerador = torch.Generator(device=simulador.device).manual_seed(seed)
    amostras = torch.multinomial(probabilidades, shots, replacement=True, generator=gerador).cpu().numpy()
    unicos, contagens = np.unique(amostras, return_counts=True)
    posicao = int(np.argmin(simulador.energies[unicos]))
    melhor = unicos[posicao]
    return QaoaResult(tuple(int(i) for i in np.flatnonzero(simulador.bits[melhor])),
                      float(simulador.energies[melhor]), expval_inicial, expval_final,
                      float(contagens[posicao] / shots), int((simulador.bits[amostras].sum(axis=1) != k).sum()),
                      finais.cpu().tolist(), time.perf_counter() - inicio, f"torch-subspace:{simulador.device.type}")
