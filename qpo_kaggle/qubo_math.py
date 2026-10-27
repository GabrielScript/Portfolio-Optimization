"""
qubo_math.py - the numpy-only QUBO helpers shared by the classical side (core/qpo.py) and the QAOA solvers.

One source: the Kaggle kernel ships this package without ``core``, so the helpers live here and core re-exports
them.
"""

from __future__ import annotations

import numpy as np

__all__ = ["approximation_ratio", "energy", "qubo_to_ising"]


def energy(q: np.ndarray, x: np.ndarray) -> float | np.ndarray:
    """x'Qx for one bitstring (n,) or a batch (m, n)."""
    valores = np.einsum("...i,ij,...j->...", x, q, x)
    return float(valores) if np.ndim(valores) == 0 else valores


def qubo_to_ising(q: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """
    (h, J, const) with x'Qx = h'z + sum_{i<j} J_ij z_i z_j + const for x = (1 - z) / 2, so z = +1 is x = 0 (|0>).
    J is symmetric with a zero diagonal; each pair counts once (the upper triangle).
    """
    s = (q + q.T) / 2.0
    diagonal = np.diag(s)
    fora = s - np.diag(diagonal)
    h = -diagonal / 2.0 - fora.sum(axis=1) / 2.0
    return h, fora / 2.0, float(diagonal.sum() / 2.0 + fora.sum() / 4.0)


def approximation_ratio(e: float, e_best: float, e_worst: float) -> float:
    """(E_worst - E) / (E_worst - E_best) over the feasible set: 1 at the optimum, 0 at the worst subset."""
    if e_worst == e_best:
        return 1.0
    return (e_worst - e) / (e_worst - e_best)
