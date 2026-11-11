"""
diversidade.py - Diversity of a set of names from its return correlation matrix
(docs/superpowers/specs/2026-11-11-fermionic-v2-design.md). NumPy, SciPy and scikit-learn only.

For a correlation matrix C_S of k names with eigenvalues l_1..l_k (their sum is k), both functionals are <= 0 and
are 0 only when C_S = I (no correlation):

    log_det      sum log l_i                 -inf at a duplicate (fermions: similar names are forbidden)
    von_neumann  -sum l_i log l_i            = k (S(rho) - log k), S the von Neumann entropy of rho = C_S / k;
                                             -k log k at one common factor; e^S = effective number of bets

Each takes one matrix or a stack (..., k, k) and returns a float or an array.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd
from scipy.special import xlogy
from sklearn.covariance import LedoitWolf

Diversity = Callable[[np.ndarray], np.ndarray]      # correlation matrices (..., k, k) -> diversity of each


def log_det(matrizes: np.ndarray) -> np.ndarray:
    """log det C per matrix; -inf when C is singular or not positive definite."""
    sinal, valor = np.linalg.slogdet(matrizes)
    return np.where(sinal > 0, valor, -np.inf)[()]


def von_neumann(matrizes: np.ndarray) -> np.ndarray:
    """-sum l log l over the eigenvalues of each C (0 log 0 = 0; rounding below zero is clipped)."""
    autovalores = np.clip(np.linalg.eigvalsh(matrizes), 0.0, None)
    return -xlogy(autovalores, autovalores).sum(axis=-1)[()]


def effective_bets(matriz: np.ndarray) -> float:
    """e^S of the density matrix C / k: k for uncorrelated names, 1 for one common factor."""
    k = len(matriz)
    return float(k * np.exp(von_neumann(matriz) / k))


def shrunk_correlation(retornos_diarios: pd.DataFrame) -> np.ndarray:
    """Ledoit-Wolf on the standardized returns: (1 - delta) R + delta I, the same shrinkage for every name
    (shrinking the covariance instead deflates the correlations of low-volatility names more)."""
    valores = retornos_diarios.to_numpy(dtype=float)
    desvios = valores.std(axis=0)
    if not (desvios > 0).all():
        raise ValueError("a name with constant returns has no correlation")
    encolhida = LedoitWolf().fit((valores - valores.mean(axis=0)) / desvios).covariance_
    escala = np.sqrt(np.diag(encolhida))
    return encolhida / np.outer(escala, escala)
