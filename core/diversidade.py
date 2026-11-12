"""
diversidade.py - Diversity of a set of names from its return correlation matrix
(docs/superpowers/specs/2026-11-11-fermionic-v2-design.md). NumPy, SciPy and scikit-learn only.

For a correlation matrix C_S of k names with eigenvalues l_1..l_k (their sum is k), both functionals are <= 0 and
are 0 only when C_S = I (no correlation):

    log_det      sum log l_i                 -inf at a duplicate (fermions: similar names are forbidden)
    von_neumann  -sum l_i log l_i            = k (S(rho) - log k), S the von Neumann entropy of rho = C_S / k;
                                             -k log k at one common factor; e^S = effective number of bets

Each takes one matrix or a stack (..., k, k) and returns a float or an array.

With weights w on the simplex (docs/superpowers/specs/2026-11-11-fermionic-size-grid-design.md): unit vectors |c_i>
with <c_i|c_j> = C_ij give the state rho(w) = sum w_i |c_i><c_i|, whose spectrum is that of
diag(sqrt w) C diag(sqrt w). ``state_entropy`` is its von Neumann entropy S (<= the Shannon entropy of w, equal
only without correlation; concave in w) and ``eic_weights`` the weights of largest S: at them every name has
-<c_i| log rho |c_i> = S (equal information contribution).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import xlogy
from sklearn.covariance import LedoitWolf

Diversity = Callable[[np.ndarray], np.ndarray]      # correlation matrices (..., k, k) -> diversity of each
GroupCap = tuple[Sequence[int], float]              # (positions of a group's names, the most weight they may sum)
LOG_FLOOR = 1e-300                                  # eigenvalues below it count as it inside a logarithm


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


def state_entropy(corr: np.ndarray, pesos: np.ndarray) -> float:
    """S(rho(w)) in nats: the von Neumann entropy of the names of correlation ``corr`` held at ``pesos`` (sum 1)."""
    raiz = np.sqrt(np.asarray(pesos, dtype=float))
    autovalores = np.clip(np.linalg.eigvalsh(raiz[:, None] * corr * raiz[None, :]), 0.0, None)
    return float(-xlogy(autovalores, autovalores).sum())


def _negative_entropy_and_gradient(pesos: np.ndarray, raiz: np.ndarray) -> tuple[float, np.ndarray]:
    """-S(rho) and its gradient <c_i| log rho + 1 |c_i>, with rho = A diag(w) A and A = ``raiz`` = C^(1/2)."""
    autovalores, base = np.linalg.eigh((raiz * pesos) @ raiz)
    autovalores = np.clip(autovalores, 0.0, None)
    projecoes = raiz @ base                                       # row i: |c_i> in rho's eigenbasis
    gradiente = (projecoes**2) @ (np.log(np.maximum(autovalores, LOG_FLOOR)) + 1.0)
    return float(xlogy(autovalores, autovalores).sum()), gradiente


def eic_weights(corr: np.ndarray, teto: float, grupos: Sequence[GroupCap] = ()) -> np.ndarray:
    """The weights of largest ``state_entropy`` with sum 1, 0 <= w_i <= ``teto`` and each group's sum within its
    cap: a concave program, solved by SLSQP from equal weights. ValueError when the caps cannot sum to 1 or the
    solver does not converge."""
    k = len(corr)
    if k * teto < 1.0:
        raise ValueError(f"{k} names at most {teto:.0%} each cannot sum to 1")
    valores, vetores = np.linalg.eigh(corr)
    raiz = (vetores * np.sqrt(np.clip(valores, 0.0, None))) @ vetores.T
    restricoes = [{"type": "eq", "fun": lambda w: w.sum() - 1.0, "jac": lambda w: np.ones(k)}]
    for posicoes, limite in grupos:
        indicador = np.zeros(k)
        indicador[list(posicoes)] = 1.0
        restricoes.append({"type": "ineq", "fun": lambda w, a=indicador, b=limite: b - a @ w,
                           "jac": lambda w, a=indicador: -a})
    solucao = minimize(_negative_entropy_and_gradient, np.full(k, 1.0 / k), args=(raiz,), jac=True, method="SLSQP",
                       bounds=[(0.0, teto)] * k, constraints=restricoes, options={"ftol": 1e-14, "maxiter": 1000})
    if not solucao.success:
        raise ValueError(f"EIC weights did not converge: {solucao.message}")
    pesos = np.clip(solucao.x, 0.0, teto)
    return pesos / pesos.sum()
