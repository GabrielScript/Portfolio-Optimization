"""
qpo.py - Quantum Portfolio Optimization, classical side (docs/superpowers/specs/2026-10-27-qpo-design.md).

The QUBO picks k of the n best Conservadora names: E(x) = x'Qx = -alpha'x / k + (lambda / 2) x'Sigma x / k^2 with
|x| = k (the negative mean-variance utility of the equal-weight portfolio x / k). The exact solver here is the
ceiling a QAOA can at best reach on the same QUBO; the QAOA itself lives in ``kaggle/qpo_qaoa.py``.
"""

from __future__ import annotations

import itertools
from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
import pandas as pd

from core.risco import emissor

IC0 = 0.059          # the Conservadora's 1998-2007 IC (DECISOES 76): before every OOS date


@dataclass(frozen=True)
class ExactSolution:
    picks: tuple[int, ...]   # indices of the chosen names, ascending
    energy: float            # E at the optimum
    worst: float             # E at the worst feasible subset (for the approximation ratio)


def energy(q: np.ndarray, x: np.ndarray) -> float | np.ndarray:
    """x'Qx for one bitstring (n,) or a batch (m, n)."""
    valores = np.einsum("...i,ij,...j->...", x, q, x)
    return float(valores) if np.ndim(valores) == 0 else valores


def build_qubo(alpha: np.ndarray, cov: np.ndarray, k: int, risk_aversion: float) -> np.ndarray:
    """Q = (lambda / (2 k^2)) Sigma - diag(alpha) / k, so x'Qx = -alpha'w + (lambda / 2) w'Sigma w at w = x / k."""
    alpha, cov = np.asarray(alpha, dtype=float), np.asarray(cov, dtype=float)
    if cov.shape != (len(alpha), len(alpha)) or k < 1:
        raise ValueError(f"alpha {alpha.shape}, Sigma {cov.shape}, k = {k}: need an (n, n) Sigma and k >= 1")
    q = risk_aversion / (2.0 * k * k) * (cov + cov.T) / 2.0
    return q - np.diag(alpha) / k


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


def _subsets(m: int, r: int) -> tuple[np.ndarray, np.ndarray]:
    """All r-subsets of range(m): their indices (C(m, r), r) and their 0/1 rows (C(m, r), m), lexicographic."""
    combinacoes = list(itertools.combinations(range(m), r))
    indices = np.array(combinacoes, dtype=np.int64).reshape(len(combinacoes), r)   # r = 0: one empty subset
    linhas = np.zeros((len(indices), m))
    linhas[np.arange(len(indices))[:, None], indices] = 1.0
    return indices, linhas


def solve_exact(q: np.ndarray, k: int) -> ExactSolution:
    """
    Minimum and maximum of x'Qx over every |x| = k, exactly: the names split in halves A and B, and for each split
    j + (k - j) the C(n_A, j) x C(n_B, k - j) energies E_A + E_B + 2 x_A'Q_AB x_B come from one matrix product.
    The largest block at n = 28, k = 15 is 3,432 x 3,003 (about 80 MB in float64).
    """
    n = len(q)
    if not 1 <= k <= n:
        raise ValueError(f"k = {k} is outside 1..{n}")
    s = (q + q.T) / 2.0
    n_a = n // 2
    s_aa, s_bb, s_ab = s[:n_a, :n_a], s[n_a:, n_a:], s[:n_a, n_a:]
    melhor: tuple[float, tuple[int, ...]] | None = None
    pior = -np.inf
    for j in range(max(0, k - (n - n_a)), min(k, n_a) + 1):
        idx_a, x_a = _subsets(n_a, j)
        idx_b, x_b = _subsets(n - n_a, k - j)
        bloco = (energy(s_aa, x_a)[:, None] + energy(s_bb, x_b)[None, :] + 2.0 * (x_a @ s_ab) @ x_b.T)
        i, m = np.unravel_index(int(np.argmin(bloco)), bloco.shape)
        if melhor is None or bloco[i, m] < melhor[0]:
            melhor = (float(bloco[i, m]), tuple(int(v) for v in (*idx_a[i], *(idx_b[m] + n_a))))
        pior = max(pior, float(bloco.max()))
    assert melhor is not None
    return ExactSolution(melhor[1], melhor[0], pior)


def approximation_ratio(e: float, e_best: float, e_worst: float) -> float:
    """(E_worst - E) / (E_worst - E_best) over the feasible set: 1 at the optimum, 0 at the worst subset."""
    if e_worst == e_best:
        return 1.0
    return (e_worst - e) / (e_worst - e_best)


def candidates(score: pd.Series, sectors: Mapping[str, str], n: int) -> list[str]:
    """
    The QUBO's names: the Conservadora score in descending order, skipping a name without sector (the fixed rule
    SEM_SETOR) and the 2nd class of an issuer already in (the higher score stays), as ``carteira_da_nota``, but with
    no per-sector count: the 30% sector cap is enforced on the weights.
    """
    escolhidos: list[str] = []
    emissores: set[str] = set()
    for papel in score.dropna().sort_values(ascending=False, kind="stable").index:
        if not sectors.get(papel) or emissor(papel) in emissores:
            continue
        escolhidos.append(papel)
        emissores.add(emissor(papel))
        if len(escolhidos) == n:
            break
    return escolhidos


def qubo_alphas(score: pd.Series, cov: pd.DataFrame, ic: float = IC0) -> pd.Series:
    """Grinold-Kahn alpha = IC x vol x z, z = the score's cross-sectional z-score (ddof 0); all ties -> zeros."""
    desvio = score.std(ddof=0)
    z = (score - score.mean()) / desvio if desvio > 0 else score * 0.0
    vol = np.sqrt(np.diag(cov.loc[score.index, score.index].to_numpy()))
    return ic * vol * z
