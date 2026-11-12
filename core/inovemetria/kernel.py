"""
kernel.py - The technology kernel of the Fermiônica arms 6-8 (spec "Stage 2"): K_ij the cosine of two issuers' counts
of applications per WIPO field when both patent, 0 otherwise, K_ii = 1. K is the Gram matrix of unit vectors on
the patenting block and the identity elsewhere, so K and any convex mix with a correlation matrix are correlation
matrices. With the counts zeroed, the mix is (C + I)/2: the control arm.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd

from core.diversidade import shrunk_correlation
from core.fermionica import Correlation
from core.risco import emissor


def technology_kernel(counts: np.ndarray) -> np.ndarray:
    """n x fields counts -> n x n kernel."""
    counts = np.asarray(counts, dtype=float)
    norms = np.linalg.norm(counts, axis=1)
    patenting = norms > 0
    unit = np.zeros_like(counts)
    unit[patenting] = counts[patenting] / norms[patenting, None]
    kernel = unit @ unit.T
    np.fill_diagonal(kernel, 1.0)
    return kernel


def decision_month(treino: pd.DataFrame) -> pd.Timestamp:
    """The month of a decision, as the house strategies read it: the training window's last session's month."""
    return treino.index[-1].to_period("M").to_timestamp()


def mixed_correlation(fields_of_month: Callable[[pd.Timestamp], pd.DataFrame], weight: float = 0.5,
                      correlation: Correlation = shrunk_correlation,
                      month_of: Callable[[pd.DataFrame], pd.Timestamp] = decision_month) -> Correlation:
    """treino -> (1 - weight) C + weight K on treino's columns; ``fields_of_month`` gives issuer x field counts at
    the decision month (an issuer it lacks has no patents)."""
    def _corr(treino: pd.DataFrame) -> np.ndarray:
        counts = fields_of_month(month_of(treino))
        rows = counts.reindex([emissor(t) for t in treino.columns]).fillna(0.0).to_numpy(dtype=float)
        return (1.0 - weight) * correlation(treino) + weight * technology_kernel(rows)
    return _corr
