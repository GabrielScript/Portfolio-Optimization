"""The technology kernel K (cosine of WIPO field counts on the patenting block, identity elsewhere) and its mix with
the returns' correlation are correlation matrices."""

import numpy as np
import pandas as pd

from core.inovemetria.kernel import mixed_correlation, technology_kernel


def test_kernel_is_a_correlation_matrix_on_random_counts():
    rng = np.random.default_rng(11)
    for _ in range(200):
        n = int(rng.integers(2, 12))
        counts = rng.poisson(0.7, size=(n, 35)) * (rng.random((n, 1)) < 0.6)     # some rows without patents
        k = technology_kernel(counts)
        assert np.allclose(np.diag(k), 1.0) and np.allclose(k, k.T)
        assert np.linalg.eigvalsh(k).min() >= -1e-10


def test_no_patents_gives_the_identity():
    assert np.array_equal(technology_kernel(np.zeros((4, 35))), np.eye(4))


def test_the_mix_is_a_correlation_matrix_and_lowers_the_determinant_of_two_same_field_names():
    rng = np.random.default_rng(5)
    returns = pd.DataFrame(rng.normal(0, 0.02, (300, 3)), columns=["AAAA3", "BBBB3", "CCCC3"])
    fields = pd.DataFrame(0, index=["AAAA", "BBBB", "CCCC"], columns=range(1, 36))
    fields.loc["AAAA", 4] = fields.loc["BBBB", 4] = 3          # AAAA and BBBB patent in the same field
    fields.loc["CCCC", 20] = 2
    same = mixed_correlation(lambda month: fields, month_of=lambda treino: pd.Timestamp("2020-01-01"))(returns)
    half = mixed_correlation(lambda month: fields * 0, month_of=lambda treino: pd.Timestamp("2020-01-01"))(returns)
    assert np.allclose(np.diag(same), 1.0) and np.linalg.eigvalsh(same).min() >= -1e-10
    assert same[0, 1] > half[0, 1] + 0.4 and abs(same[0, 2] - half[0, 2]) < 1e-12
