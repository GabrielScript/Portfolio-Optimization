"""The staggered-thirds book of spec 2026-11-16-composto-tercos."""

import numpy as np
import pandas as pd
import pytest

from core.composto_tercos import livro_em_partes

DIAS = pd.bdate_range("2020-01-01", periods=6)


def test_each_sleeve_earns_cdi_until_it_starts_then_its_own_curve():
    g = pd.Series([1.00, 1.01, 1.02, 1.03, 1.04, 1.05], index=DIAS)
    s0 = pd.Series([1.0, 2.0, 2.0, 2.0, 2.0, 4.0], index=DIAS)
    s1 = pd.Series([5.0, 5.0, 10.0, 10.0], index=DIAS[2:])        # starts on day 2, its value scale is irrelevant
    livro = livro_em_partes([s0, s1], g)
    esperado_s1 = np.array([1.00, 1.01, 1.02, 1.02, 2.04, 2.04])   # CDI to day 2, then doubles
    esperado = (s0.to_numpy() + esperado_s1) / 2
    assert livro.index.equals(DIAS)
    np.testing.assert_allclose(livro.to_numpy(), esperado)


def test_sleeves_are_never_rebalanced_against_each_other():
    g = pd.Series(1.0, index=DIAS)
    s0 = pd.Series([1, 2, 4, 8, 16, 32], index=DIAS, dtype=float)
    s1 = pd.Series(1.0, index=DIAS)
    assert livro_em_partes([s0, s1], g).iloc[-1] == pytest.approx((32 + 1) / 2)


def test_a_sleeve_starting_before_the_book_is_refused():
    g = pd.Series(1.0, index=DIAS[1:])
    with pytest.raises(ValueError):
        livro_em_partes([pd.Series(1.0, index=DIAS)], g)
