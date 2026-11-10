from __future__ import annotations

import numpy as np
import pytest

from drl import periodo


def test_two_years_of_ten_percent_compound_to_twenty_one():
    """The bug this module exists for: concatenating 1.0->1.1 with 1.0->1.1 gives 1.1, not 1.21."""
    ano = np.array([1.0, 1.1])
    assert periodo.emendar([ano, ano])[-1] == pytest.approx(1.21)

def test_the_join_between_years_has_no_fake_break():
    primeiro = np.array([1.0, 0.9, 1.2])
    segundo = np.array([1.0, 1.1])
    curva = periodo.emendar([primeiro, segundo])
    assert curva[2] == pytest.approx(1.2)        # the join repeats the level, it does not reset to 1.0
    assert curva[3] / curva[2] - 1.0 == pytest.approx(0.1)
    assert curva[-1] == pytest.approx(1.2 * 1.1)
    assert len(curva) == 4                       # the repeated boundary point is dropped


def test_an_empty_list_returns_an_empty_curve():
    assert periodo.emendar([]).size == 0


def test_a_single_year_is_returned_unchanged():
    ano = np.array([1.0, 1.05, 1.02])
    assert np.allclose(periodo.emendar([ano]), ano)


def test_rows_group_by_model_and_sort_by_year():
    linhas = [{"algoritmo": "ppo", "recompensa": "dsr", "ano_teste": 2019},
              {"algoritmo": "ppo", "recompensa": "dsr", "ano_teste": 2010},
              {"algoritmo": "a2c", "recompensa": "valor", "ano_teste": 2010}]
    grupos = periodo.por_ano(linhas)
    assert set(grupos) == {"ppo/dsr", "a2c/valor"}
    assert [r["ano_teste"] for r in grupos["ppo/dsr"]] == [2010, 2019]
