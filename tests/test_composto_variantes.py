"""The neutral-fill composite score: equal to the house score with every component present, shrunk when one is missing."""

import pandas as pd
import pytest

from core.composto_variantes import neutral_composite_score
from core.qubo_universo import composite_score
from core.views import escore_normal

NOMES = ["AAAA3", "BBBB3", "CCCC3", "DDDD3", "EEEE3", "FFFF3"]
BASE = pd.Series([0.9, 0.1, 0.5, 0.7, 0.3, 0.2], index=NOMES)


def _base(mes, candidatos):
    return BASE.reindex(list(candidatos))


def _value_complete(mes, candidatos):
    return pd.Series([1.0, 6.0, 2.0, 5.0, 3.0, 4.0], index=NOMES).reindex(list(candidatos))


def _value_without_two(mes, candidatos):
    return _value_complete(mes, candidatos).drop(["AAAA3", "BBBB3"])


def test_equals_the_house_score_when_every_name_has_every_component():
    casa = composite_score(_base, _value_complete)(pd.Timestamp("2020-01-01"), NOMES)
    neutra = neutral_composite_score(_base, _value_complete)(pd.Timestamp("2020-01-01"), NOMES)
    pd.testing.assert_series_equal(neutra, casa)


def test_a_missing_component_counts_as_zero_instead_of_being_skipped():
    mes = pd.Timestamp("2020-01-01")
    casa = composite_score(_base, _value_without_two)(mes, NOMES)
    neutra = neutral_composite_score(_base, _value_without_two)(mes, NOMES)
    z = escore_normal(BASE)
    assert casa["AAAA3"] == pytest.approx(z["AAAA3"])                 # the house score: the one it has
    assert neutra["AAAA3"] == pytest.approx(z["AAAA3"] / 2.0)         # neutral: (z + 0) / 2
    assert neutra["CCCC3"] == pytest.approx(casa["CCCC3"])            # complete names are unchanged
    assert abs(neutra["AAAA3"]) < abs(casa["AAAA3"])


def test_only_names_with_the_base_score_are_scored():
    notas = neutral_composite_score(lambda mes, c: BASE.drop("FFFF3").reindex([n for n in c if n != "FFFF3"]),
                                    _value_complete)(pd.Timestamp("2020-01-01"), NOMES)
    assert "FFFF3" not in notas.index and len(notas) == 5
