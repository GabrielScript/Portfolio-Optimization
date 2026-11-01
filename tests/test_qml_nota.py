"""A forecast as the score of the Composite's portfolio rule (docs/superpowers/specs/2026-11-01-qpo-hibrido-design.md, section 4)."""

import numpy as np
import pandas as pd
import pytest


def _tabela() -> pd.DataFrame:
    return pd.DataFrame({"month": pd.to_datetime(["2020-01-01", "2020-01-01", "2020-02-01"]),
                         "security": ["A", "B", "A"], "previsao": [0.3, -0.1, 0.9]})


def test_the_score_of_a_month_is_that_months_forecast_over_the_candidates():
    from core.qml_nota import nota_da_previsao

    nota = nota_da_previsao(_tabela())
    assert nota(pd.Timestamp("2020-01-01"), ["B", "A"]).to_dict() == {"B": -0.1, "A": 0.3}
    assert nota(pd.Timestamp("2020-02-01"), ["A"]).to_dict() == {"A": 0.9}


def test_a_candidate_without_a_forecast_has_no_score():
    from core.qml_nota import nota_da_previsao

    assert np.isnan(nota_da_previsao(_tabela())(pd.Timestamp("2020-02-01"), ["A", "B"])["B"])


def test_a_month_without_any_forecast_stops_the_run():
    from core.qml_nota import nota_da_previsao

    with pytest.raises(KeyError, match="2020-03"):
        nota_da_previsao(_tabela())(pd.Timestamp("2020-03-01"), ["A"])
