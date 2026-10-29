"""Stage B of the CDA best-ideas pre-registration: the score that orders the Conservadora's portfolio rule."""

import pandas as pd
import pytest

from scripts.cda_best_ideas_avaliar import DESEMPATE, nota_com_desempate

T = pd.Timestamp


def _painel(fundos: int) -> pd.DataFrame:
    return pd.DataFrame({"month": T("2015-01-01"), "issuer": ["PETR", "VALE", "ITUB"], "score": [0.2, 0.2, 0.0],
                         "consensus": 0.0, "qualifying": fundos})


def _conservadora(mes, candidatos):
    return pd.Series({"PETR4": 1.2, "VALE3": 1.6, "ITUB4": 1.9, "WEGE3": 0.4}).reindex(list(candidatos)).dropna()


def test_the_tie_break_is_tiny():
    assert DESEMPATE == pytest.approx(1e-6)


def test_best_ideas_lead_and_the_conservadora_breaks_ties_and_ranks_the_unnamed():
    nota = nota_com_desempate(_painel(40), _conservadora)(T("2015-01-01"), ["PETR4", "VALE3", "ITUB4", "WEGE3"])
    assert list(nota.sort_values(ascending=False).index) == ["VALE3", "PETR4", "ITUB4", "WEGE3"]


def test_a_month_with_fewer_than_30_funds_or_no_panel_is_the_conservadora_order():
    for painel, mes in ((_painel(29), T("2015-01-01")), (_painel(40), T("2010-03-01"))):
        nota = nota_com_desempate(painel, _conservadora)(mes, ["PETR4", "VALE3", "ITUB4", "WEGE3"])
        assert list(nota.sort_values(ascending=False).index) == ["ITUB4", "VALE3", "PETR4", "WEGE3"]


def test_names_without_a_conservadora_score_are_left_out():
    nota = nota_com_desempate(_painel(40), _conservadora)(T("2015-01-01"), ["PETR4", "BBAS3"])
    assert list(nota.index) == ["PETR4"]
