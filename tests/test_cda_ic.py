"""Stage A of the CDA best-ideas pre-registration: the score of the month and the months read."""

import pandas as pd
import pytest

from scripts.cda_ic import PERIODO, meses_lidos, nota_do_painel

T = pd.Timestamp


def _painel() -> pd.DataFrame:
    return pd.DataFrame({"month": [T("2015-01-01")] * 3 + [T("2015-02-01")] * 2,
                         "issuer": ["PETR", "VALE", "ITUB", "PETR", "VALE"],
                         "score": [0.2, 0.0, 0.1, 0.3, 0.05], "consensus": 0.0,
                         "qualifying": [40, 40, 40, 29, 29]})


def test_period_is_2011_01_to_2026_09():
    assert PERIODO == (T("2011-01-01"), T("2026-09-01"))


def test_months_with_fewer_than_30_qualifying_funds_are_dropped():
    assert meses_lidos(_painel()) == [T("2015-01-01")]


def test_each_security_takes_its_issuers_score_and_names_outside_the_panel_are_left_out():
    nota = nota_do_painel(_painel(), "score")(T("2015-01-01"), ["PETR3", "PETR4", "ITUB4", "WEGE3"])
    assert nota.to_dict() == {"PETR3": pytest.approx(0.2), "PETR4": pytest.approx(0.2), "ITUB4": pytest.approx(0.1)}
