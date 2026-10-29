"""Attempt 53's reading harness: what the rotation held, session by session."""

import pandas as pd

from scripts.h2_rotacao_avaliar import posse_diaria, trocas_por_ano

T = pd.Timestamp


def test_a_choice_is_held_from_the_session_after_its_decision():
    datas = pd.DatetimeIndex([T("2020-01-30"), T("2020-01-31"), T("2020-02-03"), T("2020-02-28"), T("2020-03-02")])
    escolhas = pd.Series({T("2020-01-31"): "BR", T("2020-02-28"): "CDI"})
    assert posse_diaria(escolhas, datas).tolist() == ["BR", "BR", "CDI"]


def test_switches_count_changes_of_choice_by_year_not_the_first_buy():
    escolhas = pd.Series({T("2019-12-31"): "BR", T("2020-01-31"): "BR", T("2020-02-28"): "US",
                          T("2020-03-31"): "CDI", T("2021-01-29"): "CDI"})
    assert trocas_por_ano(escolhas).to_dict() == {2019: 0, 2020: 2, 2021: 0}
