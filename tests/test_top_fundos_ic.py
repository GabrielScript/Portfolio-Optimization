"""Stage A of the top-funds pre-registration: the three-part gate."""

import numpy as np
import pandas as pd

from scripts.top_fundos_ic import veredito

MESES = pd.date_range("2015-01-01", periods=4, freq="MS")


def _ics(valores: list[float]) -> pd.Series:
    return pd.Series(valores, index=MESES)


def test_all_three_parts_pass_when_the_top_funds_ic_is_high_and_beats_the_all_fund_ic_month_by_month():
    resultado = veredito(_ics([0.05, 0.04, 0.06, 0.05]), _ics([0.01, 0.02, 0.0, 0.01]))
    assert (resultado.s2, resultado.acima_da_52, resultado.top_acrescenta, resultado.aprovado) == (True,) * 4


def test_an_ic_that_passes_s2_but_stays_under_attempt_52s_0_032_fails():
    resultado = veredito(_ics([0.03, 0.025, 0.03, 0.035]), _ics([0.0, 0.0, 0.01, 0.0]))
    assert (resultado.s2, resultado.acima_da_52, resultado.aprovado) == (True, False, False)


def test_the_top_15_restriction_must_add_over_all_funds_in_the_paired_monthly_difference():
    resultado = veredito(_ics([0.05, 0.04, 0.06, 0.05]), _ics([0.05, 0.05, 0.05, 0.05]))    # difference 0, -.01, .01, 0
    assert (resultado.s2, resultado.acima_da_52, resultado.top_acrescenta, resultado.aprovado) == (True, True, False,
                                                                                                    False)


def test_the_paired_difference_uses_only_months_where_both_ics_exist():
    resultado = veredito(_ics([0.05, 0.04, 0.06, np.nan]), _ics([0.01, 0.02, 0.0, 0.9]))
    assert resultado.top_acrescenta
