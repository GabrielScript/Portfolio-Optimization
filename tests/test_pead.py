"""SUE, signal 1 of docs/superpowers/specs/2026-11-08-pead-recompra-design.md (section 3)."""

import numpy as np
import pandas as pd
import pytest


def _lucros(emissor: str, valores: list[float], primeiro_fim: str = "2018-03-31", atraso_dias: int = 40) -> list[dict]:
    """12-month net income at consecutive quarter ends, each received ``atraso_dias`` after its end."""
    fins = pd.date_range(primeiro_fim, periods=len(valores), freq="QE")
    return [{"issuer": emissor, "account": "net_income", "dt_refer": fim, "value": valor,
             "available_from": fim + pd.Timedelta(days=atraso_dias)} for fim, valor in zip(fins, valores)]


def test_sue_is_the_last_quarters_change_over_the_deviation_of_the_eight_before_known_at_the_month():
    from core.pead import sue
    valores = [10.0, 11.0, 13.0, 12.0, 15.0, 14.0, 18.0, 17.0, 20.0, 26.0, 25.0]      # 2018-03 .. 2020-09
    fundamentos = pd.DataFrame(_lucros("AAAA", valores))
    variacoes = np.diff(valores)                                                     # dE at 2018-06 .. 2020-09
    esperado = variacoes[-1] / np.std(variacoes[-9:-1], ddof=1)
    assert sue(fundamentos, pd.Timestamp("2020-12-01")).to_dict() == {"AAAA": pytest.approx(esperado)}
    # 2020-09 is received on 2020-11-09: before it, the last dE is 2020-06's, over the 8 changes 2018-06 .. 2020-03
    antes = variacoes[-2] / np.std(variacoes[-10:-2], ddof=1)
    assert sue(fundamentos, pd.Timestamp("2020-11-01")).to_dict() == {"AAAA": pytest.approx(antes)}


def test_sue_needs_six_earlier_changes_consecutive_quarters_and_a_positive_deviation():
    from core.pead import sue
    curto = _lucros("BBBB", [1.0, 2.0, 3.0, 5.0, 4.0, 6.0, 7.0])                    # 6 changes: 5 before the last
    plano = _lucros("CCCC", [1.0 + 2.0 * i for i in range(9)] + [30.0])              # equal changes: deviation 0
    com_buraco = _lucros("DDDD", [1.0, 2.0, 4.0, 3.0, 6.0, 5.0, 8.0, 9.0])            # 2018-03 .. 2019-12
    com_buraco += _lucros("DDDD", [12.0], primeiro_fim="2020-06-30")                 # 2020-03 missing: no dE at 2020-06
    fundamentos = pd.DataFrame(curto + plano + com_buraco)
    resultado = sue(fundamentos, pd.Timestamp("2021-06-01"))
    assert "BBBB" not in resultado.index and "CCCC" not in resultado.index
    # the latest quarter end with a dE is 2019-12 (dE 1), over the 6 changes before it
    assert resultado["DDDD"] == pytest.approx(1.0 / np.std([1.0, 2.0, -1.0, 3.0, -1.0, 3.0], ddof=1))
