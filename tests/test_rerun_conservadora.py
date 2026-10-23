"""Descriptive rerun of the Conservadora on the corrected lake (``scripts/rerun_conservadora.py``, DECISOES 106 and
115): same pre-registered script, same period, only the frozen inputs change."""

import pandas as pd

from scripts.avaliar_robustez import Dados


def test_the_rerun_cuts_the_prices_at_the_first_runs_last_session():
    from scripts.rerun_conservadora import cut_to_first_run
    sessions = pd.bdate_range("2026-10-14", "2026-10-21")
    dados = Dados(pd.DataFrame({"BOVA11": range(len(sessions))}, index=sessions, dtype=float),
                  pd.DataFrame({"month": [], "security": []}), {}, {"x": "y"}, {"data_through": "2026-10-21"})
    cut = cut_to_first_run(dados)
    assert cut.precos.index[-1] == pd.Timestamp("2026-10-16") and len(cut.precos) == 3
    assert cut.sha256 == {"x": "y"} and cut.manifest["data_through"] == "2026-10-21"


def test_the_rerun_adds_both_volatilities_to_every_row_of_the_phase_table():
    from core.volatilidade import volatilidades
    from scripts.rerun_conservadora import VOLATILITY_COLUMNS, with_volatility
    sessions = pd.bdate_range("2026-01-05", periods=6)
    curves = {"A": pd.Series([100.0, 101.0, 99.0, 100.0, 98.0, 99.0], index=sessions),
              "B": pd.Series([float("nan"), 50.0, 51.0, 50.5, 50.0, 52.0], index=sessions)}
    table = with_volatility(pd.DataFrame({"CAGR": [0.1, 0.2]}, index=["A", "B"]), curves)
    assert list(table.columns) == ["CAGR", *VOLATILITY_COLUMNS]
    for name, curve in curves.items():
        expected = volatilidades(curve)
        assert tuple(table.loc[name, list(VOLATILITY_COLUMNS)]) == (expected.anual, expected.de_baixa)
