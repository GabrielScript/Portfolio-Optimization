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
