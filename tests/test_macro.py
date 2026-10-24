"""The screen's macro panel (``core.macro``): the BCB's Focus survey, live, as information."""

import pandas as pd
import pytest

from core.macro import resumo_focus


def _focus(medianas: dict[tuple[str, int], float]) -> pd.DataFrame:
    linhas = [{"data": pd.Timestamp(data), "ano_referencia": ano, "mediana": valor, "desvio": 0.1}
              for (data, ano), valor in medianas.items()]
    return pd.DataFrame(linhas)


def test_the_focus_summary_has_this_and_next_years_median_and_the_change_in_four_weeks():
    selic = _focus({("2026-09-18", 2026): 15.0, ("2026-09-18", 2027): 12.5,
                    ("2026-10-16", 2026): 14.75, ("2026-10-16", 2027): 12.25})
    resumo = resumo_focus({"Selic": selic}, ano=2026)
    linha = resumo.loc["Selic"]
    assert (linha["2026"], linha["2027"]) == (14.75, 12.25)
    assert linha["Mudança em 4 semanas (2026)"] == pytest.approx(-0.25)
    assert linha["Pesquisa"] == pd.Timestamp("2026-10-16")
