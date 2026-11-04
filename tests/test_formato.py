"""Empty cells read "—", not "None" (the user, 2026-11-04): the bitcoin ETFs before they listed. Streamlit 1.66's
dataframe writes "None" for any null, even through a pandas Styler's na_rep (seen on the screen), so only the columns
with gaps become text; the complete ones stay numbers and sort by value."""

import numpy as np
import pandas as pd

from ui.formato import TRACO, com_traco


def test_a_column_with_gaps_becomes_text_with_a_dash_and_the_brazilian_comma():
    tabela = pd.DataFrame({"Ano": ["2010", "2025"], "Composto": [0.2321, -0.0512], "BITH11": [np.nan, 0.4]})
    saida, config = com_traco(tabela, ["Composto", "BITH11"], porcento=True)
    assert saida["BITH11"].tolist() == [TRACO, "40,00%"]
    assert saida["Composto"].dtype == float                     # complete: still a number, sorted by value
    assert config["Composto"]["type_config"]["format"] == "percent"
    assert config["BITH11"]["alignment"] == "right"              # text, aligned like the numbers beside it


def test_plain_numbers_keep_two_decimals():
    saida, config = com_traco(pd.DataFrame({"Sharpe": [1.234, None], "Beta": [0.5, 0.6]}), ["Sharpe", "Beta"],
                              porcento=False)
    assert saida["Sharpe"].tolist() == ["1,23", TRACO]
    assert config["Beta"]["type_config"]["format"] == "%.2f"
