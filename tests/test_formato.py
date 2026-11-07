"""Every table in fixed Brazilian decimals (the whole-page finish review of 2026-11-06: "0.38", "R$ 0.82", "10%" beside
"1,1%", "None" cells). Empty cells read "—", not "None" (the user, 2026-11-04). Streamlit 1.66's dataframe writes "None"
for any null, even through a pandas Styler's na_rep (seen on the screen), so only the columns with gaps become text; the
complete ones stay numbers under a Styler, so they sort by value."""

import numpy as np
import pandas as pd

from ui import formato
from ui.formato import TRACO


def test_a_column_with_gaps_becomes_text_with_a_dash_and_the_brazilian_comma():
    tabela = pd.DataFrame({"Ano": ["2010", "2025"], "Composto": [0.2321, -0.0512], "BITH11": [np.nan, 0.4]})
    estilo, config = formato.br(tabela, dict.fromkeys(["Composto", "BITH11"], formato.percentual))
    assert estilo.data["BITH11"].tolist() == [TRACO, "40,00%"]
    assert estilo.data["Composto"].dtype == float                     # complete: still a number, sorted by value
    html = estilo.to_html()
    assert "23,21%" in html and "−5,12%" in html and "None" not in html
    assert config["BITH11"]["alignment"] == "right"                   # text, aligned like the numbers beside it


def test_every_kind_keeps_fixed_decimals_and_a_thousands_dot():
    tabela = pd.DataFrame({"DSR": [0.38, 1.0], "Preço (R$)": [0.82, 1234.5], "P/L": [7.25, 12.0], "Tom": [-0.14, 0.45]})
    estilo, _ = formato.br(tabela, {"DSR": formato.decimal, "Preço (R$)": formato.decimal, "P/L": formato.uma_casa,
                                    "Tom": formato.com_sinal})
    html = estilo.to_html()
    for texto in ("0,38", "1,00", "0,82", "1.234,50", "7,2", "12,0", "−0,14", "+0,45"):
        assert texto in html, texto
    assert "0.38" not in html and "0.82" not in html


def test_the_help_rides_on_each_formatted_column_without_a_format_that_would_override_the_styler():
    tabela = pd.DataFrame({"Sharpe": [1.234, 0.5], "Beta": [0.5, None]})
    _, config = formato.br(tabela, dict.fromkeys(["Sharpe", "Beta"], formato.decimal), ajuda={"Sharpe": "Retorno/risco"})
    assert config["Sharpe"]["help"] == "Retorno/risco"
    assert config["Sharpe"]["type_config"].get("format") is None
    assert config["Beta"]["alignment"] == "right"


def test_a_missing_number_outside_a_table_is_a_dash():
    assert formato.percentual(None) == TRACO and formato.decimal(float("nan")) == TRACO
    assert formato.percentual(-0.1234, casas=1) == "−12,3%"


def test_the_year_column_is_wide_enough_for_a_partial_years_label():
    # "2010 (desde 09/0" was cut in both per-year tables
    assert formato.COLUNA_DO_ANO["width"] == "medium"
