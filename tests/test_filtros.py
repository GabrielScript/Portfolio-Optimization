"""Category filters above the screen's tables (the user's UI list of 2026-11-03)."""

import pandas as pd

from ui.filtros import aplicar_filtros, colunas_de_categoria

TABELA = pd.DataFrame({"Ticker": ["PETR4", "VALE3", "TSMC34", "XINA11"], "Setor": ["Petróleo", "Petróleo", "BDR", "ETF"],
                       "Tipo": ["Ação", "Ação", "BDR", "ETF"], "Preço": [30.0, 60.0, 50.0, 10.0]})


def test_the_category_columns_are_the_text_ones_with_few_distinct_values():
    # Ticker has one value per row: a list of every ticker is no filter; numbers are not categories
    assert colunas_de_categoria(TABELA) == ["Setor", "Tipo"]


def test_each_choice_keeps_the_rows_with_one_of_the_values_and_an_empty_choice_keeps_all():
    assert aplicar_filtros(TABELA, {"Tipo": ["Ação"], "Setor": []})["Ticker"].tolist() == ["PETR4", "VALE3"]
    assert aplicar_filtros(TABELA, {"Tipo": ["BDR", "ETF"], "Setor": ["ETF"]})["Ticker"].tolist() == ["XINA11"]
    assert len(aplicar_filtros(TABELA, {})) == 4
