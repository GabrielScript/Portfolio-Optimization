"""The "Filtros" button above each of the screen's tables (the user's UI list of 2026-11-03)."""

import pandas as pd

from ui.filtros import aplicar_filtros, colunas_de_escolha, filtrar

TABELA = pd.DataFrame({"Ticker": ["PETR4", "VALE3", "TSMC34", "XINA11"], "Setor": ["Petróleo", "Petróleo", "BDR", "ETF"],
                       "Tipo": ["Ação", "Ação", "BDR", "ETF"], "Preço": [30.0, 60.0, 50.0, 10.0],
                       "Ano": [2024, 2025, 2025, 2026],
                       "Manchete": [f"Uma manchete longa demais para virar uma lista de escolha, número {n}"
                                    for n in range(4)]})


def test_every_short_text_or_whole_number_column_with_two_or_more_values_gets_a_choice_list():
    # the user asked for one choice list per column: tickers and years too, but not prices (decimals) nor long text
    assert colunas_de_escolha(TABELA) == ["Ticker", "Setor", "Tipo", "Ano"]


def test_a_column_with_a_single_value_or_too_many_gets_no_list():
    tabela = pd.DataFrame({"Ordem": ["Comprar"] * 3, "Ticker": ["A", "B", "C"]})
    assert colunas_de_escolha(tabela) == ["Ticker"]
    assert colunas_de_escolha(pd.DataFrame({"Ticker": [f"T{n}" for n in range(61)]})) == []


def test_each_choice_keeps_the_rows_with_one_of_the_values_and_an_empty_choice_keeps_all():
    assert aplicar_filtros(TABELA, {"Tipo": ["Ação"], "Setor": []})["Ticker"].tolist() == ["PETR4", "VALE3"]
    assert aplicar_filtros(TABELA, {"Tipo": ["BDR", "ETF"], "Setor": ["ETF"]})["Ticker"].tolist() == ["XINA11"]
    assert aplicar_filtros(TABELA, {"Ano": ["2025"]})["Ticker"].tolist() == ["VALE3", "TSMC34"]
    assert len(aplicar_filtros(TABELA, {})) == 4


def test_a_table_indexed_by_its_rows_names_keeps_the_index_and_can_filter_on_it():
    retornos = pd.DataFrame({"Dia": [0.01, -0.02]}, index=pd.Index(["Composite", "BOVA11"], name="Carteira"))
    assert colunas_de_escolha(retornos.reset_index()) == ["Carteira"]
    saida = filtrar(retornos, "retornos")
    assert saida.index.name == "Carteira" and saida.index.tolist() == ["Composite", "BOVA11"]
