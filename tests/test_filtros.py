"""The "Filtros" button above each of the screen's tables (the user's UI list of 2026-11-03)."""

import pandas as pd

from ui.filtros import aplicar_filtros, colunas_de_escolha, com_setor_para_filtro, filtrar

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
    assert colunas_de_escolha(pd.DataFrame({"Código": [f"T{n}" for n in range(61)]})) == []


def test_a_table_of_stocks_always_gets_the_stock_and_the_sector_lists_however_many_values():
    # the user, 2026-11-06: whenever a table holds stocks, a filter by the stock itself and by sector, always
    muitas = pd.DataFrame({"Ticker": [f"T{n}" for n in range(1_545)], "Setor": ["Bancos"] * 1_545})
    assert colunas_de_escolha(muitas) == ["Ticker", "Setor"]
    for coluna in ("Ação", "Ativo"):
        assert colunas_de_escolha(pd.DataFrame({coluna: ["PETR4"], "Hora": ["10:20"]})) == [coluna]


def test_a_table_of_stocks_without_a_sector_column_gets_a_hidden_sector_to_filter_on():
    alertas = pd.DataFrame({"Ação": ["PETR4", "ITUB4", "XPTO9"], "Nota": [0.1, 0.2, 0.3]})
    tabela, oculta = com_setor_para_filtro(alertas, {"PETR4.SA": "Petróleo", "ITUB4.SA": "Bancos"})
    assert oculta == "Setor" and tabela["Setor"].tolist()[:2] == ["Petróleo", "Bancos"]
    assert pd.isna(tabela["Setor"].iloc[2]) and list(alertas.columns) == ["Ação", "Nota"]   # input untouched
    assert com_setor_para_filtro(TABELA, {"PETR4.SA": "X"})[1] is None        # it has its own sector column
    assert com_setor_para_filtro(pd.DataFrame({"Carteira": ["A"]}), {"A.SA": "X"})[1] is None   # no stocks


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
