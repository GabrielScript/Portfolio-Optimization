"""The market panel's full list (``ativos_tela.py``): every stock, BDR and ETF the lake exports, named and typed."""

import pandas as pd

from ativos_tela import acoes_da_tela, info_curada, info_da_tela, ler_nomes


def _nomes() -> pd.DataFrame:
    return pd.DataFrame({"security": ["PETR4", "AAPL34", "TSMC34", "BOVA11", "CSUD3", "JBSS32"],
                         "name": ["PETROBRAS", "APPLE", "TAIWANSMFAC", "ISHARES BOVA", "CSU DIGITAL", "JBS N.V."],
                         "asset_type": ["stock", "bdr", "bdr", "fund_etf", "stock", "bdr"],
                         "sector": ["Petróleo, Gás e Biocombustíveis", None, None, None, None, "Consumo não Cíclico"]})


def test_the_apps_own_names_win_and_the_rest_take_the_b3_name_sector_and_a_type_label():
    curados = {"PETR4.SA": {"nome": "Petrobras", "setor": "Petróleo"},
               "AAPL34.SA": {"nome": "Apple", "setor": "EUA Tecnologia"}}
    info = info_da_tela(_nomes(), curados)
    assert info["PETR4.SA"] == {"nome": "Petrobras", "setor": "Petróleo", "tipo": "Ação"}
    assert info["AAPL34.SA"] == {"nome": "Apple", "setor": "EUA Tecnologia", "tipo": "BDR"}
    assert info["TSMC34.SA"] == {"nome": "TAIWANSMFAC", "setor": "BDR", "tipo": "BDR"}
    assert info["BOVA11.SA"] == {"nome": "ISHARES BOVA", "setor": "ETF", "tipo": "ETF"}
    assert info["CSUD3.SA"] == {"nome": "CSU DIGITAL", "setor": "", "tipo": "Ação"}
    assert info["JBSS32.SA"] == {"nome": "JBS N.V.", "setor": "Consumo não Cíclico", "tipo": "BDR"}


def test_only_stocks_and_units_get_cvm_fundamentals_so_no_bdr_borrows_a_b3_issuers_multiples():
    # fundamentals are matched by the ticker's first 4 letters: a BDR must never reach that match
    assert acoes_da_tela(_nomes()) == ["PETR4.SA", "CSUD3.SA"]


def test_without_the_lake_file_the_list_is_empty_and_the_panel_keeps_its_own(tmp_path):
    nomes = ler_nomes(tmp_path / "ativos.parquet")
    assert (list(nomes.columns), info_da_tela(nomes, {}), acoes_da_tela(nomes)) == (
        ["security", "name", "asset_type", "sector"], {}, [])


def test_the_apps_own_lists_are_typed_without_the_lake_file_stocks_us_bdrs_and_the_two_index_etfs():
    acoes = {"PETR4.SA": {"nome": "Petrobras", "setor": "Petróleo"}, "TAEE11.SA": {"nome": "Taesa", "setor": "Utilidade"}}
    eua = {"AAPL34.SA": {"nome": "Apple", "setor": "EUA Tecnologia"}, "IVVB11.SA": {"nome": "ETF", "setor": "EUA Índice"}}
    tipos = {ticker: dados["tipo"] for ticker, dados in info_curada(acoes, eua).items()}
    assert tipos == {"PETR4.SA": "Ação", "TAEE11.SA": "Ação", "AAPL34.SA": "BDR", "IVVB11.SA": "ETF"}
