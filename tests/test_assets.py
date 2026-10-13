"""O universo é só de ações da B3: nada de cripto, de ações dos EUA nem de listas paralelas."""

import assets


def test_metadados_so_tem_tickers_da_b3():
    assert all(t.endswith(".SA") for t in assets.get_ticker_info())


def test_modulo_nao_expoe_cripto():
    assert not hasattr(assets, "CRIPTO") and not hasattr(assets, "get_cripto_tickers")


def test_modulo_nao_expoe_lista_paralela_de_startups():
    nomes = ("STARTUPS_BR", "STARTUPS_US", "get_startup_tickers", "get_startup_info")
    assert not [n for n in nomes if hasattr(assets, n)]
