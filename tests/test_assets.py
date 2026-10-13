"""O universo é só de ações da B3: nada de cripto nem de ações listadas nos EUA."""

import assets


def test_metadados_so_tem_tickers_da_b3():
    assert all(t.endswith(".SA") for t in assets.get_ticker_info())


def test_startups_sao_so_as_listadas_na_b3():
    assert all(t.endswith(".SA") for t in assets.get_startup_tickers())


def test_modulo_nao_expoe_cripto():
    assert not hasattr(assets, "CRIPTO") and not hasattr(assets, "get_cripto_tickers")
