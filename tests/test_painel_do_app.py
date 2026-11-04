"""The market panel's files (scripts/painel_do_app.py): which tickers get CVM fundamentals and filings."""

import pandas as pd


def test_the_panel_files_cover_the_apps_181_and_every_other_b3_stock_of_the_lake_but_no_bdr_or_etf():
    from assets import get_all_tickers
    from scripts.painel_do_app import tickers_do_painel
    nomes = pd.DataFrame({"security": ["PETR4", "AMAR3", "AAPL34", "BOVA11"], "name": ["", "", "", ""],
                          "asset_type": ["stock", "stock", "bdr", "fund_etf"], "sector": [None] * 4})
    tickers = tickers_do_painel(nomes)
    assert (tickers[:181] == get_all_tickers(), tickers[181:], len(set(tickers)) == len(tickers)) == (
        True, ["AMAR3.SA"], True)


def test_fundos_net_is_read_page_by_page_until_its_total(monkeypatch):
    from datetime import date
    import requests
    from scripts import painel_do_app

    paginas = []

    class Resposta:
        def __init__(self, inicio):
            self.inicio = inicio

        def raise_for_status(self):
            return None

        def json(self):
            fim = min(self.inicio + painel_do_app.POR_PAGINA, 450)
            return {"recordsTotal": 450, "data": [{"id": i} for i in range(self.inicio, fim)]}

    def get(url, params, headers, timeout):
        paginas.append(params["s"])
        assert params["idCategoriaDocumento"] == painel_do_app.CATEGORIA_FATO_RELEVANTE
        return Resposta(params["s"])

    monkeypatch.setattr(requests, "get", get)
    registros = painel_do_app.baixar_fnet(date(2026, 8, 6), date(2026, 11, 4))
    assert paginas == [0, 200, 400] and len(registros) == 450 and registros["id"].is_unique


def test_the_funds_facts_join_the_companies_facts_newest_first_whatever_their_column_types():
    # the IPE's columns can be categorical; Fundos.NET's are text: the join reads both as text
    import pandas as pd
    from scripts.painel_do_app import juntar_fatos
    colunas = ["ticker", "empresa", "categoria", "assunto", "data", "link"]
    from datetime import date
    empresas = pd.DataFrame([["PETR4", "PETROBRAS", "Fato Relevante", "x", date(2026, 10, 20), "l1"],
                             ["VALE3", "VALE", "Fato Relevante", "z", date(2026, 10, 21), "l3"]],
                            columns=colunas).astype({"empresa": "category", "categoria": "category"})
    fundos = pd.DataFrame([["BDOM11", "INVESTO", "Fato Relevante", "y", pd.Timestamp("2026-10-21 18:00"), "l2"]],
                          columns=colunas)
    juntos = juntar_fatos(empresas, fundos)
    assert juntos["ticker"].tolist() == ["BDOM11", "VALE3", "PETR4"]          # same day: by ticker
    assert set(map(type, juntos["data"])) == {date}                          # the IPE's dates, as the light reads
    assert juntar_fatos(empresas, None)["ticker"].tolist() == ["PETR4", "VALE3"]
