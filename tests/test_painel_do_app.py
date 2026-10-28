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
