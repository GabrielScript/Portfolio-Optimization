"""
ativos_eua.py - The screen's optional US leg: BDRs of US companies traded on the B3 and two US index ETFs.

The 30 BDRs are the US companies (level-I BDRs, suffix 34) above the app's floor on 2026-10-23: traded in >= 90% of
the sessions of the last 12 months with a median daily volume >= R$ 1 mi. Left out: ROXO34, STOC34, MELI34 (Latin
American businesses), TSMC34, BABA34, ASML34 (not US) and S2GM34 (issuer not checked). Same list as the lake's
``APP_US_SET`` (``b3lake.export``), which puts their prices in ``lago/precos_diarios.parquet``.

Kept apart from ``assets.py``, which is in the paper-v1 guard path: the 181 and the paper do not change.

Limitations shown on the screen: the lake has no BDR dividends (the BDR columns are price-only, splits adjusted, in
reais with the dollar inside); BDR gains pay 15% income tax with no R$ 20 mil monthly exemption, and their dividends
go through the carnê-leão. BDRs trade in lots of 1 (no fractional market).
"""

ATIVOS_EUA = [
    {"ticker": "AAPL34", "nome": "Apple", "setor": "EUA Tecnologia"},
    {"ticker": "MSFT34", "nome": "Microsoft", "setor": "EUA Tecnologia"},
    {"ticker": "NVDC34", "nome": "Nvidia", "setor": "EUA Tecnologia"},
    {"ticker": "GOGL34", "nome": "Alphabet", "setor": "EUA Tecnologia"},
    {"ticker": "M1TA34", "nome": "Meta Platforms", "setor": "EUA Tecnologia"},
    {"ticker": "AVGO34", "nome": "Broadcom", "setor": "EUA Tecnologia"},
    {"ticker": "ORCL34", "nome": "Oracle", "setor": "EUA Tecnologia"},
    {"ticker": "NFLX34", "nome": "Netflix", "setor": "EUA Tecnologia"},
    {"ticker": "A1MD34", "nome": "AMD", "setor": "EUA Tecnologia"},
    {"ticker": "ITLC34", "nome": "Intel", "setor": "EUA Tecnologia"},
    {"ticker": "MUTC34", "nome": "Micron", "setor": "EUA Tecnologia"},
    {"ticker": "P2LT34", "nome": "Palantir", "setor": "EUA Tecnologia"},
    {"ticker": "M2ST34", "nome": "Strategy", "setor": "EUA Tecnologia"},
    {"ticker": "C2OI34", "nome": "Coinbase", "setor": "EUA Tecnologia"},
    {"ticker": "D1EL34", "nome": "Dell", "setor": "EUA Tecnologia"},
    {"ticker": "M2RV34", "nome": "Marvell", "setor": "EUA Tecnologia"},
    {"ticker": "S1TX34", "nome": "Seagate", "setor": "EUA Tecnologia"},
    {"ticker": "W1DC34", "nome": "Western Digital", "setor": "EUA Tecnologia"},
    {"ticker": "AMZO34", "nome": "Amazon", "setor": "EUA Consumo"},
    {"ticker": "TSLA34", "nome": "Tesla", "setor": "EUA Consumo"},
    {"ticker": "COCA34", "nome": "Coca-Cola", "setor": "EUA Consumo"},
    {"ticker": "DISB34", "nome": "Disney", "setor": "EUA Consumo"},
    {"ticker": "NIKE34", "nome": "Nike", "setor": "EUA Consumo"},
    {"ticker": "BERK34", "nome": "Berkshire Hathaway", "setor": "EUA Financeiro"},
    {"ticker": "JPMC34", "nome": "JPMorgan", "setor": "EUA Financeiro"},
    {"ticker": "GSGI34", "nome": "Goldman Sachs", "setor": "EUA Financeiro"},
    {"ticker": "VISA34", "nome": "Visa", "setor": "EUA Financeiro"},
    {"ticker": "LILY34", "nome": "Eli Lilly", "setor": "EUA Saúde"},
    {"ticker": "EXXO34", "nome": "Exxon Mobil", "setor": "EUA Energia e Materiais"},
    {"ticker": "FCXO34", "nome": "Freeport-McMoRan", "setor": "EUA Energia e Materiais"},
    {"ticker": "IVVB11", "nome": "ETF S&P 500 (iShares)", "setor": "EUA Índice"},
    {"ticker": "NASD11", "nome": "ETF Nasdaq-100 (Trend)", "setor": "EUA Índice"},
]

AVISO_EUA = ("EUA: BDRs sem dividendos no lago (retorno só de preço, em reais, com o dólar dentro); ganho em BDR paga "
             "15% de IR sem a isenção de R$ 20 mil por mês das ações, e os dividendos vão pelo carnê-leão. "
             "Desligado por padrão: no backtest de 27/10/2026, o EUA melhorou a Mín. Vol e piorou o Kelly.")


def get_tickers_eua() -> list[str]:
    """US tickers in the app's format (with .SA)."""
    return [f"{a['ticker']}.SA" for a in ATIVOS_EUA]


def get_ticker_info_eua() -> dict:
    """Ticker (app format) -> {nome, setor} of the US leg, as ``assets.get_ticker_info``."""
    return {f"{a['ticker']}.SA": {"nome": a["nome"], "setor": a["setor"]} for a in ATIVOS_EUA}
