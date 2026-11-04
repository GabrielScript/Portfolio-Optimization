"""CVM material facts of BDRs and ETFs for the market panel (the user's UI list of 2026-11-03)."""

from datetime import date

import pandas as pd

from core.fatos_bdr_etf import codigos_dos_bdrs, fatos_do_fnet, normalizar

IPE = pd.DataFrame({"codigo_cvm": [80209, 80217, 80187, 21334], "nome_companhia": [
    "NU HOLDINGS LTD.", "INTER & CO, INC.", "AURA MINERALS INC", "NUTRIPLANT INDÚSTRIA S.A."]})


def test_names_lose_accents_punctuation_and_spaces():
    assert normalizar("Ishares Ibovespa Fundo de Índice ") == "ISHARESIBOVESPAFUNDODEINDICE"
    assert normalizar("INTER & CO, INC.") == "INTERCOINC"


def test_a_bdr_maps_by_the_lakes_cvm_code_or_else_by_its_b3_name_among_foreign_issuers():
    bdrs = pd.DataFrame({"security": ["AURA33", "ROXO34", "INBR34", "NUTR34", "AAPL34"],
                         "name": ["AURA 360", "NU HOLDINGS", "INTER CO", "NUTRI", "APPLE"]})
    setores = pd.DataFrame({"issuer": ["AURA", "ROXO", "INBR"], "code_cvm": [80187.0, 59382.0, 59444.0]})
    assert codigos_dos_bdrs(bdrs, setores, IPE) == {80187: "AURA33", 80209: "ROXO34", 80217: "INBR34"}


def test_a_b3_name_that_starts_two_foreign_names_maps_to_none():
    ipe = pd.DataFrame({"codigo_cvm": [80001, 80002], "nome_companhia": ["GP INVESTMENTS, LTD", "GP INVESTMENTS II"]})
    bdrs = pd.DataFrame({"security": ["GPIV33"], "name": ["GP INVEST"]})
    assert codigos_dos_bdrs(bdrs, pd.DataFrame(columns=["issuer", "code_cvm"]), ipe) == {}


FNET = pd.DataFrame({
    "id": [1, 2, 3, 4], "nomePregao": ["INVESTO BDOM ", "ISHARES BOVACI", "FII RIONEGRO", "INVESTO BDOM"],
    "descricaoFundo": ["INVESTO BRAZIL DOMESTIC ETF CLASSE DE ÍNDICE", "ISHARES IBOVESPA FUNDO DE ÍNDICE",
                       "RIO NEGRO FII", "INVESTO BRAZIL DOMESTIC ETF CLASSE DE ÍNDICE"],
    "categoriaDocumento": "Fato Relevante", "dataEntrega": ["20/10/2026 18:30", "01/06/2026 09:00",
                                                             "21/10/2026 10:00", "22/10/2026 10:00"],
    "situacaoDocumento": ["A", "A", "A", "C"]})


def test_the_listed_funds_facts_come_from_fundos_net_by_their_b3_trading_name():
    # Fundos.NET (B3's system for the funds' CVM filings) names each fund by its B3 trading name, the register's short
    # name the app already has; a cancelled filing and one older than the window are left out
    etfs = pd.DataFrame({"security": ["BDOM11", "BOVA11"], "name": ["INVESTO BDOM", "ISHARES BOVACI"]})
    fatos = fatos_do_fnet(FNET, etfs, date(2026, 8, 6))
    assert list(fatos.columns) == ["ticker", "empresa", "categoria", "assunto", "data", "link"]
    assert fatos["ticker"].tolist() == ["BDOM11"]
    assert fatos.iloc[0][["categoria", "data", "link"]].tolist() == [
        "Fato Relevante", pd.Timestamp("2026-10-20 18:30"),
        "https://fnet.bmfbovespa.com.br/fnet/publico/exibirDocumento?id=1"]
