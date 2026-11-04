"""CVM material facts of BDRs and ETFs for the market panel (the user's UI list of 2026-11-03)."""

from datetime import date

import pandas as pd

from core.fatos_bdr_etf import (cnpjs_dos_etfs, codigos_dos_bdrs, fatos_dos_fundos, nomes_completos_do_cadastro,
                                normalizar)

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


XML = """<?xml version="1.0" encoding="UTF-8"?>
<Doc><Instrm><InstrmInf><EqtyInf><CrpnNm>ISHARES IBOVESPA FUNDO DE ÍNDICE      </CrpnNm><TckrSymb>BOVA11</TckrSymb>
</EqtyInf></InstrmInf></Instrm><Instrm><InstrmInf><EqtyInf><CrpnNm>PETROLEO BRASILEIRO S.A. PETROBRAS</CrpnNm>
<TckrSymb>PETR4</TckrSymb></EqtyInf></InstrmInf></Instrm></Doc>"""


def test_the_b3_register_gives_each_asked_tickers_full_name(tmp_path):
    arquivo = tmp_path / "BVBG.028.xml"
    arquivo.write_text(XML, encoding="utf-8")
    assert nomes_completos_do_cadastro(arquivo, {"BOVA11", "XINA11"}) == {"BOVA11": "ISHARES IBOVESPA FUNDO DE ÍNDICE"}


EVENTUAIS = pd.DataFrame({
    "CNPJ_FUNDO_CLASSE": ["10.406.511/0001-61", "10.406.511/0001-61", "11.111.111/0001-11", "22.222.222/0001-22"],
    "DENOM_SOCIAL": ["ISHARES IBOVESPA FUNDO DE ÍNDICE - RESPONSABILIDADE LIMITADA"] * 2
                    + ["OUTRO FUNDO", "ISHARES IBOVESPA FUNDO DE ÍNDICE II"],
    "DT_RECEB": ["2026-10-20", "2026-06-01", "2026-10-21", "2026-10-22"],
    "TP_DOC": ["FATO RELEV", "FATO RELEV", "FATO RELEV", "AGO"],
    "LINK_ARQ": ["https://web.cvm.gov.br/a.pdf", "https://web.cvm.gov.br/b.pdf", "https://x/c.pdf",
                 "javascript:alert(1)"]})


def test_an_etf_maps_to_the_one_cnpj_whose_cvm_name_starts_with_its_full_name():
    # two CNPJs start with "ISHARES IBOVESPA FUNDO DE INDICE": the one with that exact name (plus the CVM 175 suffix)
    # wins only when it is the single CNPJ whose name starts with it; here the "II" fund also starts with it
    ambiguo = cnpjs_dos_etfs({"BOVA11": "ISHARES IBOVESPA FUNDO DE ÍNDICE"}, EVENTUAIS)
    assert ambiguo == {}
    so_um = cnpjs_dos_etfs({"BOVA11": "ISHARES IBOVESPA FUNDO DE ÍNDICE"}, EVENTUAIS.iloc[:3])
    assert so_um == {"10.406.511/0001-61": "BOVA11"}


def test_the_fund_facts_are_the_recent_material_ones_in_the_panels_format():
    fatos = fatos_dos_fundos(EVENTUAIS, {"10.406.511/0001-61": "BOVA11"}, date(2026, 8, 1))
    assert list(fatos.columns) == ["ticker", "empresa", "categoria", "assunto", "data", "link"]
    assert fatos["ticker"].tolist() == ["BOVA11"] and fatos["categoria"].tolist() == ["Fato Relevante"]
    assert fatos["data"].tolist() == [pd.Timestamp("2026-10-20")] and fatos["link"].tolist() == [
        "https://web.cvm.gov.br/a.pdf"]
