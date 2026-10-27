"""The optional US leg of the screen (``ativos_eua.py``): 30 US BDRs + IVVB11 + NASD11, apart from the 181."""

from assets import ATIVOS_B3
from ativos_eua import ATIVOS_EUA, get_ticker_info_eua, get_tickers_eua


def test_eua_tem_30_bdrs_e_2_etfs_um_por_emissor_fora_da_lista_da_b3_e_em_setores_proprios():
    tickers = [ativo["ticker"] for ativo in ATIVOS_EUA]
    brasil = {ativo["ticker"] for ativo in ATIVOS_B3}
    assert (sum(t.endswith("34") for t in tickers), sorted(t for t in tickers if t.endswith("11")),
            len({t[:4] for t in tickers}), brasil & set(tickers),
            all(ativo["setor"].startswith("EUA ") for ativo in ATIVOS_EUA)) == (
        30, ["IVVB11", "NASD11"], 32, set(), True)


def test_tickers_e_info_no_formato_do_app():
    assert get_tickers_eua()[0].endswith(".SA") and get_ticker_info_eua()["AAPL34.SA"] == {
        "nome": "Apple", "setor": "EUA Tecnologia"}
