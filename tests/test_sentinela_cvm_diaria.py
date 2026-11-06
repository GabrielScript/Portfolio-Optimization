from datetime import date

import pandas as pd

from core.sentinela import cvm_diaria
from core.sentinela.itens import itens_cvm

ACOES = ("<i class='fi-download' onclick=OpenDownloadDocumentos('1108325','1','1583619','IPE')> </i>")
LINHA = "$&".join(["02528-3", "AERIS S/A", "Fato Relevante", " - ",
                   "<spanOrder>Pedido de recuperação extrajudicial</spanOrder> - ",
                   "<spanOrder>20261103</spanOrder> 03/11/2026", "<spanOrder>20261104</spanOrder> 04/11/2026 00:04",
                   "Ativo", "1", "AP", ACOES, "Pedido de recuperação extrajudicial", ""])
OUTRA = LINHA.replace("02528-3", "00951-2").replace("AERIS S/A", "PETROBRAS").replace("Fato Relevante",
                                                                                    "Comunicado ao Mercado")
DADOS = LINHA + "&*" + OUTRA + "&*"


def test_the_listing_is_parsed_into_one_row_per_filing_with_the_same_link_as_the_weekly_open_data():
    lista = cvm_diaria.ler_lista(DADOS)
    assert len(lista) == 2
    linha = lista.iloc[0]
    assert linha["codigo_cvm"] == 25283 and linha["categoria"] == "Fato Relevante"
    assert linha["assunto"] == "Pedido de recuperação extrajudicial"
    assert linha["entregue_em"] == pd.Timestamp("2026-11-04 00:04")
    assert linha["link"] == ("https://www.rad.cvm.gov.br/ENET/frmDownloadDocumento.aspx?Tela=ext&descTipo=IPE&"
                             "CodigoInstituicao=1&numProtocolo=1583619&numSequencia=1108325&numVersao=1")


def test_filings_of_the_watched_issuers_get_their_tickers_and_the_fatos_columns():
    lista = cvm_diaria.ler_lista(DADOS)
    fatos = cvm_diaria.fatos_dos_nomes(lista, {9512: ["PETR4"], 1: ["XXXX3"]})
    assert list(fatos["ticker"]) == ["PETR4"]
    assert set(fatos.columns) >= {"ticker", "empresa", "categoria", "assunto", "data", "link"}
    assert itens_cvm(fatos, ["PETR4"])[0].data == "2026-11-04"


def test_the_request_asks_for_fato_relevante_and_comunicado_between_two_days():
    corpo = cvm_diaria.corpo(date(2026, 11, 4), date(2026, 11, 5))
    assert "dataDe: '04/11/2026'" in corpo and "dataAte: '05/11/2026'" in corpo
    assert "categoria: 'IPE_4_-1_-1,IPE_6_-1_-1'" in corpo and "periodo: '2'" in corpo


def test_issuer_codes_map_to_the_composite_tickers():
    setores = pd.DataFrame({"issuer": ["PETR", "VALE"], "code_cvm": [9512, 4170]})
    assert cvm_diaria.codigos(setores, ["PETR4", "PETR3", "ITUB4"]) == {9512: ["PETR4", "PETR3"]}


def test_a_session_error_or_an_error_flag_raises():
    class Resposta:
        ok, status_code = True, 200

        def __init__(self, conteudo):
            self.conteudo = conteudo

        def json(self):
            return self.conteudo

    class Sessao:
        def __init__(self, conteudo):
            self.conteudo = conteudo

        def get(self, url, **kwargs):
            return Resposta(None)

        def post(self, url, **kwargs):
            return Resposta(self.conteudo)

    assert len(cvm_diaria.consultar(Sessao({"d": {"temErro": False, "dados": DADOS}}), date(2026, 11, 4),
                                    date(2026, 11, 5))) == 2
    for ruim in ({"d": {"temErro": True, "msgErro": "x", "dados": ""}}, {"d": {"expirouSessao": True}}):
        try:
            cvm_diaria.consultar(Sessao(ruim), date(2026, 11, 4), date(2026, 11, 5))
        except cvm_diaria.ErroDaCvm:
            continue
        raise AssertionError(ruim)
