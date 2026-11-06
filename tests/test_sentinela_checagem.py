from core.sentinela.checagem import validar_decisao, validar_varredura
from core.sentinela.itens import MACRO, Item
from core.sentinela.pacote import montar_pacote

FATO = Item("cvm-a", "cvm", "2026-11-04", "PETR4",
            "PETROBRAS (PETR4), Fato Relevante de 04/11/2026: Pedido de recuperação judicial de 1.234,5 milhões")
SELIC = Item("bcb-s", "bcb", "2026-11-04", MACRO, "Selic meta (% a.a.): 14,75 em 04/11/2026")
PACOTE = montar_pacote("PETR4", "Petróleo", 0.0667, [FATO, SELIC], ["PETR4: último fechamento R$ 30,00 em 05/11/2026"])


def _grave(**mudar):
    resposta = {"ticker": "PETR4", "veredito": "grave",
                "citacoes": [{"id": "cvm-a", "trecho": "pedido de recuperação  judicial de 1.234,5 milhões"}],
                "resumo": "A empresa pediu recuperação judicial de 1.234,5 milhões."}
    return resposta | mudar


def test_a_grave_with_a_verbatim_official_citation_passes():
    varredura = validar_varredura(_grave(), PACOTE)
    assert varredura.veredito == "grave" and varredura.descarte is None
    assert varredura.citacoes == [{"id": "cvm-a", "trecho": "pedido de recuperação  judicial de 1.234,5 milhões"}]


def test_sem_evento_needs_no_citation():
    varredura = validar_varredura({"ticker": "PETR4", "veredito": "sem_evento", "citacoes": [], "resumo": ""}, PACOTE)
    assert varredura.veredito == "sem_evento" and varredura.descarte is None


def test_an_answer_that_is_not_a_dict_or_has_a_wrong_ticker_or_verdict_is_discarded_as_sem_evento():
    for bruto in (None, [], _grave(ticker="VALE3"), _grave(veredito="talvez")):
        varredura = validar_varredura(bruto, PACOTE)
        assert varredura.veredito == "sem_evento" and varredura.descarte


def test_a_citation_of_an_unknown_id_or_a_quote_not_in_the_item_is_discarded():
    assert validar_varredura(_grave(citacoes=[{"id": "cvm-x", "trecho": "pedido"}]), PACOTE).descarte
    assert validar_varredura(_grave(citacoes=[{"id": "cvm-a", "trecho": "pedido de falência"}]), PACOTE).descarte


def test_a_number_in_the_summary_must_be_in_a_cited_quote():
    varredura = validar_varredura(_grave(resumo="Pediu recuperação de 2.000 milhões."), PACOTE)
    assert varredura.veredito == "sem_evento" and "2.000" in varredura.descarte


def test_a_link_in_the_summary_is_discarded():
    assert validar_varredura(_grave(resumo="veja https://x.com"), PACOTE).descarte


def test_atencao_or_grave_without_any_citation_is_discarded():
    assert validar_varredura(_grave(citacoes=[]), PACOTE).descarte


def test_a_grave_citing_only_context_is_downgraded_to_atencao():
    resposta = _grave(citacoes=[{"id": "ctx-PETR4-1", "trecho": "último fechamento R$ 30,00"}],
                      resumo="O fechamento foi R$ 30,00.")
    varredura = validar_varredura(resposta, PACOTE)
    assert varredura.veredito == "atencao" and varredura.descarte is None


def test_a_grave_citing_a_class_outside_the_official_set_is_downgraded():
    assert validar_varredura(_grave(), PACOTE, classes=frozenset({"bcb"})).veredito == "atencao"


def _decisao(**mudar):
    resposta = {"ticker": "PETR4", "acao": "vender", "confianca": 0.8,
                "citacoes": [{"id": "cvm-a", "trecho": "recuperação judicial"}],
                "motivo": "Pedido de recuperação judicial."}
    return resposta | mudar


def test_a_checked_decision_keeps_its_action_and_confidence():
    decisao = validar_decisao(_decisao(), PACOTE)
    assert (decisao.acao, decisao.confianca, decisao.descarte) == ("vender", 0.8, None)


def test_a_decision_failing_any_check_becomes_manter():
    for bruto in (None, _decisao(acao="comprar"), _decisao(confianca=1.5), _decisao(confianca="alta"),
                  _decisao(citacoes=[]), _decisao(motivo="queda de 50%"), _decisao(ticker="VALE3")):
        decisao = validar_decisao(bruto, PACOTE)
        assert decisao.acao == "manter" and decisao.descarte, bruto


def test_manter_needs_no_citation():
    decisao = validar_decisao(_decisao(acao="manter", citacoes=[], confianca=0.6, motivo="nada grave"), PACOTE)
    assert decisao.acao == "manter" and decisao.descarte is None and decisao.confianca == 0.6
