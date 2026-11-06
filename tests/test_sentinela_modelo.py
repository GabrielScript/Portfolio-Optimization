import json
from datetime import datetime, timezone

import requests

from core.sentinela import modelo
from core.sentinela.checagem import Varredura
from core.sentinela.itens import Item
from core.sentinela.pacote import montar_pacote, pacote_da_decisao

FATO = Item("cvm-a", "cvm", "2026-11-04", "PETR4", "PETROBRAS (PETR4), Fato Relevante de 04/11/2026: Pedido de "
                                                   "recuperação judicial")
ANTIGO = Item("cvm-b", "cvm", "2026-10-01", "PETR4", "PETROBRAS (PETR4), Comunicado de 01/10/2026: Apresentação")
PACOTE = montar_pacote("PETR4", "Petróleo", 0.0667, [FATO], ["PETR4: último fechamento R$ 30,00 em 05/11/2026"])
AGORA = datetime(2026, 11, 6, 15, tzinfo=timezone.utc)      # off-peak in UTC


class _Resposta:
    def __init__(self, conteudo: object, status: int = 200, fim: str = "stop"):
        self.status_code, self._conteudo, self._fim = status, conteudo, fim

    def json(self):
        texto = self._conteudo if isinstance(self._conteudo, str) else json.dumps(self._conteudo)
        return {"choices": [{"message": {"content": texto}, "finish_reason": self._fim}],
                "usage": {"prompt_tokens": 1000, "prompt_cache_hit_tokens": 800, "prompt_cache_miss_tokens": 200,
                          "completion_tokens": 300}}


def _post(conteudo, corpos: list, status: int = 200, fim: str = "stop"):
    def post(url, json, timeout, headers):
        corpos.append(json)
        return _Resposta(conteudo, status, fim)
    return post


GRAVE = {"ticker": "PETR4", "veredito": "grave", "citacoes": [{"id": "cvm-a", "trecho": "recuperação judicial"}],
         "resumo": "Pedido de recuperação judicial."}


def test_scan_sends_the_fixed_instructions_first_and_the_pack_with_low_effort_and_checks_the_answer():
    corpos = []
    varredura, resposta = modelo.varrer(PACOTE, "chave", AGORA, _post(GRAVE, corpos))
    corpo = corpos[0]
    assert corpo["model"] == modelo.MODELO and corpo["reasoning_effort"] == "low"
    assert corpo["response_format"] == {"type": "json_object"} and "tools" not in corpo
    assert corpo["messages"][0] == {"role": "system", "content": modelo.INSTRUCOES_VARREDURA}
    assert corpo["messages"][1]["content"] == PACOTE.texto()
    assert varredura.veredito == "grave" and resposta.erro is None
    assert resposta.uso.entrada == 1000 and resposta.uso.cache == 800 and resposta.uso.custo_usd > 0


def test_the_system_message_is_the_same_bytes_for_every_name():
    outro = montar_pacote("VALE3", "Mineração", 0.0667, [], [])
    corpos = []
    modelo.varrer(PACOTE, "c", AGORA, _post(GRAVE, corpos))
    modelo.varrer(outro, "c", AGORA, _post({"ticker": "VALE3", "veredito": "sem_evento", "citacoes": [],
                                            "resumo": ""}, corpos))
    assert corpos[0]["messages"][0] == corpos[1]["messages"][0]


def test_an_api_failure_or_a_non_json_or_cut_answer_is_a_discarded_sem_evento():
    for post in (_post(GRAVE, [], status=401), _post("isto não é json", []), _post(GRAVE, [], fim="length")):
        varredura, resposta = modelo.varrer(PACOTE, "c", AGORA, post)
        assert varredura.veredito == "sem_evento" and varredura.descarte and resposta.erro


def test_a_network_failure_is_caught():
    def post(url, json, timeout, headers):
        raise requests.ConnectionError("sem rede")
    varredura, resposta = modelo.varrer(PACOTE, "c", AGORA, post)
    assert varredura.veredito == "sem_evento" and "rede" in resposta.erro and resposta.falha_da_api


def test_a_bad_answer_is_not_an_api_failure():
    _, resposta = modelo.varrer(PACOTE, "c", AGORA, _post("isto não é json", []))
    assert resposta.erro and not resposta.falha_da_api


def test_decision_uses_max_effort_the_decision_instructions_and_the_scan_citations():
    varredura = Varredura("grave", GRAVE["citacoes"], GRAVE["resumo"])
    decisao_pacote = pacote_da_decisao(PACOTE, [ANTIGO, FATO])
    corpos = []
    resposta_modelo = {"ticker": "PETR4", "acao": "vender", "confianca": 0.8,
                       "citacoes": [{"id": "cvm-b", "trecho": "Apresentação"}], "motivo": "Ver itens."}
    decisao, resposta = modelo.decidir(decisao_pacote, varredura, "c", AGORA, _post(resposta_modelo, corpos))
    corpo = corpos[0]
    assert corpo["reasoning_effort"] == "max"
    assert corpo["messages"][0] == {"role": "system", "content": modelo.INSTRUCOES_DECISAO}
    assert "cvm-a" in corpo["messages"][1]["content"] and "recuperação judicial" in corpo["messages"][1]["content"]
    assert decisao.acao == "vender" and decisao.confianca == 0.8


def test_decision_pack_adds_up_to_10_recent_filings_once():
    antigos = [Item(f"cvm-{i}", "cvm", f"2026-09-{i + 1:02d}", "PETR4", f"fato {i}") for i in range(12)]
    decisao_pacote = pacote_da_decisao(PACOTE, antigos + [FATO])
    ids = [item.id for item in decisao_pacote.itens]
    assert ids[0] == "cvm-a" and len(ids) == 1 + 10 and len(set(ids)) == len(ids)
    assert ids[1:] == [f"cvm-{i}" for i in range(11, 1, -1)]


def test_a_failed_decision_call_is_manter():
    varredura = Varredura("grave", GRAVE["citacoes"], GRAVE["resumo"])
    decisao, resposta = modelo.decidir(pacote_da_decisao(PACOTE, []), varredura, "c", AGORA,
                                       _post(GRAVE, [], status=500))
    assert decisao.acao == "manter" and decisao.descarte and resposta.erro


def test_budget_caps_40_low_and_6_max_calls_a_day():
    orcamento = modelo.Orcamento()
    for _ in range(40):
        assert orcamento.gastar("varredura")
    assert not orcamento.gastar("varredura")
    for _ in range(6):
        assert orcamento.gastar("decisao")
    assert not orcamento.gastar("decisao")


def test_budget_of_a_day_counts_the_calls_already_logged():
    orcamento = modelo.Orcamento.do_dia({"varredura": 39, "decisao": 6})
    assert orcamento.gastar("varredura") and not orcamento.gastar("varredura") and not orcamento.gastar("decisao")
