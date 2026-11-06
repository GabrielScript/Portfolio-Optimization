import requests

from core.sentinela import juizes
from core.sentinela.itens import MACRO, Item

FATO = Item("cvm-a", "cvm", "2026-11-04", "PETR4", "PETROBRAS (PETR4), Fato Relevante de 04/11/2026: Recompra")
SELIC = Item("bcb-s", "bcb", "2026-11-04", MACRO, "Selic meta (% a.a.): 14,75 em 04/11/2026")


class _Resposta:
    def __init__(self, ok=True, status=200):
        self.ok, self.status_code = ok, status

    def json(self):
        return {"answers": {"material": {"noul": 0.7}, "nova": {"noul": 0.9}},
                "usage": {"input_tokens": 105, "cost": 0.000002}, "model": "d1.1-x"}


def test_each_company_item_goes_to_both_judges_with_two_questions_and_macro_items_never_do():
    pedidos = []

    def post(url, json, timeout, headers):
        pedidos.append((url, json))
        return _Resposta()

    juizos = juizes.julgar([FATO, SELIC], {"PETR4": ("PETROBRAS", "Petróleo")}, "chave", post)
    assert [(j.juiz, j.item) for j in juizos] == [("d1.1", "cvm-a"), ("d1-ultra", "cvm-a")]
    url, corpo = pedidos[0]
    assert url == juizes.URL and corpo["model"] == "d1.1" and set(corpo["questions"]) == {"material", "nova"}
    assert "PETROBRAS" in corpo["state"] and "Petróleo" in corpo["state"] and FATO.texto in corpo["state"]
    assert juizos[0].p_material == 0.7 and juizos[0].p_nova == 0.9 and juizos[0].tokens == 105
    assert juizos[0].erro is None


def test_without_a_key_no_judge_is_called():
    def post(*args, **kwargs):
        raise AssertionError("called")
    assert juizes.julgar([FATO], {}, None, post) == []


def test_a_failing_judge_is_logged_with_its_error_and_no_probability():
    def post(url, json, timeout, headers):
        if json["model"] == "d1.1":
            raise requests.ConnectionError("x")
        return _Resposta(ok=False, status=503)
    juizos = juizes.julgar([FATO], {}, "chave", post)
    assert all(j.p_material is None and j.erro for j in juizos) and len(juizos) == 2
