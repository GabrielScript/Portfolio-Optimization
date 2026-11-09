"""Unit tests for the Liquid review gate (no network: the HTTP call is injected)."""

from __future__ import annotations

from scripts.revisao_liquid import (PERGUNTAS, Revisao, Resposta, dividir_por_arquivo, pior_acao, revisar)

DIFF = """diff --git a/core/qml.py b/core/qml.py
--- a/core/qml.py
+++ b/core/qml.py
@@ -1 +1 @@
-x = 1
+x = 2
diff --git a/docs/spec.md b/docs/spec.md
--- a/docs/spec.md
+++ b/docs/spec.md
@@ -1 +1 @@
-old
+new
"""


class _Resposta:
    def __init__(self, dados: dict, status: int = 200) -> None:
        self._dados = dados
        self.status_code = status
        self.ok = status == 200
        self.text = str(dados)

    def json(self) -> dict:
        return self._dados


def _post_com(probabilidades: dict[str, float]):
    def _post(_url, json, timeout, headers):                       # noqa: A002 - the shape of requests.post
        assert headers["Authorization"].startswith("Bearer ")
        assert json["model"] == "d1.1"
        return _Resposta({"model": "d1.1", "answers": {nome: {"type": "noul", "noul": valor}
                                                       for nome, valor in probabilidades.items()},
                          "usage": {"input_tokens": 10, "cost": 0.0}})
    return _post


def test_splitter_keeps_one_section_per_file() -> None:
    secoes = dividir_por_arquivo(DIFF)
    assert list(secoes) == ["core/qml.py", "docs/spec.md"]
    assert secoes["core/qml.py"].startswith("diff --git a/core/qml.py")
    assert "+x = 2" in secoes["core/qml.py"]
    assert "old" not in secoes["core/qml.py"]


def test_a_confident_clean_review_is_auto() -> None:
    revisao = revisar("request", "diff", "", "d1.1", "chave",
                      post=_post_com({"correto": 0.95, "defeito": 0.02, "espec": 0.9, "testes": 0.85,
                                      "producao": 0.01, "futuro": 0.01}))
    assert revisao.acao() == "auto"


def test_a_high_defect_probability_blocks_auto() -> None:
    revisao = revisar("request", "diff", "", "d1.1", "chave",
                      post=_post_com({"correto": 0.95, "defeito": 0.70, "espec": 0.9, "testes": 0.85,
                                      "producao": 0.01, "futuro": 0.01}))
    assert revisao.acao() == "review"


def test_a_missing_answer_escalates() -> None:
    revisao = revisar("request", "diff", "", "d1.1", "chave",
                      post=_post_com({"correto": 0.95, "defeito": 0.02, "espec": 0.9, "testes": 0.85,
                                      "producao": 0.01}))
    assert len(revisao.respostas) == len(PERGUNTAS)
    assert revisao.acao() == "escalate"


def test_the_worst_file_decides_the_patch() -> None:
    auto = Revisao("d1.1", tuple(Resposta(nome, 0.95 if quer else 0.01) for nome, (_, quer) in PERGUNTAS.items()),
                   1.0, 10, 0.0, 100, 0)
    revisao = Revisao("d1.1", tuple(Resposta(nome, 0.10 if quer else 0.90) for nome, (_, quer) in PERGUNTAS.items()),
                      1.0, 10, 0.0, 100, 0)
    assert auto.acao() == "auto"
    assert pior_acao({"a.py": auto, "b.py": revisao}) == pior_acao({"b.py": revisao})
    assert pior_acao({"a.py": auto}) == "auto"
