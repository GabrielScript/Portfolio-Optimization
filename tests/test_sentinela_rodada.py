import json
from datetime import datetime

import numpy as np
import pandas as pd

from core.sentinela import modelo, registro, rodada
from core.sentinela.itens import MACRO, Item

AGORA = datetime(2026, 12, 2, 10, 0)
PREGOES = pd.bdate_range("2026-11-02", "2027-01-29")
FATO = Item("cvm-a", "cvm", "2026-12-01", "PETR4", "PETROBRAS (PETR4), Fato Relevante de 01/12/2026: Pedido de "
                                                   "recuperação judicial")
SELIC = Item("bcb-s", "bcb", "2026-12-01", MACRO, "Selic meta (% a.a.): 14,75 em 01/12/2026")


def _entrada(itens, agora=AGORA) -> rodada.Entrada:
    dias = pd.bdate_range("2026-10-01", "2026-12-01")
    fechamentos = pd.DataFrame({"PETR4": np.linspace(30, 32, len(dias)), "VALE3": np.linspace(60, 61, len(dias))},
                               index=dias)
    return rodada.Entrada(agora=agora, carteira={"PETR4": 0.5, "VALE3": 0.5}, registrada=pd.Timestamp("2026-10-27"),
                          empresas={"PETR4": ("PETROBRAS", "Petróleo"), "VALE3": ("VALE", "Mineração")},
                          itens=itens, fatos=[FATO], fechamentos=fechamentos, pregoes=PREGOES)


class _Resposta:
    def __init__(self, conteudo, status=200):
        self.status_code, self.ok, self._conteudo = status, status == 200, conteudo

    def json(self):
        if "answers" in self._conteudo:
            return self._conteudo
        return {"choices": [{"message": {"content": json.dumps(self._conteudo)}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 100, "completion_tokens": 10}}


def _deepseek(vereditos: dict, decisao: dict | None = None, status: int = 200, chamadas: list | None = None):
    def post(url, json, timeout, headers):
        sistema, usuario = json["messages"][0]["content"], json["messages"][1]["content"]
        ticker = usuario.split("Ação: ")[1].split(".")[0]
        if chamadas is not None:
            chamadas.append((ticker, json["reasoning_effort"]))
        if sistema == modelo.INSTRUCOES_DECISAO:
            return _Resposta(decisao, status)
        veredito = vereditos.get(ticker, "sem_evento")
        citacoes = [{"id": "cvm-a", "trecho": "recuperação judicial"}] if veredito != "sem_evento" else []
        return _Resposta({"ticker": ticker, "veredito": veredito, "citacoes": citacoes,
                          "resumo": "Pedido de recuperação judicial." if citacoes else ""}, status)
    return post


def _liquid(url, json, timeout, headers):
    return _Resposta({"answers": {"material": {"noul": 0.9}, "nova": {"noul": 0.8}}, "usage": {"input_tokens": 90}})


def _servicos(post, emails=None, chave="k") -> rodada.Servicos:
    enviados = emails if emails is not None else []
    return rodada.Servicos(chave_deepseek=chave, chave_liquid="l", post_deepseek=post, post_liquid=_liquid,
                           email=lambda assunto, corpo: enviados.append((assunto, corpo)))


VENDER = {"ticker": "PETR4", "acao": "vender", "confianca": 0.8,
          "citacoes": [{"id": "cvm-a", "trecho": "recuperação judicial"}], "motivo": "Recuperação judicial."}


def test_no_new_item_means_no_call_and_no_log(tmp_path):
    arquivos = rodada.Arquivos(tmp_path)
    chamadas = []
    relatorio = rodada.rodar(_entrada([]), arquivos, _servicos(_deepseek({}, chamadas=chamadas)))
    assert chamadas == [] and relatorio.chamadas == 0 and not arquivos.log.exists()


def test_each_name_with_new_items_gets_one_low_scan_and_its_items_are_marked_seen(tmp_path):
    arquivos = rodada.Arquivos(tmp_path)
    chamadas = []
    rodada.rodar(_entrada([FATO, SELIC]), arquivos, _servicos(_deepseek({}, chamadas=chamadas)))
    assert sorted(chamadas) == [("PETR4", "low"), ("VALE3", "low")]
    log = registro.ler_log(arquivos.log)
    assert sorted(log.loc[log["etapa"] == "varredura", "ticker"]) == ["PETR4", "VALE3"]
    assert sorted(log.loc[log["etapa"].str.startswith("juiz"), "etapa"]) == ["juiz:d1-ultra", "juiz:d1.1"]
    chamadas.clear()
    rodada.rodar(_entrada([FATO, SELIC]), arquivos, _servicos(_deepseek({}, chamadas=chamadas)))
    assert chamadas == []


def test_a_grave_goes_to_a_max_decision_an_accepted_sale_is_recorded_and_emailed(tmp_path):
    arquivos = rodada.Arquivos(tmp_path)
    chamadas, emails = [], []
    relatorio = rodada.rodar(_entrada([FATO]), arquivos,
                             _servicos(_deepseek({"PETR4": "grave"}, VENDER, chamadas=chamadas), emails))
    assert ("PETR4", "max") in chamadas and relatorio.graves == ["PETR4"]
    acoes = registro.ler_acoes(arquivos.acoes)
    assert list(acoes["estado"]) == ["aceita"] and acoes["executa_em"].iloc[0] == pd.Timestamp("2026-12-02")
    assert len(emails) == 1 and "PETR4" in emails[0][0] and "recuperação judicial" in emails[0][1]
    assert "80%" in emails[0][1] and "fictícia" in emails[0][1]


def test_a_sale_below_70_percent_is_refused_but_the_grave_is_still_emailed(tmp_path):
    arquivos = rodada.Arquivos(tmp_path)
    emails = []
    rodada.rodar(_entrada([FATO]), arquivos, _servicos(_deepseek({"PETR4": "grave"}, VENDER | {"confianca": 0.6}),
                                                       emails))
    acoes = registro.ler_acoes(arquivos.acoes)
    assert list(acoes["estado"]) == ["recusada"] and "0.60" in acoes["motivo"].iloc[0]
    assert len(emails) == 1


def test_over_the_daily_cap_the_names_wait_and_their_items_are_not_marked(tmp_path):
    arquivos = rodada.Arquivos(tmp_path)
    registro.acrescentar_log(arquivos.log, [{"hora": AGORA.replace(hour=8), "ticker": "X", "etapa": "varredura"}] * 40)
    chamadas = []
    relatorio = rodada.rodar(_entrada([FATO]), arquivos, _servicos(_deepseek({}, chamadas=chamadas)))
    assert chamadas == [] and relatorio.adiados == ["PETR4"]
    assert rodada.rodar(_entrada([FATO], AGORA.replace(day=3)), arquivos,
                        _servicos(_deepseek({}, chamadas=chamadas))).chamadas == 1


def test_an_api_failure_leaves_the_items_for_the_next_run(tmp_path):
    arquivos = rodada.Arquivos(tmp_path)
    rodada.rodar(_entrada([FATO]), arquivos, _servicos(_deepseek({}, status=503)))
    chamadas = []
    rodada.rodar(_entrada([FATO]), arquivos, _servicos(_deepseek({}, chamadas=chamadas)))
    assert chamadas == [("PETR4", "low")]


def test_without_the_deepseek_key_nothing_is_called_or_marked(tmp_path):
    arquivos = rodada.Arquivos(tmp_path)
    relatorio = rodada.rodar(_entrada([FATO]), arquivos, _servicos(_deepseek({}), chave=None))
    assert relatorio.erro and not arquivos.vistos.exists()


def test_the_19h_run_sends_one_digest_of_the_days_atencao(tmp_path):
    arquivos = rodada.Arquivos(tmp_path)
    emails = []
    rodada.rodar(_entrada([FATO]), arquivos, _servicos(_deepseek({"PETR4": "atencao"}), emails))
    assert emails == []
    noite = AGORA.replace(hour=19)
    rodada.rodar(_entrada([], noite), arquivos, _servicos(_deepseek({}), emails))
    rodada.rodar(_entrada([], noite.replace(minute=30)), arquivos, _servicos(_deepseek({}), emails))
    assert len(emails) == 1 and "PETR4" in emails[0][1] and "atenção" in emails[0][0].lower()


def test_accepted_sales_execute_in_the_account_and_are_marked_executed(tmp_path):
    arquivos = rodada.Arquivos(tmp_path)
    rodada.rodar(_entrada([FATO]), arquivos, _servicos(_deepseek({"PETR4": "grave"}, VENDER)))
    dias = pd.bdate_range("2026-12-01", periods=5)
    precos = pd.DataFrame({"PETR4": np.linspace(30, 25, 5), "VALE3": np.full(5, 60.0)}, index=dias)
    registro_composta = pd.DataFrame({"data": pd.Timestamp("2026-10-27"), "ticker": ["PETR4", "VALE3"],
                                      "peso": [0.45, 0.45]})
    feitas = rodada.atualizar_conta(arquivos, registro_composta, precos, pd.Series(0.0, index=dias),
                                    lambda ticker, valor: 0.0, inicio=dias[0])
    assert feitas == ["PETR4-2026-12-02T10:00"]
    acoes = registro.ler_acoes(arquivos.acoes)
    assert list(acoes["estado"]) == ["aceita", "executada"]
    assert rodada.atualizar_conta(arquivos, registro_composta, precos, pd.Series(0.0, index=dias),
                                  lambda ticker, valor: 0.0, inicio=dias[0]) == []


def test_seeding_marks_every_current_item_seen_for_every_name_without_any_call(tmp_path):
    arquivos = rodada.Arquivos(tmp_path)
    marcados = rodada.semear(_entrada([FATO, SELIC]), arquivos)
    assert marcados == 3                     # FATO for PETR4, SELIC for both names
    chamadas = []
    assert rodada.rodar(_entrada([FATO, SELIC]), arquivos, _servicos(_deepseek({}, chamadas=chamadas))).chamadas == 0
    log = registro.ler_log(arquivos.log)
    assert list(log["etapa"]) == ["semeado"] and str(log["itens"].iloc[0]) == "3"
