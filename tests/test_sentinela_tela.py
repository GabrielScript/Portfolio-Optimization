import numpy as np
import pandas as pd

from core.sentinela import conta, registro, tela
from core.sentinela.rodada import Arquivos

DIAS = pd.bdate_range("2026-11-06", periods=6)


def _log(arquivos: Arquivos) -> None:
    registro.acrescentar_log(arquivos.log, [
        {"hora": DIAS[1] + pd.Timedelta(hours=9), "ticker": "PETR4", "etapa": "varredura", "veredito": "grave",
         "texto": "Pedido de recuperação judicial.", "citacoes": '[{"id": "cvm-a", "trecho": "recuperação"}]',
         "custo_usd": 0.001},
        {"hora": DIAS[1] + pd.Timedelta(hours=9), "ticker": "VALE3", "etapa": "varredura", "veredito": "sem_evento",
         "descarte": "o número 5 do resumo não está nas citações", "custo_usd": 0.0005},
        {"hora": DIAS[1] + pd.Timedelta(hours=10), "ticker": "VALE3", "etapa": "varredura", "veredito": "atencao",
         "texto": "Volume alto.", "custo_usd": 0.0005},
        {"hora": DIAS[1] + pd.Timedelta(hours=9), "ticker": "PETR4", "etapa": "decisao", "acao": "vender",
         "confianca": 0.8, "custo_usd": 0.01}])


def test_panel_before_any_run_is_none(tmp_path):
    assert tela.painel(Arquivos(tmp_path), pd.DataFrame(), pd.Series(dtype=float)) is None


def test_panel_shows_alerts_newest_first_actions_costs_and_the_discard_share(tmp_path):
    arquivos = Arquivos(tmp_path)
    _log(arquivos)
    registro.acrescentar_acoes(arquivos.acoes, [
        {"id": "a1", "decidida_em": DIAS[1], "ticker": "PETR4", "acao": "vender", "confianca": 0.8,
         "executa_em": DIAS[1], "estado": "aceita"},
        {"id": "a1", "decidida_em": DIAS[1], "ticker": "PETR4", "acao": "vender", "confianca": 0.8,
         "executa_em": DIAS[1], "estado": "executada"}])
    painel = tela.painel(arquivos, pd.DataFrame(), pd.Series(dtype=float))
    assert list(painel.alertas["Ação"]) == ["VALE3", "PETR4"]
    assert list(painel.alertas["Veredito"]) == ["atenção", "grave"]
    assert list(painel.acoes["Situação"]) == ["executada"]
    assert painel.chamadas == 4 and abs(painel.custo_usd - 0.012) < 1e-9
    assert abs(painel.descartadas - 1 / 3) < 1e-9
    assert painel.curvas is None and painel.ultima_rodada == DIAS[1] + pd.Timedelta(hours=10)


def test_panel_has_both_curves_once_the_account_traded(tmp_path):
    arquivos = Arquivos(tmp_path)
    _log(arquivos)
    precos = pd.DataFrame({"PETR4": np.linspace(30, 28, 6), "VALE3": np.full(6, 60.0)}, index=DIAS)
    registro_composta = pd.DataFrame({"data": DIAS[0], "ticker": ["PETR4", "VALE3"], "peso": [0.45, 0.45]})
    ordens, _ = conta.ordens_devidas(conta.vazio(), registro_composta, [], precos, pd.Series(0.0, index=DIAS),
                                     lambda ticker, valor: 0.0, inicio=DIAS[0])
    conta.acrescentar(arquivos.ordens, ordens)
    painel = tela.painel(arquivos, precos, pd.Series(0.0, index=DIAS))
    assert set(painel.curvas.columns) == {conta.BASE, conta.CONTA, "Selic"}


def test_a_first_answer_asked_again_counts_as_a_call_and_a_cost_but_not_as_a_scan(tmp_path):
    # amendment 8: the discard share keeps counting final answers; the calls and the cost count both attempts
    arquivos = Arquivos(tmp_path)
    _log(arquivos)
    registro.acrescentar_log(arquivos.log, [
        {"hora": DIAS[2] + pd.Timedelta(hours=9), "ticker": "BBSE3", "etapa": "varredura_descartada",
         "descarte": "o número 2026 do resumo não está nas citações", "custo_usd": 0.0004},
        {"hora": DIAS[2] + pd.Timedelta(hours=9), "ticker": "BBSE3", "etapa": "varredura", "veredito": "atencao",
         "texto": "Guidance revisado.", "custo_usd": 0.0004}])
    painel = tela.painel(arquivos, pd.DataFrame(), pd.Series(dtype=float))
    assert painel.chamadas == 6 and abs(painel.custo_usd - 0.0128) < 1e-9
    assert abs(painel.descartadas - 1 / 4) < 1e-9 and list(painel.alertas["Ação"])[0] == "BBSE3"
