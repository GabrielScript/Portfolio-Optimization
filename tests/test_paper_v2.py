"""Paper-v2 da Conservadora (scripts/paper_v2.py): a decisão do dia é a do motor, e a leitura refaz o registro."""

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from core.backtest import Congelamento, retornos_da_carteira, walk_forward_dinamico
from core.execucao import refazer
from core.paper import Acao, curva_do_registro
from scripts.avaliar_conservadora import CONSERVADORA, carregar_mensal
from scripts.avaliar_execucao import liquidez_da_tabela
from scripts.avaliar_robustez import _acoes, carregar
from scripts.avaliar_universo_dinamico import CFG_BT
from scripts.avaliar_valor import PRIMEIRA_DECISAO, corte_dos_precos
from scripts.paper_v2 import (DadosDoLago, acao_do_dia, conservadora_do_dia, estrategia, executar, leitura,
                              mes_da_carteira, resultado_do_registro, treino_do_dia)


def test_o_mes_da_carteira_e_o_do_pregao_seguinte():
    assert mes_da_carteira(pd.Timestamp("2026-10-19")) == pd.Timestamp("2026-10-01")
    assert mes_da_carteira(pd.Timestamp("2026-10-30")) == pd.Timestamp("2026-11-01")      # sexta, fim do mês
    assert mes_da_carteira(pd.Timestamp("2026-12-30")) == pd.Timestamp("2027-01-01")      # 31/12 sem pregão


def test_perda_de_5_por_cento_no_ultimo_pregao_congela_e_fora_do_dia_devido_aguarda():
    curva = pd.Series([1.0, 1.01, 0.95], index=pd.bdate_range("2026-10-15", periods=3))
    assert acao_do_dia(curva, devido=True) is Acao.CONGELADO
    assert acao_do_dia(curva.iloc[:2], devido=True) is Acao.REBALANCEAR
    assert acao_do_dia(curva.iloc[:2], devido=False) is Acao.AGUARDAR
    assert acao_do_dia(pd.Series(dtype=float), devido=True) is Acao.REBALANCEAR


@pytest.fixture(scope="module")
def dados_de_2012():
    """Os arquivos fixados da prova final, cortados para o motor fazer 2 decisões (02/01 e 03/04/2012)."""
    dados, mensal = carregar(), carregar_mensal()
    corte = corte_dos_precos(dados.precos.index, PRIMEIRA_DECISAO, CFG_BT.janela_treino)
    precos = dados.precos.iloc[corte:corte + CFG_BT.janela_treino + 2 + CFG_BT.janela_teste + 5]
    return DadosDoLago(precos, dados.universo, dados.setores, mensal.retornos, mensal.proventos, {}, {})


def test_a_carteira_do_dia_e_a_do_motor_nas_mesmas_datas(dados_de_2012):
    dados = dados_de_2012
    treinos = []

    def espiar(treino: pd.DataFrame) -> pd.Series:
        treinos.append(treino)
        return estrategia(dados)(treino)

    motor = walk_forward_dinamico(_acoes(dados.precos), dados.universo, espiar, CFG_BT, "motor", Congelamento())
    assert len(motor.pesos_alvo) == 2
    for (vale_de, pesos), treino_do_motor in zip(motor.pesos_alvo.iterrows(), treinos):
        decisao = dados.precos.index[dados.precos.index.get_loc(vale_de) - 1]
        ate_a_decisao = replace(dados, precos=dados.precos.loc[:decisao])
        pd.testing.assert_frame_equal(treino_do_dia(ate_a_decisao, decisao), treino_do_motor)
        do_dia = conservadora_do_dia(ate_a_decisao, decisao)
        pd.testing.assert_series_equal(do_dia.sort_index(), pesos[pesos > 0].sort_index(), check_names=False)


def test_o_registro_refeito_sem_custo_e_a_curva_do_registro():
    datas = pd.bdate_range("2026-10-19", periods=80)
    gerador = np.random.default_rng(2)
    precos = pd.DataFrame(100 * np.cumprod(1 + gerador.normal(0.0005, 0.02, (80, 3)), axis=0), index=datas,
                          columns=["AAAA3", "BBBB3", "CCCC3"])
    registro = pd.DataFrame({"data": [datas[0]] * 2 + [datas[63]] * 2, "ticker": ["AAAA3", "BBBB3", "BBBB3", "CCCC3"],
                             "peso": [0.5, 0.5, 0.4, 0.6], "preco": 1.0})
    resultado = resultado_do_registro(registro, precos)
    assert list(resultado.pesos_alvo.index) == [datas[1], datas[64]]
    refeita = refazer(resultado, retornos_da_carteira(precos), capital=1.0, com_ir=False)
    pd.testing.assert_series_equal(refeita.valores, curva_do_registro(registro, precos), check_names=False,
                                   check_freq=False, rtol=1e-12)
    assert resultado.valores.index[0] == datas[0]


def test_no_ultimo_pregao_do_mes_o_treino_usa_o_universo_do_mes_seguinte():
    datas = pd.bdate_range(end="2026-10-30", periods=1_262)
    precos = pd.DataFrame(100.0 + np.arange(1_262)[:, None] * [0.1, 0.2, 0.3], index=datas, columns=["A", "B", "C"])
    universo = pd.DataFrame({"month": pd.to_datetime(["2026-10-01", "2026-10-01", "2026-11-01", "2026-11-01"]),
                             "security": ["A", "B", "C", "A"]})
    dados = DadosDoLago(precos, universo, {}, pd.DataFrame(), pd.DataFrame(), {}, {})
    assert list(treino_do_dia(dados, datas[-1]).columns) == ["C", "A"]
    assert list(treino_do_dia(dados, datas[-2]).columns) == ["A", "B"]
    assert len(treino_do_dia(dados, datas[-1])) == 1_260


def _calendario(tmp_path, datas: pd.DatetimeIndex):
    arquivo = tmp_path / "precos_diarios.parquet"
    pd.DataFrame({"date": datas, "BOVA11": 1.0}).to_parquet(arquivo)
    return arquivo


def _nunca():
    raise AssertionError("não devia chamar")


def test_com_o_lago_rodando_sai_sem_decidir(tmp_path):
    trava = tmp_path / "daily.lock"
    trava.touch()
    calendario = _calendario(tmp_path, pd.bdate_range("2026-01-01", periods=10))
    assert executar(calendario, tmp_path / "carteiras.csv", trava, _nunca, _nunca, _nunca) is None


def test_fora_do_dia_devido_nem_atualiza_o_lago(tmp_path):
    datas = pd.bdate_range("2026-01-01", periods=70)
    registro = tmp_path / "carteiras.csv"
    pd.DataFrame({"data": [datas[10].date()], "ticker": ["AAAA3"], "peso": [1.0], "preco": [1.0]}).to_csv(
        registro, index=False)
    rodada = executar(_calendario(tmp_path, datas), registro, tmp_path / "daily.lock", _nunca, _nunca, _nunca)
    assert rodada.acao is Acao.AGUARDAR and rodada.carteira is None


def test_no_dia_devido_com_o_universo_atrasado_nao_decide(tmp_path, dados_de_2012):
    datas = dados_de_2012.precos.index
    atualizados = []
    rodada = executar(_calendario(tmp_path, datas), tmp_path / "carteiras.csv", tmp_path / "daily.lock",
                      lambda: atualizados.append(True),
                      lambda: replace(dados_de_2012, precos=dados_de_2012.precos.iloc[:-1]), _nunca)
    assert atualizados == [True]
    assert rodada.carteira is None and "sem decidir" in rodada.nota
    assert not (tmp_path / "carteiras.csv").exists()


def test_no_dia_devido_registra_a_carteira_do_motor_e_guarda(tmp_path, dados_de_2012):
    decisao = dados_de_2012.precos.index[CFG_BT.janela_treino]      # o fechamento da 1ª decisão do motor
    dados = replace(dados_de_2012, precos=dados_de_2012.precos.loc[:decisao])
    guardadas = []
    rodada = executar(_calendario(tmp_path, dados.precos.index), tmp_path / "carteiras.csv", tmp_path / "daily.lock",
                      lambda: None, lambda: dados, lambda feita, lidos: guardadas.append(feita))
    assert rodada.acao is Acao.REBALANCEAR and guardadas == [rodada]
    registro = pd.read_csv(tmp_path / "carteiras.csv", parse_dates=["data"])
    assert set(registro["data"]) == {decisao}
    pd.testing.assert_series_equal(registro.set_index("ticker")["peso"].sort_index(),
                                   conservadora_do_dia(dados, decisao).sort_index(), check_names=False)


def test_a_leitura_refaz_o_registro_e_o_bova11_nos_valores_de_saida():
    datas = pd.bdate_range("2026-10-19", periods=130)
    gerador = np.random.default_rng(4)
    precos = pd.DataFrame(100 * np.cumprod(1 + gerador.normal(0.0005, 0.015, (130, 3)), axis=0), index=datas,
                          columns=["AAAA3", "BBBB3", "BOVA11"])
    registro = pd.DataFrame({"data": [datas[0]] * 2, "ticker": ["AAAA3", "BBBB3"], "peso": [0.5, 0.5], "preco": 1.0})
    meses = pd.date_range("2026-09-01", "2027-05-01", freq="MS")
    tabela = pd.DataFrame([(mes.date(), papel, 0.001, 0.015, 1e8) for mes in meses for papel in precos.columns],
                          columns=["month", "security", "meio_spread", "vol_diaria", "volume_mediano"])
    dividendos = pd.DataFrame({"security": ["AAAA3"], "ex_date": [datas[50].date()], "corporate_action": ["DIVIDENDO"],
                               "yield": [0.01]})
    dados = DadosDoLago(precos, pd.DataFrame(), {}, pd.DataFrame(), pd.DataFrame(), {}, {})
    medidas = leitura(registro, dados, dividendos, liquidez_da_tabela(tabela))
    assert list(medidas.index) == [CONSERVADORA, "BOVA11"]
    assert 0.0 < medidas.loc[CONSERVADORA, "P aritmético"] < 1.0
    assert medidas.loc["BOVA11", "IR na saída (R$)"] > 0.0 or precos["BOVA11"].iloc[-1] <= precos["BOVA11"].iloc[0]


def test_atualizar_universo_chama_o_python_do_venv_do_lago_e_nao_o_uv(monkeypatch, tmp_path):
    # a tarefa agendada roda com o PATH do sistema, onde o uv (em ~/.local/bin) não está
    import subprocess

    from scripts.paper_v2 import atualizar_universo
    chamadas = []
    monkeypatch.setattr(subprocess, "run", lambda comando, **opcoes: chamadas.append((comando, opcoes["cwd"])))
    atualizar_universo(tmp_path)
    assert chamadas == [([str(tmp_path / ".venv" / "Scripts" / "python.exe"), "-m", "b3lake.cli", "universe"],
                         tmp_path)]
