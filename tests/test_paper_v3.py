"""Paper-v3 (scripts/paper_v3.py, docs/superpowers/specs/2026-10-28-paper-v3-composto-design.md): the composite of
attempt 51 and a same-day Conservadora, decided as the engine decides and recorded on the same close."""

import pandas as pd
import pytest

from scripts import avaliar_valor
from scripts.avaliar_conservadora import carregar_mensal
from scripts.qpo_montar import FOLDER
from scripts.qubo_universo_avaliar import PASTA_CVM, _value_score


def test_the_composite_score_is_attempt_51s(monkeypatch):
    from scripts.paper_v3 import nota_do_composto
    monkeypatch.setattr(avaliar_valor, "PASTA_UNIVERSO", avaliar_valor.PASTA_UNIVERSO)    # _value_score sets it
    mensal = carregar_mensal(FOLDER)
    esperada, _ = _value_score(mensal, com_crescimento=True)
    balancos, _ = avaliar_valor.carregar_balancos(PASTA_CVM)
    nota = nota_do_composto(mensal.retornos, mensal.proventos, balancos)
    universo = pd.read_parquet(FOLDER / "universe_monthly.parquet", columns=["month", "security"])
    for mes in (pd.Timestamp("2016-04-01"), pd.Timestamp("2026-10-01")):
        candidatos = list(universo.loc[pd.to_datetime(universo["month"]) == mes, "security"])
        pd.testing.assert_series_equal(nota(mes, candidatos), esperada(mes, candidatos))
        assert nota(mes, candidatos).notna().sum() > 50


@pytest.fixture(scope="module")
def dados_de_2012():
    """The final proof's frozen files and the CVM folder of attempt 51, cut so the engine makes 2 decisions
    (02/01 and 03/04/2012), as paper-v2's test."""
    from scripts.avaliar_robustez import carregar
    from scripts.avaliar_universo_dinamico import CFG_BT
    from scripts.avaliar_valor import PRIMEIRA_DECISAO, corte_dos_precos
    from scripts.paper_v2 import DadosDoLago
    from scripts.paper_v3 import DadosDoComposto
    dados, mensal = carregar(), carregar_mensal()
    corte = corte_dos_precos(dados.precos.index, PRIMEIRA_DECISAO, CFG_BT.janela_treino)
    precos = dados.precos.iloc[corte:corte + CFG_BT.janela_treino + 2 + CFG_BT.janela_teste + 5]
    balancos, _ = avaliar_valor.carregar_balancos(PASTA_CVM)
    lago = DadosDoLago(precos, dados.universo, dados.setores, mensal.retornos, mensal.proventos, {}, {})
    return DadosDoComposto(lago, balancos, {})


def test_the_days_composite_is_the_engines_and_the_control_is_paper_v2s_conservadora(dados_de_2012):
    from dataclasses import replace

    from core.backtest import Congelamento, walk_forward_dinamico
    from core.composto import estrategia_por_nota
    from scripts.avaliar_robustez import _acoes
    from scripts.avaliar_universo_dinamico import CFG_BT
    from scripts.paper_v2 import conservadora_do_dia
    from scripts.paper_v3 import COMPOSTO, CONSERVADORA, carteiras_do_dia
    lago = dados_de_2012.lago
    nota = nota_do_composto_de(dados_de_2012)
    motor = walk_forward_dinamico(_acoes(lago.precos), lago.universo, estrategia_por_nota(nota, lago.setores), CFG_BT,
                                  "motor", Congelamento())
    assert len(motor.pesos_alvo) == 2
    for vale_de, pesos in motor.pesos_alvo.iterrows():
        decisao = lago.precos.index[lago.precos.index.get_loc(vale_de) - 1]
        ate_a_decisao = replace(dados_de_2012, lago=replace(lago, precos=lago.precos.loc[:decisao]))
        carteiras = carteiras_do_dia(ate_a_decisao, decisao)
        assert set(carteiras) == {COMPOSTO, CONSERVADORA}
        pd.testing.assert_series_equal(carteiras[COMPOSTO].sort_index(), pesos[pesos > 0].sort_index(),
                                       check_names=False)
        pd.testing.assert_series_equal(carteiras[CONSERVADORA].sort_index(),
                                       conservadora_do_dia(ate_a_decisao.lago, decisao).sort_index())


def nota_do_composto_de(dados):
    from scripts.paper_v3 import nota_do_composto
    return nota_do_composto(dados.lago.retornos_mensais, dados.lago.proventos, dados.balancos)


def _calendario(tmp_path, datas: pd.DatetimeIndex):
    arquivo = tmp_path / "precos_diarios.parquet"
    pd.DataFrame({"date": datas, "BOVA11": 1.0}).to_parquet(arquivo)
    return arquivo


def _arquivos(tmp_path, datas: pd.DatetimeIndex):
    from scripts.paper_v3 import Arquivos
    return Arquivos(_calendario(tmp_path, datas), tmp_path / "composto.csv", tmp_path / "conservadora.csv",
                    tmp_path / "daily.lock")


def _nunca(*_):
    raise AssertionError("should not be called")


def test_without_the_market_values_of_the_month_of_the_close_it_does_not_decide(tmp_path, dados_de_2012):
    from dataclasses import replace

    from core.paper import Acao
    from scripts.paper_v3 import Passos, executar
    datas = dados_de_2012.lago.precos.index
    sem_o_mes = dados_de_2012.balancos.valor_mercado
    sem_o_mes = sem_o_mes[pd.to_datetime(sem_o_mes["month"]) < datas[-1].to_period("M").to_timestamp()]
    atrasado = replace(dados_de_2012, balancos=replace(dados_de_2012.balancos, valor_mercado=sem_o_mes))
    rodada = executar(_arquivos(tmp_path, datas), Passos(lambda: None, lambda: atrasado, _nunca))
    assert rodada.acao is Acao.AGUARDAR and rodada.carteiras is None
    assert "market_caps" in rodada.nota
    assert not (tmp_path / "composto.csv").exists()


def test_if_either_arm_is_refused_neither_is_recorded(tmp_path, dados_de_2012, monkeypatch):
    from core.backtest import CarteiraRecusada
    from core.risco import Regra, Violacao
    from scripts import paper_v3
    from scripts.paper_v3 import Passos, executar

    def recusa(*_):
        raise CarteiraRecusada([Violacao(Regra.VOL, "carteira", 0.45, 0.40)])

    monkeypatch.setattr(paper_v3, "carteiras_do_dia", recusa)
    rodada = executar(_arquivos(tmp_path, dados_de_2012.lago.precos.index),
                      Passos(lambda: None, lambda: dados_de_2012, _nunca))
    assert rodada.desfecho == "recusada" and "vol ex-ante" in rodada.nota
    assert not (tmp_path / "composto.csv").exists() and not (tmp_path / "conservadora.csv").exists()


def test_on_the_due_day_both_arms_are_recorded_on_the_same_close(tmp_path, dados_de_2012):
    from scripts.paper_trading import ler_registro
    from scripts.paper_v3 import COMPOSTO, CONSERVADORA, Passos, executar
    guardadas = []
    datas = dados_de_2012.lago.precos.index
    rodada = executar(_arquivos(tmp_path, datas),
                      Passos(lambda: None, lambda: dados_de_2012, lambda feita, _: guardadas.append(feita)))
    assert guardadas == [rodada] and rodada.carteiras is not None
    for braco, arquivo in ((COMPOSTO, "composto.csv"), (CONSERVADORA, "conservadora.csv")):
        registro = ler_registro(tmp_path / arquivo)
        assert set(registro["data"]) == {datas[-1]}
        assert registro.set_index("ticker")["peso"].to_dict() == pytest.approx(rodada.carteiras[braco].to_dict())
    again = executar(_arquivos(tmp_path, datas), Passos(_nunca, _nunca, _nunca))
    assert again.nota == "close already decided"


def test_a_failed_lake_update_does_not_decide(tmp_path, dados_de_2012):
    import subprocess

    from core.paper import Acao
    from scripts.paper_v3 import Passos, executar

    def falha():
        raise subprocess.CalledProcessError(1, ["b3lake.cli", "fundamentals"])

    rodada = executar(_arquivos(tmp_path, dados_de_2012.lago.precos.index), Passos(falha, _nunca, _nunca))
    assert rodada.acao is Acao.AGUARDAR and rodada.nota.startswith("lake update failed: CalledProcessError")


def test_the_reading_refaz_each_arm_as_paper_v2_does_and_compares_the_composite_with_the_control():
    from scripts import paper_v2
    from scripts.paper_v3 import COMPOSTO, CONSERVADORA, P_CONTRA_A_CONSERVADORA, leitura
    from tests.test_paper_v2 import _leitura_sintetica
    registro, dados, dividendos, liquidez = _leitura_sintetica()
    controle = registro.assign(ticker=["BBBB3", "AAAA3"], peso=[0.8, 0.2])
    medidas = leitura({COMPOSTO: registro, CONSERVADORA: controle}, dados.precos, dividendos, liquidez)
    assert list(medidas.index) == [COMPOSTO, CONSERVADORA, "BOVA11"]
    for braco, sozinho in ((COMPOSTO, registro), (CONSERVADORA, controle)):
        do_v2 = paper_v2.leitura(sozinho, dados, dividendos, liquidez)
        assert medidas.loc[braco, "Saída final (R$)"] == pytest.approx(do_v2.iloc[0]["Saída final (R$)"])
    assert 0.0 < medidas.loc[COMPOSTO, P_CONTRA_A_CONSERVADORA] < 1.0


def test_the_decision_asks_for_a_composite_final_exit_value_at_least_the_conservadoras():
    from scripts.paper_v3 import COMPOSTO, CONSERVADORA, criterio
    tabela = pd.DataFrame({"Saída final (R$)": [11_000.0, 11_000.0, 15_000.0], "CAGR": [0.05, 0.30, 0.9]},
                          index=[COMPOSTO, CONSERVADORA, "BOVA11"])
    assert criterio(tabela)                                       # a tie passes; CAGR and BOVA11 do not enter
    assert not criterio(tabela.assign(**{"Saída final (R$)": [10_999.0, 11_000.0, 1.0]}))


def test_the_live_cvm_loader_reads_what_the_research_loader_reads_and_hashes_both_files():
    import hashlib

    from scripts.paper_v3 import ARQUIVOS_DA_CVM, ler_balancos
    esperados, _ = avaliar_valor.carregar_balancos(PASTA_CVM)
    balancos, sha256 = ler_balancos(PASTA_CVM, avaliar_valor.PASTA_UNIVERSO / "sectors.parquet")
    pd.testing.assert_frame_equal(balancos.fundamentos, esperados.fundamentos)
    pd.testing.assert_frame_equal(balancos.valor_mercado, esperados.valor_mercado)
    assert dict(balancos.setores) == dict(esperados.setores)
    assert sha256 == {nome: hashlib.sha256((PASTA_CVM / nome).read_bytes()).hexdigest() for nome in ARQUIVOS_DA_CVM}


def test_the_lake_update_runs_universe_then_fundamentals_with_the_lakes_python(tmp_path):
    from scripts.paper_v3 import atualizar_lago
    chamadas = []
    atualizar_lago(tmp_path, lambda comando, **opcoes: chamadas.append((comando, opcoes["cwd"], opcoes["check"])))
    python = str(tmp_path / ".venv" / "Scripts" / "python.exe")
    assert chamadas == [([python, "-m", "b3lake.cli", "universe"], tmp_path, True),
                        ([python, "-m", "b3lake.cli", "fundamentals"], tmp_path, True)]
