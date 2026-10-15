"""App inteiro via AppTest, SEM rede: os carregadores de ``ui.servicos`` viram dados sintéticos."""

from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from assets import BENCHMARK_ACOES, get_tickers_by_setor
from core import market_data
from core.config import Objetivo
from ui import servicos

APP = Path(__file__).resolve().parent.parent / "app.py"
# 8 ações: viável com teto de 20% por ativo, e a reamostragem roda em segundos.
SETOR_PEQUENO = "Tecnologia da Informação"
TIMEOUT_S = 300
CARREGAR_COTACOES_REAL = servicos.carregar_cotacoes      # guardada antes de a fixture trocá-la


def _precos_sinteticos(tickers: tuple[str, ...], anos: int, dia: date) -> market_data.DadosPrecos:
    gerador = np.random.default_rng(7)
    datas = pd.bdate_range(end=pd.Timestamp(dia), periods=650)     # 504 de treino + 3 rebalanceamentos
    retornos = gerador.normal(0.0006, 0.015, size=(len(datas), len(tickers)))
    # Perto de R$ 10, como boa parte das ações da B3: 2 unidades cabem no teto mesmo com R$ 500 a investir.
    precos = pd.DataFrame(10 * np.cumprod(1 + retornos, axis=0), index=datas, columns=list(tickers))
    return market_data.DadosPrecos(precos)


def _cdi_sintetico(anos: int, dia: date) -> pd.Series:
    return pd.Series(0.0005, index=pd.bdate_range(end=pd.Timestamp(dia), periods=900))


def _growth_sintetico(tickers, precos, setores, usar_cvm, dia) -> pd.DataFrame:
    scores = np.linspace(20, 90, len(tickers))
    return pd.DataFrame({"score": scores, "queima_caixa": False}, index=list(tickers))


def _cotacoes_sinteticas(tickers: tuple[str, ...], precos_recentes: pd.DataFrame) -> pd.DataFrame:
    """Último fechamento como 'negócio do dia', sem rede."""
    ultimos = precos_recentes.ffill().iloc[-1]
    return pd.DataFrame({"preco": ultimos.reindex(list(tickers)),
                         "horario": pd.Timestamp(precos_recentes.index[-1]),
                         "fonte": market_data.FONTE_INTRADIARIA}, index=list(tickers))


@pytest.fixture
def app(monkeypatch) -> AppTest:
    monkeypatch.setattr(servicos, "carregar_precos", _precos_sinteticos)
    monkeypatch.setattr(servicos, "carregar_cdi", _cdi_sintetico)
    monkeypatch.setattr(servicos, "carregar_selic", lambda dia: market_data.TaxaSelic(0.1375, "sintética"))
    monkeypatch.setattr(servicos, "carregar_growth", _growth_sintetico)
    monkeypatch.setattr(servicos, "carregar_cotacoes", _cotacoes_sinteticas)
    at = AppTest.from_file(str(APP), default_timeout=TIMEOUT_S)
    at.run()
    at.sidebar.multiselect[0].set_value([SETOR_PEQUENO]).run()
    return at


def _tabela_walk_forward(at: AppTest) -> pd.DataFrame:
    tabelas = [df.value for df in at.dataframe if "Métrica" in df.value.columns]
    assert len(tabelas) == 1
    return tabelas[0]


def test_reamostragem_vem_desligada_e_o_walk_forward_tem_4_estrategias(app):
    assert not app.exception
    assert app.sidebar.checkbox(key="reamostrar").value is False
    colunas = list(_tabela_walk_forward(app).columns)
    assert colunas[1].startswith("Otimizada")
    assert len(colunas) == 1 + 4 + 1                      # Métrica + 4 estratégias + BOVA11
    assert not [e for e in app.expander if e.label == "Estabilidade nas reamostras"]


def test_app_nao_mostra_nada_de_cripto(app):
    rotulos = [w.label for w in app.sidebar.checkbox] + [w.label for w in app.sidebar.slider]
    textos = rotulos + [m.value for m in app.markdown] + [c.value for c in app.caption]
    assert not [t for t in textos if "cripto" in t.lower() or "btc" in t.lower()]


def test_reamostragem_ligada_vira_a_principal_na_carteira_e_no_walk_forward(app):
    app.sidebar.checkbox(key="reamostrar").check().run()

    assert not app.exception
    tabela = _tabela_walk_forward(app)
    assert list(tabela.columns)[1].startswith("Reamostrada")
    assert len(tabela.columns) == 1 + 5 + 1
    assert {"PSR (Sharpe > 0)", "DSR (deflacionado)"} <= set(tabela["Métrica"])
    estabilidade = [e for e in app.expander if e.label == "Estabilidade nas reamostras"]
    assert len(estabilidade) == 1
    assert "% das reamostras em que aparece" in estabilidade[0].dataframe[0].value.columns


def _plano_de_compra(at: AppTest) -> pd.DataFrame:
    tabelas = [df.value for df in at.dataframe if "Qtd." in df.value.columns]
    assert len(tabelas) == 1
    return tabelas[0]


def test_carteira_mostra_cotacao_e_quantidade_inteira_com_minimo_de_2_acoes(app):
    assert not app.exception
    plano = _plano_de_compra(app)
    assert {"Cotação (R$)", "Qtd.", "Lote padrão", "Fracionário", "Peso efetivo"} <= set(plano.columns)
    assert ((plano["Qtd."] == 0) | (plano["Qtd."] >= 2)).all()
    assert (plano["Lote padrão"] + plano["Fracionário"] == plano["Qtd."]).all()
    assert {"Investido", "Custo estimado", "Sobra em caixa"} <= {m.label for m in app.metric}


def test_aceita_valor_a_investir_abaixo_de_mil_reais(app):
    app.sidebar.number_input(key="orcamento").set_value(500.0).run()

    assert not app.exception
    assert app.sidebar.number_input(key="orcamento").value == 500.0
    sobra = next(m for m in app.metric if m.label == "Sobra em caixa")
    investido = next(m for m in app.metric if m.label == "Investido")
    assert sobra.value.startswith("R$") and investido.value.startswith("R$")


def _precos_com_a_ultima_acao_cara(tickers: tuple[str, ...], anos: int, dia: date) -> market_data.DadosPrecos:
    """A última ação do universo custa 1.000 vezes mais: 2 unidades passam de qualquer teto da sidebar."""
    precos = _precos_sinteticos(tickers, anos, dia).precos
    acoes = [t for t in tickers if t != BENCHMARK_ACOES]
    precos[acoes[-1]] *= 1_000
    return market_data.DadosPrecos(precos)


def test_acao_cara_demais_para_o_teto_sai_antes_da_otimizacao_com_aviso(app, monkeypatch):
    monkeypatch.setattr(servicos, "carregar_precos", _precos_com_a_ultima_acao_cara)
    # HRP dá peso a todas: sem o filtro, a ação cara entraria no plano (com 0 ações e o peso em caixa).
    app.sidebar.selectbox[0].set_value(Objetivo.HRP.value).run()

    assert not app.exception
    cara = get_tickers_by_setor([SETOR_PEQUENO])[-1].replace(".SA", "")
    assert len([i.value for i in app.info if cara in i.value]) == 1
    assert cara not in set(_plano_de_compra(app)["Ticker"])


def _precos_todos_caros(tickers: tuple[str, ...], anos: int, dia: date) -> market_data.DadosPrecos:
    precos = _precos_sinteticos(tickers, anos, dia).precos * 1_000
    return market_data.DadosPrecos(precos)


def test_valor_pequeno_demais_para_o_teto_explica_o_que_aumentar(app, monkeypatch):
    monkeypatch.setattr(servicos, "carregar_precos", _precos_todos_caros)
    app.run()

    assert not app.exception
    erros = [e.value for e in app.error]
    assert len(erros) == 1 and "valor a investir" in erros[0]
    assert not [df for df in app.dataframe if "Qtd." in df.value.columns]


def test_acao_sem_cotacao_sai_do_plano_com_aviso_e_o_peso_dela_vai_para_as_outras(app, monkeypatch):
    pedidos = []

    def cotacoes_sem_a_primeira(tickers, precos_recentes):
        pedidos.append(tickers)
        cotacoes = _cotacoes_sinteticas(tickers, precos_recentes)
        cotacoes.loc[tickers[0], "preco"] = float("nan")
        return cotacoes

    monkeypatch.setattr(servicos, "carregar_cotacoes", cotacoes_sem_a_primeira)
    app.run()

    assert not app.exception
    assert not [e.value for e in app.error]
    tickers = [t.replace(".SA", "") for t in pedidos[-1]]
    avisos = [w.value for w in app.warning if "Sem cotação" in w.value]
    assert len(avisos) == 1 and tickers[0] in avisos[0]
    plano = _plano_de_compra(app)
    assert set(plano["Ticker"]) == set(tickers[1:])
    assert plano["Peso alvo"].sum() == pytest.approx(1.0)        # reescalados: a proporção entre elas se mantém


@pytest.fixture
def yahoo_contador(app, monkeypatch) -> list[tuple[str, ...]]:
    """Cache verdadeiro de ``carregar_cotacoes``; só o Yahoo é falso: a n-ésima busca devolve R$ 10 × n."""
    buscas = []

    def cotacoes_atuais(tickers, precos_diarios):
        buscas.append(tuple(tickers))
        return pd.DataFrame({"preco": 10.0 * len(buscas), "horario": pd.Timestamp(date.today()),
                             "fonte": market_data.FONTE_INTRADIARIA}, index=list(tickers))

    monkeypatch.setattr(market_data, "cotacoes_atuais", cotacoes_atuais)
    monkeypatch.setattr(servicos, "carregar_cotacoes", CARREGAR_COTACOES_REAL)
    CARREGAR_COTACOES_REAL.clear()
    yield buscas
    CARREGAR_COTACOES_REAL.clear()


def test_atualizar_cotacoes_busca_de_novo_so_as_desta_carteira(app, yahoo_contador):
    outra_carteira = (("OUTRA.SA",), pd.DataFrame({"OUTRA.SA": [5.0]}))
    CARREGAR_COTACOES_REAL(*outra_carteira)              # 1ª busca (R$ 10): cotação de outro usuário

    app.run()                                            # 2ª busca: R$ 20
    assert set(_plano_de_compra(app)["Cotação (R$)"]) == {20.0}
    app.run()                                            # rerun: vem do cache
    assert set(_plano_de_compra(app)["Cotação (R$)"]) == {20.0}
    app.button(key="atualizar_cotacoes").click().run()   # 3ª busca: R$ 30

    assert not app.exception
    assert set(_plano_de_compra(app)["Cotação (R$)"]) == {30.0}
    CARREGAR_COTACOES_REAL(*outra_carteira)
    assert yahoo_contador.count(("OUTRA.SA",)) == 1      # a entrada da outra carteira continua no cache


def test_reamostragem_so_vale_para_o_kelly(app):
    app.sidebar.checkbox(key="reamostrar").check().run()
    app.sidebar.selectbox[0].set_value(Objetivo.MIN_VOL.value).run()

    assert not app.exception
    assert app.sidebar.checkbox(key="reamostrar").disabled
    assert list(_tabela_walk_forward(app).columns)[1].startswith("Otimizada")
    assert not [e for e in app.expander if e.label == "Estabilidade nas reamostras"]
