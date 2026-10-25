"""App inteiro via AppTest, SEM rede: os carregadores de ``ui.servicos`` viram dados sintéticos."""

import re
from dataclasses import replace
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

from assets import BENCHMARK_ACOES, BENCHMARKS, get_tickers_by_setor
from core import market_data
from core.config import Objetivo
from ui import servicos

APP = Path(__file__).resolve().parent.parent / "app.py"
# 35 ações em 4 setores: viável com os tetos fixos (20% por ação, 30% por setor), e a reamostragem roda em segundos.
SETOR_PEQUENO = "Tecnologia da Informação"
SETORES_DO_TESTE = [SETOR_PEQUENO, "Telecomunicações", "Petróleo, Gás e Biocombustíveis", "Materiais Básicos"]
TIMEOUT_S = 300
JANELA_ANOS, TESTE_ANOS = 3, 1           # os mínimos dos sliders: walk-forward curto e rápido
CARREGAR_COTACOES_REAL = servicos.carregar_cotacoes      # guardada antes de a fixture trocá-la


def _datas_sinteticas(anos: int, dia: date) -> pd.DatetimeIndex:
    return pd.bdate_range(end=pd.Timestamp(dia), periods=anos * 252 + 10)


def _precos_sinteticos(tickers: tuple[str, ...], anos: int, dia: date, versao_dados: int) -> market_data.DadosPrecos:
    gerador = np.random.default_rng(7)
    datas = _datas_sinteticas(anos, dia)
    retornos = gerador.normal(0.0006, 0.015, size=(len(datas), len(tickers)))
    # Perto de R$ 10, como boa parte das ações da B3: 2 unidades cabem no teto mesmo com R$ 500 a investir.
    precos = pd.DataFrame(10 * np.cumprod(1 + retornos, axis=0), index=datas, columns=list(tickers))
    return market_data.DadosPrecos(precos)


def _cdi_sintetico(anos: int, dia: date) -> pd.Series:
    return pd.Series(0.0005, index=pd.bdate_range(end=pd.Timestamp(dia), periods=900))


def _cotacoes_sinteticas(tickers: tuple[str, ...], precos_recentes: pd.DataFrame) -> pd.DataFrame:
    """Último fechamento como 'negócio do dia', sem rede."""
    ultimos = precos_recentes.ffill().iloc[-1]
    return pd.DataFrame({"preco": ultimos.reindex(list(tickers)),
                         "horario": pd.Timestamp(precos_recentes.index[-1]),
                         "fonte": market_data.FONTE_INTRADIARIA}, index=list(tickers))


def _painel_sem_rede(monkeypatch) -> None:
    """The market panel's live loaders (Focus and BCB series) and the lake's headlines on D:, off."""
    monkeypatch.setattr(servicos, "carregar_focus", lambda dia: None)
    monkeypatch.setattr(servicos, "carregar_indicadores_bcb", lambda dia: None)
    monkeypatch.setattr(servicos, "carregar_manchetes", lambda: (None, None))
    monkeypatch.setattr(servicos, "carregar_precos_da_tabela", lambda tickers, dia, versao: _precos_sinteticos(
        tickers, 3, dia, versao).precos)


@pytest.fixture
def app(monkeypatch) -> AppTest:
    monkeypatch.setattr(servicos, "carregar_satelite_registrado", _satelite_sintetico)
    monkeypatch.setattr(servicos, "carregar_precos", _precos_sinteticos)
    monkeypatch.setattr(servicos, "carregar_cdi", _cdi_sintetico)
    monkeypatch.setattr(servicos, "carregar_selic", lambda dia: market_data.TaxaSelic(0.1375, "sintética"))
    monkeypatch.setattr(servicos, "carregar_cotacoes", _cotacoes_sinteticas)
    _painel_sem_rede(monkeypatch)
    at = AppTest.from_file(str(APP), default_timeout=TIMEOUT_S)
    at.run()
    at.sidebar.selectbox(key="objetivo").set_value(Objetivo.KELLY.value)   # os testes antigos são do Kelly
    at.sidebar.segmented_control(key="modo_plano").set_value("estudo")    # e do plano da carteira otimizada
    at.sidebar.multiselect(key="setores").set_value(SETORES_DO_TESTE)
    at.sidebar.slider(key="janela").set_value(JANELA_ANOS)
    at.sidebar.slider(key="anos_teste").set_value(TESTE_ANOS).run()
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
    assert len(colunas) == 1 + 4 + 4                      # Métrica + 4 estratégias + BOVA11, SMAL11, DIVO11, IVVB11
    assert not [e for e in app.expander if e.label == "Estabilidade nas reamostras"]


def test_walk_forward_treina_na_janela_da_carteira_de_hoje(app):
    # O walk-forward baixa janela + teste anos; o 1º rebalanceamento vem depois de janela × 252 pregões
    # (o 1º pregão do download não tem retorno).
    datas = _datas_sinteticas(JANELA_ANOS + TESTE_ANOS, date.today())
    legendas = [c.value for c in app.caption if c.value.startswith("Treino:")]
    assert len(legendas) == 1
    assert f"fora da amostra desde {datas[JANELA_ANOS * 252 + 1]:%d/%m/%Y}" in legendas[0]


def test_bova11_no_walk_forward_comeca_junto_com_as_estrategias(app):
    # Mesmo cfg das estratégias: compra na véspera do 1º rebalanceamento (pregão janela × 252 do download),
    # paga 10 bps uma vez (o padrão da sidebar) e segura até o fim.
    tickers = tuple(get_tickers_by_setor(SETORES_DO_TESTE)) + REFERENCIAS_DA_TELA   # o pedido do walk-forward
    bova = _precos_sinteticos(tickers, JANELA_ANOS + TESTE_ANOS, date.today(), 0).precos[BENCHMARK_ACOES]
    esperado = (1 - 10 / 10_000) * bova.iloc[-1] / bova.iloc[JANELA_ANOS * 252] - 1

    tabela = _tabela_walk_forward(app).set_index("Métrica")

    assert tabela.loc["Retorno total", "BOVA11"] == f"{esperado * 100:.2f}%"


REFERENCIAS_DA_TELA = (*BENCHMARKS, "IVVB11.SA")        # as do app.py (o IVVB11 é só da tela)


def _precos_sem_historico_longo_para_a_ultima(tickers: tuple[str, ...], anos: int, dia: date,
                                              versao_dados: int) -> market_data.DadosPrecos:
    """A última ação do universo não tem janela + teste anos de histórico: some só do download do walk-forward."""
    precos = _precos_sinteticos(tickers, anos, dia, versao_dados).precos
    if anos > JANELA_ANOS:
        precos = precos.drop(columns=[t for t in tickers if t not in REFERENCIAS_DA_TELA][-1])
    return market_data.DadosPrecos(precos)


def test_legenda_do_walk_forward_lista_a_acao_sem_historico_para_ele(app, monkeypatch):
    monkeypatch.setattr(servicos, "carregar_precos", _precos_sem_historico_longo_para_a_ultima)
    app.run()

    assert not app.exception
    curta = get_tickers_by_setor(SETORES_DO_TESTE)[-1].replace(".SA", "")
    legenda = next(c.value for c in app.caption if c.value.startswith("Treino:"))
    assert legenda.endswith(f"Fora do walk-forward por histórico curto: {curta}.")


def _precos_sem_nenhuma_acao_no_historico_longo(tickers: tuple[str, ...], anos: int, dia: date,
                                                versao_dados: int) -> market_data.DadosPrecos:
    """Nenhuma ação tem janela + teste anos de histórico: o download do walk-forward só traz o BOVA11."""
    precos = _precos_sinteticos(tickers, anos, dia, versao_dados).precos
    return market_data.DadosPrecos(precos[[BENCHMARK_ACOES]] if anos > JANELA_ANOS else precos)


def test_walk_forward_sem_acoes_com_historico_suficiente_avisa_em_vez_de_quebrar(app, monkeypatch):
    monkeypatch.setattr(servicos, "carregar_precos", _precos_sem_nenhuma_acao_no_historico_longo)
    app.run()

    assert not app.exception
    assert len([i.value for i in app.info if i.value.startswith("Walk-forward indisponível")]) == 1


def test_app_nao_mostra_nada_de_cripto(app):
    rotulos = [w.label for w in app.sidebar.checkbox] + [w.label for w in app.sidebar.slider]
    textos = rotulos + [m.value for m in app.markdown] + [c.value for c in app.caption]
    assert not [t for t in textos if "cripto" in t.lower() or "btc" in t.lower()]


def test_app_nao_mostra_nada_do_growth(app):
    # A carteira de hoje é a mesma que o walk-forward testa: sem filtro nem tilt por fundamentos de hoje.
    rotulos = [w.label for w in app.sidebar.checkbox] + [w.label for w in app.sidebar.slider]
    textos = (rotulos + [m.value for m in app.markdown] + [c.value for c in app.caption]
              + [i.value for i in app.info] + [e.label for e in app.expander])
    assert not [t for t in textos if "growth" in t.lower()]
    assert not {"Percentil mínimo", "Tilt (× vol)", "Receita 10 anos via CVM (lento)"} & set(rotulos)
    assert not {"Growth Score", "Queima de caixa"} & set(_plano_de_compra(app).columns)


def test_reamostragem_ligada_vira_a_principal_na_carteira_e_no_walk_forward(app):
    app.sidebar.checkbox(key="reamostrar").check().run()

    assert not app.exception
    tabela = _tabela_walk_forward(app)
    assert list(tabela.columns)[1].startswith("Reamostrada")
    assert len(tabela.columns) == 1 + 5 + 4
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


def _precos_com_a_ultima_acao_cara(tickers: tuple[str, ...], anos: int, dia: date,
                                   versao_dados: int) -> market_data.DadosPrecos:
    """A última ação do universo custa 1.000 vezes mais: 2 unidades passam de qualquer teto da sidebar."""
    precos = _precos_sinteticos(tickers, anos, dia, versao_dados).precos
    acoes = [t for t in tickers if t != BENCHMARK_ACOES]
    precos[acoes[-1]] *= 1_000
    return market_data.DadosPrecos(precos)


def test_acao_cara_demais_para_o_teto_sai_antes_da_otimizacao_com_aviso(app, monkeypatch):
    monkeypatch.setattr(servicos, "carregar_precos", _precos_com_a_ultima_acao_cara)
    # HRP dá peso a todas: sem o filtro, a ação cara entraria no plano (com 0 ações e o peso em caixa).
    app.sidebar.selectbox(key="objetivo").set_value(Objetivo.HRP.value).run()

    assert not app.exception
    cara = get_tickers_by_setor(SETORES_DO_TESTE)[-1].replace(".SA", "")
    assert len([i.value for i in app.info if cara in i.value]) == 1
    assert cara not in set(_plano_de_compra(app)["Ticker"])


def _precos_todos_caros(tickers: tuple[str, ...], anos: int, dia: date, versao_dados: int) -> market_data.DadosPrecos:
    precos = _precos_sinteticos(tickers, anos, dia, versao_dados).precos * 1_000
    return market_data.DadosPrecos(precos)


def test_valor_pequeno_demais_para_o_teto_explica_o_que_aumentar(app, monkeypatch):
    monkeypatch.setattr(servicos, "carregar_precos", _precos_todos_caros)
    app.run()

    assert not app.exception
    erros = [e.value for e in app.error]
    assert len(erros) == 1 and "valor a investir" in erros[0]
    assert not [df for df in app.dataframe if "Qtd." in df.value.columns]


def test_acao_sem_cotacao_sai_do_plano_com_aviso_e_o_peso_dela_vai_para_as_outras_ate_o_teto(app, monkeypatch):
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
    # o peso dela vai para as outras só até o teto por ação (20%); o que não couber fica em caixa
    assert plano["Peso alvo"].max() <= 0.20 + 1e-9
    assert plano["Peso alvo"].sum() <= 1.0 + 1e-9


@pytest.fixture
def yahoo_contador(app, monkeypatch) -> list[tuple[str, ...]]:
    """Cache verdadeiro de ``carregar_cotacoes``; só o Yahoo é falso: a n-ésima busca devolve R$ 10 × n."""
    buscas = []

    def cotacoes_atuais(tickers, precos_diarios):
        buscas.append(tuple(tickers))
        # a bar of a session the lake does not have yet (trading now): the official close does not replace it
        return pd.DataFrame({"preco": 10.0 * len(buscas), "horario": precos_diarios.index.max() + pd.Timedelta(days=1),
                             "fonte": market_data.FONTE_INTRADIARIA}, index=list(tickers))

    monkeypatch.setattr(market_data, "cotacoes_atuais", cotacoes_atuais)
    monkeypatch.setattr(servicos, "carregar_cotacoes", CARREGAR_COTACOES_REAL)
    CARREGAR_COTACOES_REAL.clear()
    yield buscas
    CARREGAR_COTACOES_REAL.clear()


def test_atualizar_cotacoes_busca_de_novo_so_as_desta_carteira(app, yahoo_contador):
    outra_carteira = (("OUTRA.SA",), pd.DataFrame({"OUTRA.SA": [5.0]}, index=[pd.Timestamp(date.today())]))
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


def _precos_tecnologia_em_alta(tickers: tuple[str, ...], anos: int, dia: date,
                               versao_dados: int) -> market_data.DadosPrecos:
    """Tecnologia sobe ~0,2% ao dia a mais: sem teto por setor, o otimizador põe tudo nela."""
    precos = _precos_sinteticos(tickers, anos, dia, versao_dados).precos
    tecnologia = [t for t in get_tickers_by_setor([SETOR_PEQUENO]) if t in precos.columns]
    precos[tecnologia] = precos[tecnologia].mul(np.exp(0.002 * np.arange(len(precos))), axis=0)
    return market_data.DadosPrecos(precos)


def _peso_por_setor(plano: pd.DataFrame, coluna: str) -> pd.Series:
    return plano.groupby("Setor")[coluna].sum()


def test_teto_por_setor_limita_o_peso_alvo_de_cada_setor(app, monkeypatch):
    # a tela pode apertar a regra fixa de 30%: com 25%, nem a Tecnologia em alta passa disso
    monkeypatch.setattr(servicos, "carregar_precos", _precos_tecnologia_em_alta)
    app.sidebar.slider(key="teto_setor").set_value(25).run()

    assert not app.exception
    assert (_peso_por_setor(_plano_de_compra(app), "Peso alvo") <= 0.25 + 1e-6).all()


def test_plano_mostra_o_peso_efetivo_de_cada_setor(app):
    assert not app.exception
    por_setor = _peso_por_setor(_plano_de_compra(app), "Peso efetivo")
    maior = por_setor.idxmax()
    legendas = [c.value for c in app.caption if c.value.startswith("Por setor")]
    assert len(legendas) == 1 and f"{maior} {por_setor[maior]:.1%}" in legendas[0]


def test_um_setor_so_com_teto_por_setor_explica_o_que_aumentar(app):
    app.sidebar.multiselect(key="setores").set_value([SETOR_PEQUENO]).run()      # 30% num setor só não soma 100%

    assert not app.exception
    erros = [e.value for e in app.error]
    assert len(erros) == 1 and "teto por setor" in erros[0]


def test_reamostragem_so_vale_para_o_kelly(app):
    app.sidebar.checkbox(key="reamostrar").check().run()
    app.sidebar.selectbox(key="objetivo").set_value(Objetivo.MIN_VOL.value).run()

    assert not app.exception
    assert app.sidebar.checkbox(key="reamostrar").disabled
    assert list(_tabela_walk_forward(app).columns)[1].startswith("Otimizada")
    assert not [e for e in app.expander if e.label == "Estabilidade nas reamostras"]


def test_a_tela_diz_que_os_precos_sao_do_lago_e_ate_que_pregao(app):
    ultimo = _datas_sinteticas(JANELA_ANOS, date.today())[-1]
    fontes = [c.value for c in app.caption if c.value.startswith("Preços:")]
    assert fontes == [f"Preços: lago de dados da B3 (COTAHIST e eventos da B3/CVM, retorno total), até {ultimo:%d/%m/%Y}."]


def test_aviso_de_descartados_explica_os_dois_motivos(app, monkeypatch):
    def precos_com_descartes(tickers, anos, dia, versao_dados):
        dados = _precos_sinteticos(tickers, anos, dia, versao_dados)
        return market_data.DadosPrecos(dados.precos, descartados_sem_dados=["SHOW3.SA"],
                                       descartados_cobertura=["XPTO3.SA"])

    monkeypatch.setattr(servicos, "carregar_precos", precos_com_descartes)
    app.run()
    avisos = [w.value for w in app.warning if "SHOW3" in w.value]
    assert avisos == ["Fora da análise. Sem preços no lago (não negocia mais ou não é ação): SHOW3. "
                      "Negociou em menos de 95% dos pregões da janela (dia sem negócio fica sem preço): XPTO3."]


def test_walk_forward_compara_com_bova11_smal11_divo11_e_ivvb11(app):
    # referências de mercado: Ibovespa, small caps, dividendos e S&P 500 em reais (o alfa segue contra o BOVA11)
    assert list(_tabela_walk_forward(app).columns)[-4:] == ["BOVA11", "SMAL11", "DIVO11", "IVVB11"]


def test_referencia_sem_historico_na_janela_aparece_na_legenda(app, monkeypatch):
    # DIVO11 só existe desde 2012: num walk-forward de 20 anos ela não tem a janela inteira
    def sem_divo11(tickers, anos, dia, versao_dados):
        dados = _precos_sinteticos(tuple(t for t in tickers if t != "DIVO11.SA"), anos, dia, versao_dados)
        return market_data.DadosPrecos(dados.precos, descartados_cobertura=["DIVO11.SA"] if "DIVO11.SA" in tickers else [])

    monkeypatch.setattr(servicos, "carregar_precos", sem_divo11)
    app.run()
    legendas = [c.value for c in app.caption if c.value.startswith("Sem histórico")]
    assert legendas == [f"Sem histórico para os {JANELA_ANOS + TESTE_ANOS} anos do walk-forward: DIVO11."]


def _satelite_sintetico() -> tuple[date, pd.Series]:
    """Um registro do paper com papéis dos 4 setores do teste (Tecnologia com 2 papéis, 30%): soma 90%."""
    por_setor = [get_tickers_by_setor([setor]) for setor in SETORES_DO_TESTE]
    return date(2026, 10, 16), pd.Series({por_setor[0][0]: 0.2, por_setor[0][1]: 0.1, por_setor[1][0]: 0.2,
                                          por_setor[2][0]: 0.2, por_setor[3][0]: 0.2})


def _servicos_sinteticos(monkeypatch) -> None:
    """Every loader of ``ui.servicos`` on synthetic data: registrations, sectors, prices, CDI, Selic and quotes."""
    monkeypatch.setattr(servicos, "carregar_satelite_registrado", _satelite_sintetico)
    monkeypatch.setattr(servicos, "carregar_conservadora_registrada", _conservadora_sintetica)
    monkeypatch.setattr(servicos, "setores_do_lago", _setores_sinteticos)
    monkeypatch.setattr(servicos, "carregar_liquidez", lambda: None)
    monkeypatch.setattr(servicos, "carregar_precos", _precos_sinteticos)
    monkeypatch.setattr(servicos, "carregar_cdi", _cdi_sintetico)
    monkeypatch.setattr(servicos, "carregar_selic", lambda dia: market_data.TaxaSelic(0.1375, "sintética"))
    monkeypatch.setattr(servicos, "carregar_cotacoes", _cotacoes_sinteticas)
    _painel_sem_rede(monkeypatch)


@pytest.fixture
def app_padrao(monkeypatch) -> AppTest:
    """A tela como o usuário abre (sem escolher objetivo), com os 4 setores do teste."""
    _servicos_sinteticos(monkeypatch)
    at = AppTest.from_file(str(APP), default_timeout=TIMEOUT_S)
    at.run()
    at.sidebar.multiselect(key="setores").set_value(SETORES_DO_TESTE)
    at.sidebar.slider(key="janela").set_value(JANELA_ANOS)
    at.sidebar.slider(key="anos_teste").set_value(TESTE_ANOS).run()
    return at


@pytest.fixture
def app_bova_satelite(app_padrao) -> AppTest:
    """The screen with the "BOVA11 + satélite" plan chosen (the default until v1.0, DECISOES 69)."""
    app_padrao.sidebar.segmented_control(key="modo_plano").set_value("dinheiro_real").run()
    return app_padrao


def test_padrao_da_tela_e_o_nucleo_satelite_da_regra_b(app_padrao):
    # S1-P.b (2026-10-18): a regra B aprovou núcleo Mín. Vol 70% + satélite Kelly λ=2 30%
    assert not app_padrao.exception
    assert app_padrao.sidebar.selectbox(key="objetivo").value == "nucleo_satelite"
    assert app_padrao.sidebar.slider(key="fracao_satelite").value == 30
    assert list(_tabela_walk_forward(app_padrao).columns)[1] == "Otimizada (núcleo Mín. Vol + satélite Kelly 30%)"


def test_nenhum_objetivo_da_barra_lateral_se_diz_recomendado(app_padrao):
    # DECISOES 69: a carteira da barra lateral é o estudo (100% ações); o padrão do dinheiro real é o plano BOVA11 +
    # satélite do paper. A regra A (opção 3) segue: o Kelly sozinho não é o principal
    opcoes = app_padrao.sidebar.selectbox(key="objetivo").options
    assert [o for o in opcoes if "recomendado" in o] == []


def test_sliders_nao_afrouxam_as_regras(app):
    assert app.sidebar.slider(key="peso_maximo").max == 20
    assert app.sidebar.slider(key="teto_setor").max == 30
    assert app.sidebar.slider(key="vol_maxima").max == 40


def test_mistura_da_tela_nao_passa_de_50_por_cento_de_satelite(app_padrao):
    assert app_padrao.sidebar.slider(key="fracao_satelite").max == 50


def test_carteira_com_violacao_nao_mostra_plano(app, monkeypatch):
    from core import optimization
    otimizar_real = optimization.otimizar

    def concentrada_em_tres(mu, cov, cfg):
        # 1/3 em cada uma das 3 primeiras (Tecnologia): passa de 20% por ação e de 30% no setor
        pesos = pd.Series(0.0, index=mu.index)
        pesos[mu.index[:3]] = 1 / 3
        return replace(otimizar_real(mu, cov, cfg), pesos=pesos)

    monkeypatch.setattr(optimization, "otimizar", concentrada_em_tres)
    app.run()

    assert not app.exception
    assert len([e.value for e in app.error if "recusada pelas regras de risco" in e.value]) == 1
    assert not [df for df in app.dataframe if "Qtd." in df.value.columns]


def test_hrp_da_tela_respeita_teto_de_setor(app):
    # o HRP espalha por número de papéis: Materiais Básicos (14 de 35) passaria de 30% sem a projeção
    app.sidebar.selectbox(key="objetivo").set_value(Objetivo.HRP.value).run()

    assert not app.exception
    plano = _plano_de_compra(app)
    assert (_peso_por_setor(plano, "Peso alvo") <= 0.30 + 1e-6).all()
    assert (plano["Peso alvo"] <= 0.20 + 1e-6).all()


def test_plano_bova11_satelite_usa_o_bova11_e_o_satelite_do_paper(app_bova_satelite):
    # DECISOES 69: o plano da regra do paper-v1, não a carteira otimizada da barra lateral
    assert not app_bova_satelite.exception
    plano = _plano_de_compra(app_bova_satelite).set_index("Ticker")
    _, satelite = _satelite_sintetico()
    assert plano.index[0] == "BOVA11" and set(plano.index[1:]) <= {t.removesuffix(".SA") for t in satelite.index}
    assert 0.6 <= plano.at["BOVA11", "Valor (R$)"] / 10_000 <= 0.75


def test_titulo_do_plano_bova11_satelite_nao_se_confunde_com_o_padrao_do_estudo(app_bova_satelite):
    # "padrão do estudo" é o objetivo da barra lateral (só ações); este plano é a regra do paper-v1
    titulos = [m.value for m in app_bova_satelite.markdown if m.value.startswith("### Plano de compra")]
    assert titulos == ["### Plano de compra: BOVA11 + satélite (dinheiro real, a regra do paper-v1)"]


def test_bova11_satelite_com_100_reais_mostra_o_bova11_e_o_satelite_em_caixa(app_bova_satelite):
    # R$ 100 (teto de R$ 20 por ação): no estudo a tela para ("Aumente o valor"); aqui, BOVA11 e caixa
    app_bova_satelite.sidebar.number_input(key="orcamento").set_value(100.0).run()
    assert not app_bova_satelite.exception and not [e for e in app_bova_satelite.error if "Aumente" in e.value]
    assert _plano_de_compra(app_bova_satelite)["Ticker"].tolist() == ["BOVA11"]
    assert len([i.value for i in app_bova_satelite.info if i.value.startswith("Satélite em caixa")]) == 1


def test_limitacoes_dizem_que_o_vies_favorece_as_estrategias_e_mostram_o_teste_sem_vies(app_padrao):
    # auditoria App-R4: o viés de sobrevivência favorece as estratégias CONTRA o BOVA11 (o índice não tem o viés)
    textos = " ".join(m.value for m in app_padrao.markdown)
    expanders = [e for e in app_padrao.expander if e.label.startswith("Teste sem viés de sobrevivência")]
    assert "favorece as estratégias contra o BOVA11" in textos and "melhor evidência" not in textos
    assert len(expanders) == 1 and "BOVA11" in expanders[0].dataframe[0].value["Estratégia"].tolist()


def test_plano_do_dinheiro_real_aparece_mesmo_com_o_estudo_invalido(app_padrao):
    # o plano padrão usa só o valor e o custo da barra lateral: um estudo inviável (3 × 20% < 100%) não o esconde
    app_padrao.sidebar.slider(key="n_ativos_max").set_value(3).run()
    assert not app_padrao.exception
    assert [e for e in app_padrao.error if e.value.startswith("Inviável")]
    _, alvo = _conservadora_sintetica()
    assert set(_ordens_alepha(app_padrao).index) == {t.removesuffix(".SA") for t in alvo.index}


def test_carteira_de_hoje_se_diz_estudo_quando_o_plano_e_o_do_dinheiro_real(app_padrao):
    # o gráfico em R$ da carteira otimizada fica logo abaixo de um plano diferente para o mesmo valor
    legendas = [c.value for c in app_padrao.caption if c.value.startswith("Estudo, só ações")]
    assert len(legendas) == 1 and "plano acima" in legendas[0]


# ---------------------------------------------------------------- Alepha: the Conservadora as the default plan

def _conservadora_sintetica() -> tuple[date, pd.Series]:
    """A paper-v2 registration of 5 names at 20% (the per-stock cap), from the test's sectors."""
    por_setor = [get_tickers_by_setor([setor]) for setor in SETORES_DO_TESTE]
    tickers = [por_setor[0][0], por_setor[1][0], por_setor[2][0], por_setor[3][0], por_setor[3][1]]
    return date(2026, 10, 19), pd.Series(0.2, index=tickers)


def _setores_sinteticos(tickers: tuple[str, ...]) -> dict[str, str]:
    return {ticker: f"Setor {posicao}" for posicao, ticker in enumerate(tickers)}


@pytest.fixture
def app_alepha(app_padrao) -> AppTest:
    """The screen as the user opens it: a synthetic Conservadora registration and no liquidity snapshot."""
    return app_padrao


def _ordens_alepha(at: AppTest) -> pd.DataFrame:
    tabelas = [df.value for df in at.dataframe if "Ordem" in df.value.columns]
    assert len(tabelas) == 1
    return tabelas[0].set_index("Ticker")


def test_the_screen_is_alepha_and_its_default_plan_buys_the_registered_conservadora(app_alepha):
    # the user's decisions of 2026-10-23 (DECISOES 117): the Conservadora sleeve by default; named Alepha, no longer
    # "Fundo Alepha" (user, 2026-10-24; DECISOES 126)
    assert not app_alepha.exception
    assert app_alepha.sidebar.segmented_control(key="modo_plano").value == "conservadora"
    assert any(">Alepha<" in m.value for m in app_alepha.markdown if "main-title" in m.value)
    textos = [m.value for m in app_alepha.markdown] + [c.value for c in app_alepha.caption]
    assert not any("Fundo Alepha" in texto for texto in textos)
    assert any("não é um fundo de investimento" in c.value for c in app_alepha.caption)
    _, alvo = _conservadora_sintetica()
    ordens = _ordens_alepha(app_alepha)
    assert set(ordens.index) == {t.removesuffix(".SA") for t in alvo.index}
    assert set(ordens["Ordem"]) == {"compra"}



def test_no_screen_text_call_has_two_bare_dollar_signs():
    # Streamlit renders "$...$" as LaTeX: the live app showed "R$ 10 mil ... R$ 100 mil" and "R$/US$" as math
    # (2026-10-24). Most of these texts only appear on real data, so the source is read: in every st.<text> call
    # of the app and ui/, at most one "$" may be left unescaped
    import ast
    chamadas_de_texto = {"markdown", "caption", "warning", "info", "error", "success", "write"}
    achados = []
    for arquivo in [APP, *sorted((APP.parent / "ui").glob("*.py"))]:
        for no in ast.walk(ast.parse(arquivo.read_text(encoding="utf-8"))):
            if not (isinstance(no, ast.Call) and isinstance(no.func, ast.Attribute)
                    and no.func.attr in chamadas_de_texto and no.args):
                continue
            literais = "".join(c.value for c in ast.walk(no.args[0])
                               if isinstance(c, ast.Constant) and isinstance(c.value, str))
            if len(re.findall(r"(?<!\\)\$", literais)) >= 2:
                achados.append(f"{arquivo.name}:{no.lineno}")
    assert achados == []

def test_from_the_held_positions_the_plan_sells_a_name_outside_the_registration_at_its_average_cost(monkeypatch):
    from ui import alepha
    fora = get_tickers_by_setor([SETORES_DO_TESTE[1]])[1].removesuffix(".SA")
    monkeypatch.setattr(alepha, "editor_de_posicoes", lambda: pd.DataFrame(
        {"ticker": [fora], "quantidade": [50], "preco_medio": [0.01]}))
    _servicos_sinteticos(monkeypatch)
    at = AppTest.from_file(str(APP), default_timeout=TIMEOUT_S)
    at.run()
    assert not at.exception
    venda = _ordens_alepha(at).loc[fora]
    assert (venda["Ordem"], venda["Qtd."]) == ("venda", 50)
    assert venda["Ganho realizado (R$)"] == pytest.approx(venda["Valor (R$)"] - venda["Custo (R$)"] - 50 * 0.01)


def test_below_the_minimum_capital_the_plan_warns_that_it_is_far_from_the_tested_portfolio(app_alepha):
    # 5 names at 20% with quotes near R$ 10: 2 shares each need about R$ 100 per 20%, so R$ 100 in all is not enough
    app_alepha.sidebar.number_input(key="orcamento").set_value(100.0).run()
    assert not app_alepha.exception
    assert len([w for w in app_alepha.warning if "longe da carteira testada" in w.value]) == 1


def test_small_capital_keeps_the_held_bova11_and_buys_stocks_with_the_new_cash(monkeypatch):
    # the user's decision of 2026-10-23 (DECISOES 124): start now with little money; 1 BOVA11 held + R$ 100
    from ui import alepha
    monkeypatch.setattr(alepha, "editor_de_posicoes", lambda: pd.DataFrame(
        {"ticker": ["BOVA11"], "quantidade": [1], "preco_medio": [10.0]}))
    _servicos_sinteticos(monkeypatch)
    at = AppTest.from_file(str(APP), default_timeout=TIMEOUT_S)
    at.run()
    at.sidebar.number_input(key="orcamento").set_value(100.0).run()
    assert not at.exception
    ordens = _ordens_alepha(at)
    assert ordens.loc["BOVA11", "Ordem"] == "manter"
    assert (ordens.drop(index="BOVA11")["Ordem"] == "compra").any()
    assert any("Capital pequeno" in i.value for i in at.info)


def test_a_failure_of_the_information_only_zone_plan_still_shows_the_real_plan(monkeypatch):
    # the zone's redo can leave the cash negative with no buy to shrink (ValueError): only its columns go missing
    from core import rebalanceio
    _, alvo = _conservadora_sintetica()
    _servicos_sinteticos(monkeypatch)
    monkeypatch.setattr(servicos, "carregar_liquidez", lambda: pd.DataFrame({
        "security": [t.removesuffix(".SA") for t in alvo.index], "month": pd.Timestamp("2026-09-01"),
        "meio_spread": 0.001, "vol_diaria": 0.02, "volume_mediano": 1e8}))
    real = rebalanceio.plano_de_rebalanceio

    def zona_falha(*args, zona=None, **kwargs):
        if zona is not None:
            raise ValueError("o caixa e as vendas não pagam os custos (faltam R$ 1.00)")
        return real(*args, **kwargs)
    monkeypatch.setattr(rebalanceio, "plano_de_rebalanceio", zona_falha)
    at = AppTest.from_file(str(APP), default_timeout=TIMEOUT_S)
    at.run()
    assert not at.exception
    assert not [e for e in at.error if e.value.startswith("Plano indisponível")]
    assert set(_ordens_alepha(at).index) == {t.removesuffix(".SA") for t in alvo.index}


def test_all_assets_is_ticked_by_default_and_the_sidebar_says_what_each_group_changes(monkeypatch):
    # the user's request of 2026-10-23: "Todos os ativos" checked by default; the study's selectors do not change
    # the Alepha plan, and the sidebar says so
    _servicos_sinteticos(monkeypatch)
    at = AppTest.from_file(str(APP), default_timeout=TIMEOUT_S)
    at.run()
    assert not at.exception
    assert at.sidebar.checkbox(key="todos").value is True
    textos = [m.value for m in at.sidebar.markdown] + [c.value for c in at.sidebar.caption]
    assert any("não mudam o plano do Alepha" in texto for texto in textos)


def _curvas_sinteticas() -> pd.DataFrame:
    datas = pd.bdate_range("2019-01-01", "2026-10-16")
    crescimento = np.linspace(1.0, 3.0, len(datas))
    return pd.concat([pd.DataFrame({"fase": fase, "date": datas, "Conservadora": crescimento * (1 + fase / 1000),
                                    "BOVA11": np.linspace(1.0, 2.0, len(datas))}) for fase in (0, 21, 42)],
                     ignore_index=True)


def test_the_fundo_alepha_section_shows_the_conservadoras_results_by_period_and_calendar_year(monkeypatch):
    # A2 (user's request of 2026-10-23): the frozen backtest by period (whole, last 5 and 3 years) and by year
    _servicos_sinteticos(monkeypatch)
    monkeypatch.setattr(servicos, "carregar_curvas_conservadora", _curvas_sinteticas)
    at = AppTest.from_file(str(APP), default_timeout=TIMEOUT_S)
    at.run()
    assert not at.exception
    periodos = [df.value for df in at.dataframe if "Período" in df.value.columns]
    assert len(periodos) == 1
    assert list(periodos[0]["Período"]) == ["Período inteiro", "Últimos 5 anos", "Últimos 3 anos"]
    anos = [df.value for df in at.dataframe if "Ano" in df.value.columns]
    assert len(anos) == 1 and anos[0]["Ano"].tolist()[-1] == "2026 (até 16/10)"


def test_the_market_panel_shows_election_timing_news_macro_and_every_stock(monkeypatch):
    # the user's request of 2026-10-23: timing, news and sentiment on politics (elections) and the economy for all
    # stocks, visible in the app; information only
    _servicos_sinteticos(monkeypatch)
    focus = pd.DataFrame({"data": pd.to_datetime(["2026-09-18", "2026-10-16", "2026-10-16"]),
                          "ano_referencia": [2026, 2026, 2027], "mediana": [15.0, 14.75, 12.25], "desvio": 0.1})
    monkeypatch.setattr(servicos, "carregar_focus", lambda dia: {"Selic": focus})
    at = AppTest.from_file(str(APP), default_timeout=TIMEOUT_S)
    at.run()
    assert not at.exception
    assert any("Mercado agora" in m.value for m in at.markdown)
    acoes = [df.value for df in at.dataframe if "P/L" in df.value.columns]
    assert len(acoes) == 1 and len(acoes[0]) >= 100 and "Retorno 12m" in acoes[0].columns
    assert any("Selic" in df.value.index for df in at.dataframe if "Pesquisa" in df.value.columns)
    assert any("nada aqui muda o plano" in c.value for c in at.caption)


def test_small_capital_keeps_only_bova11_and_sells_another_held_etf(monkeypatch):
    # review finding 2: the core kept in small-capital mode is BOVA11, not every ETF of the screen (IVVB11, SMAL11...)
    from ui import alepha
    monkeypatch.setattr(alepha, "editor_de_posicoes", lambda: pd.DataFrame(
        {"ticker": ["BOVA11", "IVVB11"], "quantidade": [1, 1], "preco_medio": [10.0, 10.0]}))
    _servicos_sinteticos(monkeypatch)
    at = AppTest.from_file(str(APP), default_timeout=TIMEOUT_S)
    at.run()
    at.sidebar.number_input(key="orcamento").set_value(50.0).run()
    assert not at.exception
    ordens = _ordens_alepha(at)
    assert (ordens.loc["BOVA11", "Ordem"], ordens.loc["IVVB11", "Ordem"]) == ("manter", "venda")


def test_the_plan_shows_a_light_per_stock_and_flags_one_that_turned_red_without_changing_the_orders(monkeypatch):
    # the user's goal of 2026-10-24 (DECISOES 132): display only, next to each stock of the plan
    from datetime import timedelta
    _, alvo = _conservadora_sintetica()
    vermelho = alvo.index[0].removesuffix(".SA")
    dia = date.today() - timedelta(days=2)
    fatos = pd.DataFrame({"ticker": [vermelho], "empresa": ["X"], "categoria": ["Fato Relevante"],
                          "assunto": ["Multa"], "data": [dia], "link": ["https://www.rad.cvm.gov.br/a"],
                          "tom_do_fato": ["bad"], "p_ruim": [0.9], "p_bom": [0.0], "p_forte": [0.8]})
    _servicos_sinteticos(monkeypatch)
    monkeypatch.setattr(servicos, "carregar_painel", lambda: (None, fatos, None))
    at = AppTest.from_file(str(APP), default_timeout=TIMEOUT_S)
    at.run()
    assert not at.exception
    luzes = [df.value for df in at.dataframe if "Sinal" in df.value.columns]
    assert len(luzes) == 1
    luzes = luzes[0].set_index("Ticker")
    assert set(luzes.index) == {t.removesuffix(".SA") for t in alvo.index}
    assert luzes.loc[vermelho, "Sinal"] == "🔴 vermelho"
    assert luzes.loc[vermelho, "Fatos da CVM"] == f"{dia:%d/%m} Multa"
    assert set(luzes.drop(index=vermelho)["Sinal"]) == {"🟡 amarelo"}
    assert set(luzes["Notícias e redes"]) == {"—"}
    assert set(luzes.drop(index=vermelho)["Link"]) == {""}      # a blank cell, not Streamlit's grey "None"
    assert len([w for w in at.warning if f"{vermelho} ficou vermelho em {dia:%d/%m}" in w.value]) == 1
    assert any("não muda os pesos" in c.value for c in at.caption)
    assert set(_ordens_alepha(at)["Ordem"]) == {"compra"}


def _app_online(monkeypatch, tmp_path, token: str | None) -> AppTest:
    """The deployed app's view: no lake disk, the real headline loader, the private store's download stubbed (it
    answers only to the token "certo")."""
    from core import manchetes_online
    carregar_manchetes = servicos.carregar_manchetes
    _servicos_sinteticos(monkeypatch)
    monkeypatch.setattr(servicos, "carregar_manchetes", carregar_manchetes)
    monkeypatch.setattr(servicos, "PASTA_DO_LOG", tmp_path / "sem-lago")
    publicado = manchetes_online.para_json(pd.DataFrame({
        "quando": pd.to_datetime(["2026-10-26 09:10"]), "fonte": ["valor"], "titulo": ["Copom mantém Selic"],
        "tema": ["Juros e inflação"], "tom": [-0.25], "link": ["https://valor.com.br/a"]}), datetime(2026, 10, 26, 10, 5))
    monkeypatch.setattr(manchetes_online, "baixar", lambda chave: publicado if chave == "certo" else None)
    st.cache_data.clear()
    at = AppTest.from_file(str(APP), default_timeout=TIMEOUT_S)
    if token is not None:
        at.secrets["manchetes"] = {"token": token}
    at.run()
    return at


def test_online_the_market_panel_shows_the_headlines_from_the_private_store_with_their_publish_time(monkeypatch,
                                                                                                     tmp_path):
    # the user's pick of 2026-10-24 (option (b), DECISOES 134): public repo, headlines read with a token in secrets
    at = _app_online(monkeypatch, tmp_path, "certo")
    assert not at.exception
    tabelas = [df.value for df in at.dataframe if "titulo" in df.value.columns]
    assert len(tabelas) == 1 and tabelas[0]["titulo"].tolist() == ["Copom mantém Selic"]
    assert any("publicadas em 26/10 às 10:05" in c.value for c in at.caption)


def test_online_without_a_valid_token_the_panel_says_the_headlines_are_unavailable_and_still_renders(monkeypatch,
                                                                                                      tmp_path):
    at = _app_online(monkeypatch, tmp_path, "vencido")          # an expired token and no token read alike
    assert not at.exception
    assert not [df for df in at.dataframe if "titulo" in df.value.columns]
    assert any("Manchetes indisponíveis na versão online agora" in c.value for c in at.caption)


# ---------------------------------------------------------------- the DeepSeek adviser (DECISOES 136)

def _app_do_assessor(monkeypatch, chave: str | None) -> tuple[AppTest, list]:
    """The page with the adviser's model call stubbed: it answers an opinion on the first plan stock that cites the
    pack's Selic three times, and one claim with a number the evidence does not hold."""
    from core import assessor
    _servicos_sinteticos(monkeypatch)
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    pedidos = []

    def pedir_parecer(pacote, universo, ferramentas, chave, agora, post=None):
        pedidos.append((pacote, universo, chave))
        selic = next(e for e in pacote if e.texto.startswith("Selic meta"))
        contra = [{"afirmacao": "Selic de 13,75%", "evidencia": selic.id, "citacao": "13,75%"}] * 3
        primeira = universo[0].removesuffix(".SA")
        bruto = {"cenario_macro": [], "ranking": [primeira], "resumo": "Cenário de juros altos.", "limitacoes": [],
                 "acoes": [{"ticker": primeira, "tese": [{"afirmacao": "Juro de 13,75%", "evidencia": selic.id,
                                                         "citacao": "13,75%"}],
                            "contra_tese_macro": contra, "postura": "neutra", "confianca": 0.4,
                            "riscos": [{"afirmacao": "Queda de 99%", "evidencia": selic.id, "citacao": "13,75%"}]}]}
        validacao = assessor.validar_parecer(bruto, {e.id: e for e in pacote}, universo)
        return assessor.Resultado(validacao, None, assessor.Uso(entrada=1000, cache=600, saida=200, raciocinio=100,
                                                                requisicoes=2, custo_usd=0.00042), ["buscar_web"])
    monkeypatch.setattr(assessor, "pedir_parecer", pedir_parecer)
    st.cache_data.clear()
    at = AppTest.from_file(str(APP), default_timeout=TIMEOUT_S)
    if chave is not None:
        at.secrets["deepseek"] = {"api_key": chave}
    at.run()
    return at, pedidos


def test_without_the_deepseek_key_the_adviser_says_it_is_unavailable_and_the_page_still_renders(monkeypatch):
    at, pedidos = _app_do_assessor(monkeypatch, None)
    assert not at.exception and not pedidos
    assert any("Assessor indisponível" in c.value for c in at.caption)
    assert not [b for b in at.button if b.key == "gerar_parecer"]


def test_the_adviser_judges_only_the_plan_stocks_shows_checked_claims_and_its_cost(monkeypatch):
    at, pedidos = _app_do_assessor(monkeypatch, "chave-de-teste")
    assert not at.exception and not pedidos                       # nothing is paid before the click
    at.button(key="gerar_parecer").click().run()
    assert not at.exception and len(pedidos) == 1
    pacote, universo, chave = pedidos[0]
    _, plano = _conservadora_sintetica()
    assert chave == "chave-de-teste" and sorted(universo) == sorted(plano.index)
    tabela = next(df.value for df in at.dataframe if "Postura" in df.value.columns)
    primeira = plano.index[0].removesuffix(".SA")
    assert tabela["Ticker"].tolist()[0] == primeira and tabela["Postura"].tolist()[0] == "🟡 neutra"
    assert any("1 afirmação descartada" in e.label for e in at.expander)
    assert any("cache 60%" in c.value and "US$ 0,0004" in c.value for c in at.caption)
    assert any("não muda os pesos" in c.value for c in at.caption)
