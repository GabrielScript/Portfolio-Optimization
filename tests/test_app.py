"""App inteiro via AppTest, SEM rede: os carregadores de ``ui.servicos`` viram dados sintéticos."""

from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from assets import BENCHMARK_ACOES, BENCHMARKS, get_tickers_by_setor
from core import market_data
from core.config import Objetivo
from ui import servicos

APP = Path(__file__).resolve().parent.parent / "app.py"
# 8 ações: viável com teto de 20% por ativo, e a reamostragem roda em segundos.
SETOR_PEQUENO = "Tecnologia da Informação"
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


@pytest.fixture
def app(monkeypatch) -> AppTest:
    monkeypatch.setattr(servicos, "carregar_precos", _precos_sinteticos)
    monkeypatch.setattr(servicos, "carregar_cdi", _cdi_sintetico)
    monkeypatch.setattr(servicos, "carregar_selic", lambda dia: market_data.TaxaSelic(0.1375, "sintética"))
    monkeypatch.setattr(servicos, "carregar_cotacoes", _cotacoes_sinteticas)
    at = AppTest.from_file(str(APP), default_timeout=TIMEOUT_S)
    at.run()
    at.sidebar.multiselect(key="setores").set_value([SETOR_PEQUENO])
    at.sidebar.slider(key="teto_setor").set_value(100)               # um setor só: sem teto por setor
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
    assert len(colunas) == 1 + 4 + 3                      # Métrica + 4 estratégias + BOVA11, SMAL11, DIVO11
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
    tickers = tuple(get_tickers_by_setor([SETOR_PEQUENO])) + BENCHMARKS     # o mesmo pedido do walk-forward
    bova = _precos_sinteticos(tickers, JANELA_ANOS + TESTE_ANOS, date.today(), 0).precos[BENCHMARK_ACOES]
    esperado = (1 - 10 / 10_000) * bova.iloc[-1] / bova.iloc[JANELA_ANOS * 252] - 1

    tabela = _tabela_walk_forward(app).set_index("Métrica")

    assert tabela.loc["Retorno total", "BOVA11"] == f"{esperado * 100:.2f}%"


def _precos_sem_historico_longo_para_a_ultima(tickers: tuple[str, ...], anos: int, dia: date,
                                              versao_dados: int) -> market_data.DadosPrecos:
    """A última ação do universo não tem janela + teste anos de histórico: some só do download do walk-forward."""
    precos = _precos_sinteticos(tickers, anos, dia, versao_dados).precos
    if anos > JANELA_ANOS:
        precos = precos.drop(columns=[t for t in tickers if t not in BENCHMARKS][-1])
    return market_data.DadosPrecos(precos)


def test_legenda_do_walk_forward_lista_a_acao_sem_historico_para_ele(app, monkeypatch):
    monkeypatch.setattr(servicos, "carregar_precos", _precos_sem_historico_longo_para_a_ultima)
    app.run()

    assert not app.exception
    curta = get_tickers_by_setor([SETOR_PEQUENO])[-1].replace(".SA", "")
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
    assert len(tabela.columns) == 1 + 5 + 3
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
    app.sidebar.selectbox[0].set_value(Objetivo.HRP.value).run()

    assert not app.exception
    cara = get_tickers_by_setor([SETOR_PEQUENO])[-1].replace(".SA", "")
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


SETOR_AUXILIAR = "Telecomunicações"          # 4 ações: com teto de 70%, os dois setores ainda somam 100%


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
    monkeypatch.setattr(servicos, "carregar_precos", _precos_tecnologia_em_alta)
    app.sidebar.multiselect(key="setores").set_value([SETOR_PEQUENO, SETOR_AUXILIAR])
    app.sidebar.slider(key="teto_setor").set_value(70).run()

    assert not app.exception
    assert (_peso_por_setor(_plano_de_compra(app), "Peso alvo") <= 0.70 + 1e-6).all()


def test_plano_mostra_o_peso_efetivo_de_cada_setor(app):
    assert not app.exception
    efetivo = _peso_por_setor(_plano_de_compra(app), "Peso efetivo")[SETOR_PEQUENO]
    legendas = [c.value for c in app.caption if c.value.startswith("Por setor")]
    assert len(legendas) == 1 and f"{SETOR_PEQUENO} {efetivo:.1%}" in legendas[0]


def test_um_setor_so_com_teto_por_setor_explica_o_que_aumentar(app):
    app.sidebar.slider(key="teto_setor").set_value(30).run()

    assert not app.exception
    erros = [e.value for e in app.error]
    assert len(erros) == 1 and "teto por setor" in erros[0]


def test_reamostragem_so_vale_para_o_kelly(app):
    app.sidebar.checkbox(key="reamostrar").check().run()
    app.sidebar.selectbox[0].set_value(Objetivo.MIN_VOL.value).run()

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


def test_walk_forward_compara_com_bova11_smal11_e_divo11(app):
    # referências de mercado: Ibovespa, small caps e dividendos (o alfa continua medido contra o BOVA11)
    assert list(_tabela_walk_forward(app).columns)[-3:] == ["BOVA11", "SMAL11", "DIVO11"]


def test_referencia_sem_historico_na_janela_aparece_na_legenda(app, monkeypatch):
    # DIVO11 só existe desde 2012: num walk-forward de 20 anos ela não tem a janela inteira
    def sem_divo11(tickers, anos, dia, versao_dados):
        dados = _precos_sinteticos(tuple(t for t in tickers if t != "DIVO11.SA"), anos, dia, versao_dados)
        return market_data.DadosPrecos(dados.precos, descartados_cobertura=["DIVO11.SA"] if "DIVO11.SA" in tickers else [])

    monkeypatch.setattr(servicos, "carregar_precos", sem_divo11)
    app.run()
    legendas = [c.value for c in app.caption if c.value.startswith("Sem histórico")]
    assert legendas == [f"Sem histórico para os {JANELA_ANOS + TESTE_ANOS} anos do walk-forward: DIVO11."]
