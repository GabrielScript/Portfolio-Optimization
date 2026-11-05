"""App inteiro via AppTest, SEM rede: os carregadores de ``ui.servicos`` viram dados sintéticos."""

import re
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

from assets import get_tickers_by_setor
from core import market_data
from ui import servicos

APP = Path(__file__).resolve().parent.parent / "app.py"
# 35 stocks in 4 sectors: the frontier is feasible with the 30% sector cap and solves in seconds.
SETOR_PEQUENO = "Tecnologia da Informação"
SETORES_DO_TESTE = [SETOR_PEQUENO, "Telecomunicações", "Petróleo, Gás e Biocombustíveis", "Materiais Básicos"]
TIMEOUT_S = 300
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
    monkeypatch.setattr(servicos, "carregar_ata_copom", lambda dia: None)
    monkeypatch.setattr(servicos, "carregar_desempenho", lambda tickers, dia, versao, referencia: None)
    monkeypatch.setattr(servicos, "carregar_precos_da_tabela", lambda tickers, dia, versao: _precos_sinteticos(
        tickers, 3, dia, versao).precos)


def _satelite_sintetico() -> tuple[date, pd.Series]:
    """Um registro do paper com papéis dos 4 setores do teste (Tecnologia com 2 papéis, 30%): soma 90%."""
    por_setor = [get_tickers_by_setor([setor]) for setor in SETORES_DO_TESTE]
    return date(2026, 10, 16), pd.Series({por_setor[0][0]: 0.2, por_setor[0][1]: 0.1, por_setor[1][0]: 0.2,
                                          por_setor[2][0]: 0.2, por_setor[3][0]: 0.2})


def _servicos_sinteticos(monkeypatch) -> None:
    """Every loader of ``ui.servicos`` on synthetic data: registrations, sectors, prices, CDI, Selic and quotes."""
    monkeypatch.setattr(servicos, "carregar_satelite_registrado", _satelite_sintetico)
    monkeypatch.setattr(servicos, "carregar_plano_registrado", _conservadora_sintetica)
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
    at.pills(key="setores").set_value(SETORES_DO_TESTE).run()
    return at


# ---------------------------------------------------------------- Alepha: the Conservadora as the default plan

def _conservadora_sintetica() -> tuple[date, pd.Series]:
    """A paper-v2 registration of 5 names at 20% (the per-stock cap), from the test's sectors."""
    por_setor = [get_tickers_by_setor([setor]) for setor in SETORES_DO_TESTE]
    tickers = [por_setor[0][0], por_setor[1][0], por_setor[2][0], por_setor[3][0], por_setor[3][1]]
    return date(2026, 10, 19), pd.Series(0.2, index=tickers)


def _setores_sinteticos(tickers: tuple[str, ...], _pasta=None) -> dict[str, str]:
    return {ticker: f"Setor {posicao}" for posicao, ticker in enumerate(tickers)}


@pytest.fixture
def app_alepha(app_padrao) -> AppTest:
    """The screen as the user opens it: a synthetic Conservadora registration and no liquidity snapshot."""
    return app_padrao


def test_the_page_explains_aleph_quant_instead_of_a_buy_plan(app_alepha):
    assert not app_alepha.exception
    # renamed "Aleph Quant" by the user on 2026-11-04 (was "Aleph Neumann", DECISOES 240 and 250)
    assert any("Aleph Quant" in t.value for t in app_alepha.title)
    # the native theme replaced the injected CSS (the user's design request of 2026-11-04)
    assert not any("<style" in m.value for m in app_alepha.markdown)
    # the user's UI list of 2026-11-04: the buy plan, the held positions and the month's tax are gone (they were empty
    # on every visit); what Aleph Quant is takes their place
    titulos = [m.value for m in app_alepha.markdown]
    assert "## O que é o Aleph Quant" in titulos
    assert not [t for t in titulos if t.startswith("### Plano de compra")]
    assert not [e for e in app_alepha.expander if "Minhas posições" in e.label or "IR do mês" in e.label]
    assert not [df for df in app_alepha.dataframe if "Ordem" in df.value.columns]
    sobre = next(m.value for m in app_alepha.markdown if "ferramenta de decisão" in m.value)
    assert all(palavra in sobre for palavra in ("Composite", "Conservadora", "ETFs", "BDRs", "B3", "dinheiro real"))


def test_no_screen_text_speaks_of_the_lake_paper_v3_decisoes_or_the_cdi(monkeypatch):
    # the user, 2026-11-04: clear words for any investor; the Selic is the only rate on the screen
    _servicos_sinteticos(monkeypatch)
    monkeypatch.setattr(servicos, "carregar_curvas_conservadora", _curvas_sinteticas)
    dias = pd.date_range("2015-01-01", "2026-10-30", freq="D")
    monkeypatch.setattr(servicos, "carregar_juros", lambda: pd.DataFrame({"CDI": 0.0004, "Selic": 0.00041}, index=dias))
    at = AppTest.from_file(str(APP), default_timeout=TIMEOUT_S)
    at.run()
    assert not at.exception
    textos = [*(m.value for m in at.markdown), *(c.value for c in at.caption), *(i.value for i in at.info),
              *(w.value for w in at.warning), *(str(df.value.columns.tolist()) for df in at.dataframe),
              *(str(f) for f in _figuras(at)), *(e.label for e in at.expander)]
    proibidas = re.compile(r"lago|paper-v3|DECISOES|CDI", re.IGNORECASE)
    achados = sorted({m.group(0) for texto in textos for m in [proibidas.search(texto)] if m})
    assert achados == []

def test_the_registered_plan_is_the_composites_last_record(tmp_path):
    from ui.servicos import carregar_plano_registrado
    arquivo = tmp_path / "composto.csv"
    arquivo.write_text("data,ticker,peso,preco\n2026-07-27,VALE3,1.0,60\n2026-10-27,PETR4,0.5,52.79\n"
                       "2026-10-27,ITSA4,0.5,13.33\n", encoding="utf-8")
    data, alvo = carregar_plano_registrado(arquivo)
    assert data == date(2026, 10, 27) and alvo.to_dict() == {"PETR4.SA": 0.5, "ITSA4.SA": 0.5}



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

def test_the_sidebar_keeps_only_the_amount_and_the_frontiers_universe_and_caps(monkeypatch):
    # the user's UI list of 2026-11-03: no plan selector, no study block (objective, lambda, Kelly satellite, vol cap,
    # windows, cost, shrinkage, bagging); the universe is every asset with the B3-listed US ones; the frontier's cap
    # per asset goes up to 25% (user, 2026-11-04)
    _servicos_sinteticos(monkeypatch)
    at = AppTest.from_file(str(APP), default_timeout=TIMEOUT_S)
    at.run()
    assert not at.exception
    # the user, 2026-11-04: every stock by default, so the "all assets" switch is gone; no amount without the plan
    assert at.sidebar.checkbox(key="eua").value is True
    assert not {w.key for w in at.sidebar.checkbox} & {"todos"} and not at.sidebar.number_input
    chaves = {w.key for w in [*at.sidebar.slider, *at.sidebar.checkbox, *at.sidebar.selectbox,
                              *at.sidebar.segmented_control]}
    assert not {"modo_plano", "objetivo", "fracao_satelite", "vol_maxima", "janela", "anos_teste", "custo_bps",
                "reamostrar", "n_ativos_max"} & chaves
    assert at.sidebar.slider(key="peso_maximo").max == 25 and at.sidebar.slider(key="teto_setor").max == 50
    assert at.sidebar.slider(key="teto_setor").value == 30
    assert not [m for m in at.sidebar.markdown if m.value == "## Estudo"]


def _figuras(at: AppTest) -> list[str]:
    return [str(grafico.proto) for grafico in at.get("plotly_chart")]


def test_the_composite_sits_where_it_is_next_to_the_efficient_frontier(app_padrao):
    # the user's request of 2026-11-03: the Composite and the Markowitz frontier replace "Carteira de hoje"; the point
    # is the Composite's own estimate, never moved onto the curve
    assert not app_padrao.exception
    titulos = [m.value for m in app_padrao.markdown]
    assert "## Composite na fronteira eficiente" in titulos
    assert not {"## Carteira de hoje", "## Walk-forward fora da amostra (líquido de custos)"} & set(titulos)
    assert any("Fronteira Eficiente" in figura and "Composite" in figura for figura in _figuras(app_padrao))
    assert any("Composite" in c.value and "fronteira" in c.value for c in app_padrao.caption)
    # the user, 2026-11-04: a plainer text that reads the point against the Selic and the average stock, not a verdict
    texto = next(c.value for c in app_padrao.caption if "Como ler" in c.value)
    assert "Selic" in texto and "média das" in texto and "olhando para trás" in texto
    assert "rende menos do que promete" not in texto


def test_an_infeasible_frontier_explains_what_to_widen_and_the_plan_still_shows(app_padrao):
    # one small sector plus the Composite's names, each capped at 5%, cannot add up to 100%
    app_padrao.pills(key="setores").set_value([SETOR_PEQUENO])
    app_padrao.sidebar.checkbox(key="eua").uncheck()
    app_padrao.sidebar.slider(key="peso_maximo").set_value(5).run()
    assert not app_padrao.exception
    assert len([e.value for e in app_padrao.error if e.value.startswith("Fronteira inviável")]) == 1


def test_the_screen_drops_the_crypto_block_and_the_cvm_light_table(app_padrao):
    titulos = [m.value for m in app_padrao.markdown]
    assert "### BOVA11 e os ETFs de cripto da B3" not in titulos
    assert "#### Sinal de cada ação: fatos relevantes da CVM" not in titulos
    assert not [df for df in app_padrao.dataframe if "Fatos da CVM" in df.value.columns]


SECAO_SEM_VIES = "## Backtest sem viés de sobrevivência: Composite, Conservadora e BOVA11"


def _curvas_sinteticas() -> pd.DataFrame:
    """The live rerun's file (scripts/curvas_ao_vivo.py): 3 phases, the two arms, BOVA11, two crypto ETFs, the CDI."""
    datas = pd.bdate_range("2019-01-01", "2026-10-29")
    crescimento = np.linspace(1.0, 3.0, len(datas)) * (1 + 0.01 * np.sin(np.arange(len(datas)) / 7))
    cripto = np.where(datas >= "2021-08-05", np.linspace(50.0, 100.0, len(datas)), np.nan)
    return pd.concat([pd.DataFrame({"fase": fase, "date": datas, "Conservadora": crescimento * (1 + fase / 1000),
                                    "Composto": crescimento * (1.1 + fase / 1000),
                                    "BOVA11": np.linspace(1.0, 2.0, len(datas)) * (1 + 0.01 * np.cos(np.arange(
                                        len(datas)) / 5)),
                                    "BITH11": cripto, "QBTC11": cripto / 4, "CDI": 0.0004}).iloc[fase:]
                      for fase in (0, 21, 42)],
                     ignore_index=True)


def test_the_fundo_alepha_section_shows_the_live_results_by_period_and_year_with_the_composite_and_crypto(monkeypatch):
    # A2 (2026-10-23) made live (user's request of 2026-10-30, DECISOES 188): the registered Conservadora and the
    # composite rerun to the last close, by period and by calendar year, next to BOVA11, BITH11 and QBTC11, with the
    # risk measures of an aggressive investor; one rebalance phase and the Selic beside them (the user, 2026-11-04)
    _servicos_sinteticos(monkeypatch)
    monkeypatch.setattr(servicos, "carregar_curvas_conservadora", _curvas_sinteticas)
    monkeypatch.setattr(servicos, "carregar_juros", lambda: None)             # no file: the table without them
    at = AppTest.from_file(str(APP), default_timeout=TIMEOUT_S)
    at.run()
    assert not at.exception
    # the user's UI list of 2026-11-03: the survivorship-free backtest replaces the biased walk-forward, with its
    # curves, right below the real plan and above the simulator
    titulos = [m.value for m in at.markdown]
    assert titulos.index(SECAO_SEM_VIES) < titulos.index("## Simulador")
    assert any(all(nome in figura for nome in ("Conservadora", "Composto", "BOVA11", "Selic")) for figura in _figuras(at))
    assert not [s for s in at.segmented_control if s.key == "fase_das_curvas"]          # no phase picker
    periodos = [df.value for df in at.dataframe if "Período" in df.value.columns]
    assert len(periodos) == 1
    assert list(periodos[0]["Período"]) == ["Período inteiro", "Últimos 5 anos", "Últimos 3 anos"]
    assert {"Retorno a.a. Conservadora", "Retorno a.a. Composto", "Retorno a.a. BOVA11", "Retorno a.a. Selic"} <= set(
        periodos[0].columns)
    anos = [df.value for df in at.dataframe if {"Ano", "BITH11"} <= set(df.value.columns)]
    assert len(anos) == 1 and anos[0]["Ano"].tolist()[-1] == "2026 (até 29/10)"
    assert ["Conservadora", "Composto", "BOVA11", "BITH11", "QBTC11"] == list(anos[0].columns[1:6])
    assert anos[0].set_index("Ano").loc["2019", "BITH11"] == "—"    # before its listing: a dash, not "None" (2026-11-04)
    resumo = [df.value for df in at.dataframe if "Sortino" in df.value.columns]
    assert len(resumo) == 1 and list(resumo[0]["Estratégia"]) == ["Conservadora", "Composto", "BOVA11", "BITH11",
                                                                   "QBTC11"]
    assert {"Alfa a.a. (BOVA11)", "Queda máx.", "Calmar", "CVaR 95% diário"} <= set(resumo[0].columns)
    at.selectbox(key="medida_por_ano").set_value("Queda máx.").run()
    assert not at.exception
    anos = [df.value for df in at.dataframe if {"Ano", "BITH11"} <= set(df.value.columns)][0]
    assert (anos[["Conservadora", "BOVA11"]].dropna() <= 0).all().all()


def test_the_year_table_has_the_selic_next_to_the_returns_and_only_there(monkeypatch):
    # the user's request of 2026-11-01, then 2026-11-04: each year's return next to the Selic only (no CDI)
    _servicos_sinteticos(monkeypatch)
    monkeypatch.setattr(servicos, "carregar_curvas_conservadora", _curvas_sinteticas)
    dias = pd.date_range("2015-01-01", "2026-10-30", freq="D")
    monkeypatch.setattr(servicos, "carregar_juros", lambda: pd.DataFrame({"CDI": 0.0004, "Selic": 0.00041}, index=dias))
    at = AppTest.from_file(str(APP), default_timeout=TIMEOUT_S)
    at.run()
    assert not at.exception
    anos = [df.value for df in at.dataframe if {"Ano", "BITH11"} <= set(df.value.columns)][0]
    assert list(anos.columns[1:5]) == ["Conservadora", "Composto", "BOVA11", "Selic"] and "CDI" not in anos.columns
    assert anos.set_index("Ano").loc["2020", "Selic"] == pytest.approx(1.00041 ** 366 - 1)
    at.selectbox(key="medida_por_ano").set_value("Queda máx.").run()
    assert not at.exception                                     # a rate has no drawdown to show
    anos = [df.value for df in at.dataframe if {"Ano", "BITH11"} <= set(df.value.columns)][0]
    assert not {"CDI", "Selic"} & set(anos.columns)


def test_the_market_panel_shows_news_the_dollar_and_every_stock_without_the_election(monkeypatch):
    # the user's request of 2026-10-23 (news and macro for every stock), trimmed by the UI list of 2026-11-03: no
    # election item, two news numbers, the dollar alone from the Focus survey, no source codes or dates
    _servicos_sinteticos(monkeypatch)
    focus = pd.DataFrame({"data": pd.to_datetime(["2026-09-18", "2026-10-16", "2026-10-16"]),
                          "ano_referencia": [2026, 2026, 2027], "mediana": [5.5, 5.4, 5.6], "desvio": 0.1})
    monkeypatch.setattr(servicos, "carregar_focus", lambda dia: {"Selic": focus, "Câmbio": focus})
    diario = pd.DataFrame({"dia": ["2026-11-02", "2026-11-03"], "itens": [40, 50], "relevantes": [10, 12],
                           "tom": [0.1, -0.2], "eleicao": [0.3, 0.2],
                           **{tema: 0.1 for tema in __import__("core.sentimento", fromlist=["x"]).TOPICOS_DO_PAINEL}})
    monkeypatch.setattr(servicos, "carregar_painel", lambda: (diario, None, None))
    at = AppTest.from_file(str(APP), default_timeout=TIMEOUT_S)
    at.run()
    assert not at.exception
    assert any(m.value == "## Mercado agora: notícias, macro e ações" for m in at.markdown)
    assert not [i for i in at.info if "Eleição" in i.value]
    rotulos = [m.label for m in at.metric]
    assert {"Notícias relevantes para a bolsa", "Tom do mercado"} <= set(rotulos)
    assert not {"Itens no dia", "Sobre a eleição", "IPCA 12 meses (%)"} & set(rotulos)
    tom = next(m for m in at.metric if m.label == "Tom do mercado")
    assert "notícias boas" in tom.help
    ano = date.today().year
    assert {f"Projeção Focus para o fim de {ano}", f"Projeção Focus para o fim de {ano + 1}"} <= set(rotulos)
    assert not [m for m in at.metric if m.label == "Selic meta"]
    assert not [df for df in at.dataframe if "Pesquisa" in df.value.columns]          # no Focus table, no dates
    acoes = [df.value for df in at.dataframe if {"Retorno 12m", "Setor"} <= set(df.value.columns)]
    assert len(acoes) == 1 and len(acoes[0]) >= 100
    assert any("nada aqui muda o plano" in c.value for c in at.caption)
    assert not [c for c in at.caption if "DECISOES 112" in c.value or "Log desde" in c.value]


def test_the_market_panel_shows_the_central_bank_indicators_the_risk_measures_and_the_facts_in_portuguese(
        monkeypatch):
    # the user's UI list of 2026-11-04: IBC-Br, IDP, credit, its cost, defaults and the Copom minutes with a "?" each;
    # CAGR, Sharpe, Sortino, Calmar, CVaR, alpha, its t and the information ratio in the table; the facts' tone in
    # Portuguese with each Jev score explained; no "lago" in the captions
    from core import macro_bcb
    from ui import painel
    _servicos_sinteticos(monkeypatch)
    mes = pd.Timestamp("2026-09-01")
    monkeypatch.setattr(servicos, "carregar_indicadores_bcb", lambda dia: {
        macro_bcb.IBC_BR: (mes, 2.1), macro_bcb.INADIMPLENCIA: (mes, 4.98), macro_bcb.JUROS_DO_CREDITO: (mes, 32.4)})
    monkeypatch.setattr(servicos, "carregar_ata_copom", lambda dia: macro_bcb.Ata(
        pd.Timestamp("2026-09-16"), "Ata da 281ª reunião do Copom", "https://www.bcb.gov.br/a.pdf"))
    monkeypatch.setattr(servicos, "carregar_desempenho", lambda tickers, dia, versao, referencia: pd.DataFrame(
        {"Sharpe": [0.8], "t do alfa": [2.3]}, index=["PETR4.SA"]))
    fatos = pd.DataFrame({"ticker": ["PETR4", "VALE3"], "empresa": ["Petrobras", "Vale"],
                          "categoria": ["Fato Relevante"] * 2, "assunto": ["a", "b"],
                          "data": pd.to_datetime(["2026-10-30", "2026-10-29"]), "link": ["https://cvm/1"] * 2,
                          "tom_do_fato": ["bad", None], "p_ruim": [0.9, None], "p_bom": [0.05, None],
                          "p_forte": [0.7, None]})
    monkeypatch.setattr(servicos, "carregar_painel", lambda: (None, fatos, None))
    at = AppTest.from_file(str(APP), default_timeout=TIMEOUT_S)
    at.run()
    assert not at.exception
    assert any(m.value == "### Banco Central (ao vivo)" for m in at.markdown)
    inadimplencia = next(m for m in at.metric if m.label == macro_bcb.INADIMPLENCIA)
    assert inadimplencia.value == "4,98%" and inadimplencia.help == macro_bcb.EXPLICACOES[macro_bcb.INADIMPLENCIA]
    assert any("Ata da 281ª reunião do Copom" in m.value for m in at.markdown)
    tabela = [df.value for df in at.dataframe if {"Retorno 12m", "Sharpe"} <= set(df.value.columns)][0]
    assert tabela.set_index("Ticker").loc["PETR4", ["Sharpe", "t do alfa"]].tolist() == [0.8, 2.3]
    na_tela = [df.value for df in at.dataframe if "tom_do_fato" in df.value.columns][0]
    assert na_tela["tom_do_fato"].tolist() == ["ruim", painel.SEM_NOTA]
    assert not [c for c in at.caption if "lago" in c.value.lower()]


def _tabela_de_acoes(at: AppTest) -> pd.DataFrame:
    return [df.value for df in at.dataframe if {"P/L", "Retorno 12m"} <= set(df.value.columns)][0]


def test_the_market_panel_lists_bdrs_and_etfs_and_filters_by_type_without_a_type_column(monkeypatch):
    # the user's request of 2026-10-27 (every BDR and ETF of the B3 next to the stocks); the UI list of 2026-11-03
    # took the "Tipo" column off and put a filter per category column in a "Filtros" button
    _servicos_sinteticos(monkeypatch)
    nomes = pd.DataFrame({"security": ["TSMC34", "XINA11"], "name": ["TAIWANSMFAC", "XPETFCHINA"],
                          "asset_type": ["bdr", "fund_etf"], "sector": [None, None]})
    monkeypatch.setattr(servicos, "carregar_ativos_da_tela", lambda: nomes)
    at = AppTest.from_file(str(APP), default_timeout=TIMEOUT_S)
    at.run()
    assert not at.exception
    tabela = _tabela_de_acoes(at).set_index("Ticker")
    assert "Tipo" not in tabela.columns
    assert tabela.loc["TSMC34", ["Empresa", "Setor"]].tolist() == ["TAIWANSMFAC", "BDR"]
    assert {"XINA11", "PETR4"} <= set(tabela.index)
    assert any("BDRs: retorno só de preço" in c.value for c in at.caption)
    at.multiselect(key="filtro_acoes_Tipo").set_value(["BDR"]).run()
    assert not at.exception
    so_bdrs = set(_tabela_de_acoes(at)["Ticker"])
    assert "TSMC34" in so_bdrs and not {"XINA11", "PETR4"} & so_bdrs
    assert all(ticker.endswith(("32", "33", "34", "35")) for ticker in so_bdrs)
    at.multiselect(key="filtro_acoes_Tipo").set_value(["ETF"]).run()
    assert "XINA11" in set(_tabela_de_acoes(at)["Ticker"]) and "TSMC34" not in set(_tabela_de_acoes(at)["Ticker"])


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
    # the user, 2026-11-04: no "Link" column, the headline itself is the blue link to the article
    assert len(tabelas) == 1 and "link" not in tabelas[0].columns
    from ui import painel
    assert tabelas[0]["titulo"].tolist() == [f"https://valor.com.br/a{painel.MARCA_DA_MANCHETE}Copom mantém Selic"]
    assert re.search(painel.TEXTO_DA_MANCHETE, tabelas[0]["titulo"].iloc[0]).group(1) == "Copom mantém Selic"
    assert not [c for c in at.caption if "publicadas em" in c.value]      # the headline captions are gone


def test_online_without_a_valid_token_the_panel_says_the_headlines_are_unavailable_and_still_renders(monkeypatch,
                                                                                                      tmp_path):
    at = _app_online(monkeypatch, tmp_path, "vencido")          # an expired token and no token read alike
    assert not at.exception
    assert not [df for df in at.dataframe if "titulo" in df.value.columns]
    assert any(c.value == "Manchetes indisponíveis agora." for c in at.caption)


# ---------------------------------------------------------------- the DeepSeek adviser (DECISOES 136)

def _app_do_assessor(monkeypatch, chave: str | None) -> tuple[AppTest, list]:
    """The page with the adviser's model call stubbed: it answers an opinion on the first plan stock that cites the
    pack's Selic three times, and one claim with a number the evidence does not hold."""
    from core import assessor
    _servicos_sinteticos(monkeypatch)
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    pedidos, esforcos = [], []
    _app_do_assessor.esforcos = esforcos

    def pedir_parecer(pacote, universo, ferramentas, chave, agora, post=None, carteira=None, esforco=assessor.ESFORCO,
                      partes=1):
        assert carteira == assessor.CARTEIRA_COMPOSTA          # the plan the screen judges is the Composite (DECISOES 242)
        pedidos.append((pacote, universo, chave))
        esforcos.append((esforco, partes))
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


def test_the_opinion_depth_is_picked_on_the_screen_max_by_default_and_each_depth_is_its_own_opinion(monkeypatch):
    # the user, 2026-11-04: max effort takes 7-8 minutes, too long for a presentation; switching the effort must not
    # need a file edit or a request to Claude
    from core import assessor
    at, pedidos = _app_do_assessor(monkeypatch, "chave-de-teste")
    esforcos = _app_do_assessor.esforcos
    # 2026-11-04, later: "Rápida" is high effort in 3 parallel parts (~75 s) and "Expressa" in 5 (~65 s)
    controle = at.segmented_control(key="esforco_do_parecer")
    from ui.assessor import ROTULO_DA_PROFUNDIDADE
    assert controle.value == assessor.PROFUNDIDADE_PADRAO == "maxima"
    assert controle.options == [ROTULO_DA_PROFUNDIDADE[p] for p in assessor.PROFUNDIDADES]
    at.button(key="gerar_parecer").click().run()
    at.segmented_control(key="esforco_do_parecer").set_value("rapida").run()
    at.button(key="gerar_parecer").click().run()
    at.segmented_control(key="esforco_do_parecer").set_value("expressa").run()
    at.button(key="gerar_parecer").click().run()
    assert not at.exception and esforcos == [("max", 1), ("high", 3), ("high", 5)]   # each depth is its own opinion
    at.segmented_control(key="esforco_do_parecer").set_value("maxima").run()
    at.button(key="gerar_parecer").click().run()
    assert esforcos == [("max", 1), ("high", 3), ("high", 5)]       # the max opinion of the day comes from the cache


def test_paid_opinions_stop_at_the_daily_cap_for_every_visitor_of_the_process(monkeypatch):
    # the deployed app is public: the owner's key pays, so a cleared cache or many visitors cannot run up the bill
    from core import assessor
    pagos = []
    monkeypatch.setattr(assessor, "pedir_parecer", lambda *argumentos, **nomeados: pagos.append(1) or
                        assessor.Resultado(None, None, assessor.Uso(), []))
    monkeypatch.setattr(servicos, "PARECERES_POR_DIA", 2)
    st.cache_data.clear()
    servicos.contador_de_pareceres().clear()
    resultados = [servicos.gerar_parecer(f"pacote {i}", ("PETR4.SA",), date(2026, 10, 26), (), None, "k")
                  for i in range(3)]
    assert len(pagos) == 2 and "limite diário" in resultados[2].erro
    assert servicos.gerar_parecer("pacote 9", ("PETR4.SA",), date(2026, 10, 27), (), None, "k").erro is None


def test_the_us_stocks_screen_is_on_standby_while_the_b3_listed_us_assets_stay(app_alepha):
    # the user's decision of 2026-11-03 (DECISOES 240): B3 only; US stocks off the screen, code and data kept; the
    # B3-listed BDRs and ETFs are in the frontier's universe by default (UI list of 2026-11-03)
    assert not app_alepha.exception
    assert not any(m.value == "## Ações americanas" for m in app_alepha.markdown)
    assert app_alepha.sidebar.checkbox(key="eua").value is True


def _ativos_da_fronteira(at: AppTest) -> int:
    legenda = next(c.value for c in at.caption if c.value.startswith("Como ler"))
    return int(re.search(r"cada uma das (\d+) ações", legenda).group(1))


def test_the_b3_listed_us_assets_add_32_names_to_the_frontier(app_padrao):
    com_eua = _ativos_da_fronteira(app_padrao)
    app_padrao.sidebar.checkbox(key="eua").uncheck().run()
    assert not app_padrao.exception
    assert _ativos_da_fronteira(app_padrao) == com_eua - 32


def test_every_table_has_its_filters(monkeypatch):
    # the user's UI list of 2026-11-03: a "Filtros" button above each table, one multiselect per category column
    from ui import alepha, filtros, painel, simulador
    from ui import assessor as ui_assessor
    filtradas = []

    def gravar(tabela, chave, so_para_filtro=()):
        saida = filtros.filtrar(tabela, chave, so_para_filtro)
        filtradas.append(tuple(saida.columns))
        return saida

    for modulo in (alepha, painel, simulador, ui_assessor):
        monkeypatch.setattr(modulo, "filtrar", gravar)
    _, alvo = _conservadora_sintetica()
    hoje = date.today()
    fatos = pd.DataFrame({"ticker": [alvo.index[0].removesuffix(".SA"), "VALE3", "PETR4"],
                          "empresa": ["X", "VALE", "PETROBRAS"],
                          "categoria": ["Fato Relevante", "Comunicado ao Mercado", "Fato Relevante"],
                          "assunto": ["a", "b", "c"], "data": [hoje] * 3,
                          "link": [f"https://www.rad.cvm.gov.br/{n}" for n in "abc"]})
    manchetes = pd.DataFrame({"quando": pd.to_datetime(["2026-11-03 09:00", "2026-11-03 10:00", "2026-11-03 11:00"]),
                              "fonte": ["valor", "infomoney", "valor"], "titulo": ["a", "b", "c"],
                              "tema": ["Juros e inflação", "Empresas", "Empresas"], "tom": [0.1, -0.2, 0.3],
                              "link": ["https://valor.com.br/a", "https://infomoney.com.br/b", "https://valor.com.br/c"]})
    registro = {"quando": "2026-11-03T20:37:00+00:00", "base_do_composto": "2026-10-27", "modelo": "deepseek-flash",
                "esforco": "max", "universo": ["PETR4", "VALE3", "ITSA4", "BBSE3"],
                "parecer": {"acoes": [{"ticker": "PETR4", "postura": "favoravel", "confianca": 0.7},
                                      {"ticker": "BBSE3", "postura": "favoravel", "confianca": 0.6},
                                      {"ticker": "VALE3", "postura": "neutra", "confianca": 0.5}]}}
    from core.sentimento import TOPICOS_DO_PAINEL
    diario = pd.DataFrame({"dia": ["2026-11-02", "2026-11-03"], "itens": [40, 50], "relevantes": [10, 12],
                           "tom": [0.1, -0.2], "eleicao": [0.3, 0.2], **dict.fromkeys(TOPICOS_DO_PAINEL, 0.1)})
    at, _ = _app_do_assessor(monkeypatch, "chave-de-teste")
    monkeypatch.setattr(servicos, "carregar_painel", lambda: (diario, fatos, None))
    monkeypatch.setattr(servicos, "carregar_manchetes", lambda: (manchetes, None))
    monkeypatch.setattr(servicos, "carregar_parecer_deepseek", lambda: registro)
    monkeypatch.setattr(servicos, "carregar_curvas_conservadora", _curvas_sinteticas)
    at.run()
    at.button(key="gerar_parecer").click().run()
    assert not at.exception
    mostradas = [tuple(df.value.columns) for df in at.dataframe]
    sem_filtro = [colunas[:3] for colunas in mostradas if colunas not in filtradas]
    assert mostradas and not sem_filtro, sem_filtro      # every table shown, with or without a category column
    assert len(at.get("popover")) >= len(mostradas)
    chaves = {m.key for m in at.multiselect}
    assert {"filtro_fatos_categoria", "filtro_manchetes_fonte",
            "filtro_parecer_deepseek_Parecer do DeepSeek", "filtro_acoes_Tipo", "filtro_acoes_Setor"} <= chaves
