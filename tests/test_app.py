"""App inteiro via AppTest, SEM rede: os carregadores de ``ui.servicos`` viram dados sintéticos."""

from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from core import market_data
from core.config import Objetivo
from ui import servicos

APP = Path(__file__).resolve().parent.parent / "app.py"
# 8 ações + 2 cripto: viável com teto de 20% por ativo e 10% em cripto, e a reamostragem roda em segundos.
SETOR_PEQUENO = "Tecnologia da Informação"
TIMEOUT_S = 300


def _precos_sinteticos(tickers: tuple[str, ...], anos: int, dia: date) -> market_data.DadosPrecos:
    gerador = np.random.default_rng(7)
    datas = pd.bdate_range(end=pd.Timestamp(dia), periods=650)     # 504 de treino + 3 rebalanceamentos
    retornos = gerador.normal(0.0006, 0.015, size=(len(datas), len(tickers)))
    precos = pd.DataFrame(100 * np.cumprod(1 + retornos, axis=0), index=datas, columns=list(tickers))
    return market_data.DadosPrecos(precos, fonte_cambio="sintético")


def _cdi_sintetico(anos: int, dia: date) -> pd.Series:
    return pd.Series(0.0005, index=pd.bdate_range(end=pd.Timestamp(dia), periods=900))


def _growth_sintetico(tickers, precos, setores, usar_cvm, dia) -> pd.DataFrame:
    scores = np.linspace(20, 90, len(tickers))
    return pd.DataFrame({"score": scores, "queima_caixa": False}, index=list(tickers))


@pytest.fixture
def app(monkeypatch) -> AppTest:
    monkeypatch.setattr(servicos, "carregar_precos", _precos_sinteticos)
    monkeypatch.setattr(servicos, "carregar_cdi", _cdi_sintetico)
    monkeypatch.setattr(servicos, "carregar_selic", lambda dia: market_data.TaxaSelic(0.1375, "sintética"))
    monkeypatch.setattr(servicos, "carregar_growth", _growth_sintetico)
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


def test_reamostragem_so_vale_para_o_kelly(app):
    app.sidebar.checkbox(key="reamostrar").check().run()
    app.sidebar.selectbox[0].set_value(Objetivo.MIN_VOL.value).run()

    assert not app.exception
    assert app.sidebar.checkbox(key="reamostrar").disabled
    assert list(_tabela_walk_forward(app).columns)[1].startswith("Otimizada")
    assert not [e for e in app.expander if e.label == "Estabilidade nas reamostras"]
