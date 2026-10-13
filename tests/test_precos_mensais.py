"""Testes de market_data.carregar_precos_mensais (sem rede)."""

from datetime import date

import numpy as np
import pandas as pd
import pytest

from core import market_data

HOJE = date(2026, 10, 13)


@pytest.fixture
def yahoo(monkeypatch):
    """AAA.SA sobe 1 por dia útil; BBB.SA só existe em setembro e falta no último pregão dele."""
    dias = pd.bdate_range("2026-08-03", "2026-10-13")
    precos = pd.DataFrame({"AAA.SA": np.arange(len(dias), dtype=float) + 10.0}, index=dias)
    precos["BBB.SA"] = np.nan
    setembro = (dias >= "2026-09-01") & (dias <= "2026-09-29")
    precos.loc[setembro, "BBB.SA"] = 50.0
    pedidos = []

    def baixar(tickers, inicio, fim):
        pedidos.append((list(tickers), inicio, fim))
        return precos[tickers]
    monkeypatch.setattr(market_data, "baixar_fechamentos", baixar)
    return precos, pedidos


def test_ultimo_preco_de_cada_mes_completo(yahoo):
    precos, pedidos = yahoo
    mensais = market_data.carregar_precos_mensais(["AAA.SA", "BBB.SA"], date(2026, 8, 1), HOJE)

    assert pedidos == [(["AAA.SA", "BBB.SA"], date(2026, 8, 1), HOJE)]
    assert isinstance(mensais.index, pd.PeriodIndex) and mensais.index.name == "mes"
    assert list(mensais.index) == [pd.Period("2026-08", freq="M"), pd.Period("2026-09", freq="M")]
    assert mensais.loc[pd.Period("2026-08", freq="M"), "AAA.SA"] == precos.loc["2026-08-31", "AAA.SA"]
    assert mensais.loc[pd.Period("2026-09", freq="M"), "AAA.SA"] == precos.loc["2026-09-30", "AAA.SA"]


def test_sem_filtro_de_cobertura_e_ultimo_valor_valido_do_mes(yahoo):
    mensais = market_data.carregar_precos_mensais(["AAA.SA", "BBB.SA"], date(2026, 8, 1), HOJE)
    assert np.isnan(mensais.loc[pd.Period("2026-08", freq="M"), "BBB.SA"])
    assert mensais.loc[pd.Period("2026-09", freq="M"), "BBB.SA"] == 50.0
