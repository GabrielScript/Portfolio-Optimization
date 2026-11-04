"""The DeepSeek adviser's pack (ui/parecer.py) for any registered portfolio."""

from datetime import date

import numpy as np
import pandas as pd

from core import market_data
from ui import parecer, servicos


def test_a_plan_stock_outside_the_apps_181_still_gets_its_market_numbers(monkeypatch):
    # the first DeepSeek opinion (03/11/2026) left MLAS3 and SAUD3 as "evidencia_insuficiente": the pack only had the
    # numbers of the app's 181 stocks, and the Composite holds names of the whole lake
    pedidos = []

    def precos(tickers, dia, versao):
        pedidos.append(tickers)
        datas = pd.bdate_range(end=pd.Timestamp(dia), periods=800)
        retornos = np.random.default_rng(3).normal(0.0005, 0.015, size=(len(datas), len(tickers)))
        return pd.DataFrame(10 * np.cumprod(1 + retornos, axis=0), index=datas, columns=list(tickers))

    monkeypatch.setattr(servicos, "carregar_painel", lambda: (None, None, None))
    monkeypatch.setattr(servicos, "carregar_precos_da_tabela", precos)
    monkeypatch.setattr(servicos, "carregar_focus", lambda dia: None)
    monkeypatch.setattr(servicos, "carregar_indicadores_bcb", lambda dia: None)
    monkeypatch.setattr(servicos, "carregar_selic", lambda dia: market_data.TaxaSelic(0.15, "sintética"))
    monkeypatch.setattr(servicos, "carregar_manchetes", lambda: (None, None))
    monkeypatch.setattr(servicos, "carregar_ativos_da_tela", lambda: pd.DataFrame(
        {"security": ["MLAS3"], "name": ["GRUPO MULTI"], "asset_type": ["stock"], "sector": ["Consumo Cíclico"]}))
    alvo = pd.Series({"MLAS3.SA": 0.5, "PETR4.SA": 0.5})

    pacote, _ = parecer.pacote_e_fontes(alvo, date(2026, 10, 27), "k", date(2026, 11, 4))

    assert "MLAS3.SA" in pedidos[0]
    mercado = [e.texto for e in pacote if e.classe == "mercado"]
    assert any(texto.startswith("MLAS3") for texto in mercado) and any(texto.startswith("PETR4") for texto in mercado)
