import pandas as pd
import pytest

from scripts import aporte_real


def test_the_ticket_buys_the_composite_with_whole_shares_in_the_fractional_market_at_the_closing_auction():
    pesos = pd.Series(1 / 3, index=["AAAA3", "BBBB4", "CCCC3"])
    cotacoes = pd.Series({"AAAA3": 10.0, "BBBB4": 20.0, "CCCC3": 160.0})
    boleta, plano = aporte_real.boleta(pesos, cotacoes, 500.0)
    assert plano.investido + plano.custo_estimado <= 500.0 + 1e-9
    assert set(boleta.columns) == {"Ativo", "Quantidade", "Preço de referência (R$)", "Valor (R$)", "Ordem"}
    assert (boleta["Ordem"] == "compra no leilão de fechamento").all()
    assert boleta.loc[boleta["Ativo"] == "AAAA3F", "Quantidade"].iloc[0] > 0           # under 100 shares: fractional
    assert boleta["Valor (R$)"].sum() == pytest.approx(plano.investido)


def test_lots_of_100_go_to_the_standard_market_and_the_rest_to_the_fractional():
    linhas = aporte_real.linhas_da_ordem("AAAA3", 250, 2.0)
    assert [(linha["Ativo"], linha["Quantidade"]) for linha in linhas] == [("AAAA3", 200), ("AAAA3F", 50)]
