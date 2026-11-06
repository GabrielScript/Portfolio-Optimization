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


def test_the_amount_follows_the_users_balance_rule_of_2026_11_06():
    # the user, 2026-11-06 (second rule): R$ 500 with 11,000-12,000, R$ 750 with 12,500-13,000, R$ 1,000 above 13,000;
    # below 11,000 nothing; the gap 12,000-12,500 keeps the lower tier (DECISOES 284)
    from scripts.aporte_real import valor_do_aporte
    assert valor_do_aporte(10_999.99) == 0
    assert valor_do_aporte(11_000) == valor_do_aporte(12_000) == valor_do_aporte(12_499.99) == 500
    assert valor_do_aporte(12_500) == valor_do_aporte(12_999.99) == 750
    assert valor_do_aporte(13_000) == valor_do_aporte(30_000) == 1_000


def test_the_alpha_is_the_agents_account_against_the_composite_alone():
    from scripts.aporte_real import alfa_contra_a_base
    assert round(alfa_contra_a_base(11_500.0, 11_000.0), 6) == round(11_500 / 11_000 - 1, 6)


def test_before_the_sentinela_account_has_a_close_the_script_says_so_instead_of_crashing():
    from datetime import date
    from types import SimpleNamespace

    import pytest

    from scripts.aporte_real import saldo_da_conta
    for painel in (None, SimpleNamespace(curvas=None)):
        with pytest.raises(SystemExit, match="ainda não tem saldo"):
            saldo_da_conta(date(2026, 11, 6), carregar=lambda dia, versao, painel=painel: painel)
