"""Rebalance from the current positions (S4, ``core/execucao.py``): orders = target quantities of the purchase plan on
the whole patrimony minus the quantities held; then cost, the month's IR and the no-trade zone."""

import pandas as pd
import pytest

from core.execucao import (MesFiscal, Posicao, RegrasRebalanceio, ZonaSemNegociacao, ordens_do_rebalanceio,
                           plano_de_rebalanceio)

SEM_TETO = RegrasRebalanceio(peso_maximo=1.0, custo_bps=0.0, quantidade_minima=1)


def test_the_orders_take_the_held_quantities_to_the_purchase_plan_of_the_whole_patrimony():
    # patrimony 10 x 10 + 3 x 10 + 100 cash = 230; plan: A 11 (floor of 115 / 10), B 5, then the R$ 20 left buys 1 B
    alvo = pd.Series({"A": 0.5, "B": 0.5})
    posicoes = {"A": Posicao(10, 9.0), "C": Posicao(3, 5.0)}
    cotacoes = pd.Series({"A": 10.0, "B": 20.0, "C": 10.0})
    ordens = ordens_do_rebalanceio(alvo, posicoes, cotacoes, caixa=100.0, regras=SEM_TETO)
    assert ordens["delta"].to_dict() == {"A": 1, "B": 6, "C": -3}
    assert ordens["lado"].to_dict() == {"A": "compra", "B": "compra", "C": "venda"}


def test_each_order_splits_into_standard_lots_and_the_fractional_rest_and_has_its_value():
    # A goes from 250 to 0 shares (sell 250 = 200 in lots + 50 fractional); B from 0 to 125 (100 + 25)
    ordens = ordens_do_rebalanceio(pd.Series({"B": 1.0}), {"A": Posicao(250, 1.0)}, pd.Series({"A": 2.0, "B": 4.0}),
                                   caixa=0.0, regras=SEM_TETO)
    assert ordens.loc[["A", "B"], ["lote_padrao", "fracionario"]].to_numpy().tolist() == [[200, 50], [100, 25]]
    assert ordens.loc[["A", "B"], "valor"].tolist() == [500.0, 500.0]


def um_por_cento(ticker: str, valor: float) -> float:
    return 0.01 * valor


def test_each_order_pays_its_cost_and_the_cash_left_is_cash_plus_sales_minus_buys_and_costs():
    # reserve of 1%: patrimony 230 -> 227.72 to invest; A 11 (110), B 5 (100), and the R$ 17.72 left buys 1 A
    regras = RegrasRebalanceio(peso_maximo=1.0, custo_bps=100.0, quantidade_minima=1)
    plano = plano_de_rebalanceio(pd.Series({"A": 0.5, "B": 0.5}), {"A": Posicao(10, 9.0), "C": Posicao(3, 5.0)},
                                 pd.Series({"A": 10.0, "B": 20.0, "C": 10.0}), 100.0, regras, um_por_cento)
    assert plano.ordens["custo"].to_dict() == pytest.approx({"A": 0.2, "B": 1.0, "C": 0.3})
    assert plano.caixa_final == pytest.approx(100.0 + 30.0 - 20.0 - 100.0 - 1.5)


def test_when_the_costs_pass_the_plans_reserve_the_buys_shrink_until_the_cash_is_not_negative():
    # no reserve: the plan buys 10 A for the R$ 100; with 1% of cost that is R$ 101, so A drops to 9
    plano = plano_de_rebalanceio(pd.Series({"A": 1.0}), {}, pd.Series({"A": 10.0}), 100.0, SEM_TETO, um_por_cento)
    assert plano.ordens.loc["A", "delta"] == 9
    assert plano.caixa_final == pytest.approx(100.0 - 90.0 - 0.9)


def sem_custo(ticker: str, valor: float) -> float:
    return 0.0


def test_a_sale_that_takes_the_months_stock_sales_past_20_thousand_pays_the_ir_of_the_whole_month():
    # R$ 19 thousand sold elsewhere this month with R$ 2 thousand of gain; selling all A (R$ 2,000, gain R$ 500)
    # makes the month R$ 21 thousand: 15% of R$ 2,500 is due, all of it caused by this order
    mes = MesFiscal(vendas_de_acoes=19_000.0, resultado_de_acoes=2_000.0)
    plano = plano_de_rebalanceio(pd.Series({"B": 1.0}), {"A": Posicao(100, 15.0)}, pd.Series({"A": 20.0, "B": 10.0}),
                                 0.0, SEM_TETO, sem_custo, mes=mes)
    assert plano.ordens.loc["A", "ganho_realizado"] == pytest.approx(500.0)
    assert plano.ordens.loc["A", "ir_marginal"] == pytest.approx(375.0)
    assert plano.ir_do_mes == pytest.approx(375.0)
    assert plano.vendas_de_acoes_no_mes == pytest.approx(21_000.0)


def test_an_etf_sale_pays_15_percent_without_the_exemption_and_does_not_count_as_a_stock_sale():
    plano = plano_de_rebalanceio(pd.Series({"B": 1.0}), {"BOVA11": Posicao(1, 180.0)},
                                 pd.Series({"BOVA11": 200.0, "B": 10.0}), 0.0, SEM_TETO, sem_custo,
                                 etfs=frozenset({"BOVA11"}))
    assert plano.ir_do_mes == pytest.approx(0.15 * 20.0)
    assert plano.ordens.loc["BOVA11", "ir_marginal"] == pytest.approx(3.0)
    assert plano.vendas_de_acoes_no_mes == 0.0


def test_the_no_trade_zone_skips_a_resize_whose_cost_is_above_its_expected_gain():
    # patrimony 2,000; 5 B would cost R$ 50.50 with the R$ 50 of cash, so the order is 4 B (R$ 40, 2 p.p. of B):
    # gain (2/2) x 0.02^2 x 63 x 0.02^2 x 2,000 = R$ 0.02, cost 1% = R$ 0.40: skipped, and the R$ 50 stay in cash
    zona = ZonaSemNegociacao(vol_diaria=pd.Series({"A": 0.02, "B": 0.02}))
    plano = plano_de_rebalanceio(pd.Series({"A": 0.5, "B": 0.5}), {"A": Posicao(100, 10.0), "B": Posicao(95, 10.0)},
                                 pd.Series({"A": 10.0, "B": 10.0}), 50.0, SEM_TETO, um_por_cento, zona=zona)
    assert plano.ordens.loc["B", "delta"] == 0
    assert plano.ordens.loc["B", "ganho_esperado"] == pytest.approx(0.02 ** 2 * 63 * 0.02 ** 2 * 2_000)
    assert plano.ordens.loc["B", "situacao"] == "pular: custo acima do ganho"
    assert plano.caixa_final == pytest.approx(50.0)


def test_the_no_trade_zone_never_skips_a_ticker_entering_or_leaving_the_portfolio():
    # A leaves and B enters; both orders cost more than their risk gain (1% against a tiny vol) and still trade
    zona = ZonaSemNegociacao(vol_diaria=pd.Series({"A": 1e-4, "B": 1e-4}))
    plano = plano_de_rebalanceio(pd.Series({"B": 1.0}), {"A": Posicao(10, 10.0)}, pd.Series({"A": 10.0, "B": 10.0}),
                                 0.0, SEM_TETO, um_por_cento, zona=zona)
    assert (plano.ordens["custo"] > plano.ordens["ganho_esperado"]).all()
    assert plano.ordens["delta"].to_dict() == {"A": -10, "B": 9}
    assert plano.ordens["situacao"].to_dict() == {"A": "executar", "B": "executar"}


def test_an_order_the_zone_would_skip_trades_when_skipping_it_breaks_the_cap_per_ticker():
    # cap 50%: A at 1,050 of 2,000 (52.5%) must sell 5 even though the gain is tiny; B's resize is skipped
    zona = ZonaSemNegociacao(vol_diaria=pd.Series({"A": 1e-4, "B": 1e-4}))
    regras = RegrasRebalanceio(peso_maximo=0.5, custo_bps=0.0, quantidade_minima=1)
    plano = plano_de_rebalanceio(pd.Series({"A": 0.5, "B": 0.5}), {"A": Posicao(105, 10.0), "B": Posicao(95, 10.0)},
                                 pd.Series({"A": 10.0, "B": 10.0}), 0.0, regras, um_por_cento, zona=zona)
    assert plano.ordens["delta"].to_dict() == {"A": -5, "B": 0}
    assert plano.ordens["situacao"].to_dict() == {"A": "forçada: regra de risco", "B": "pular: custo acima do ganho"}
    assert plano.caixa_final == pytest.approx(50.0 - 0.5)


def test_an_order_the_zone_would_skip_trades_when_skipping_it_breaks_a_group_ceiling():
    # sector {A, B} capped at 50%: skipping both sells would leave it at 52%, so both trade; C's buy is skipped
    from core.config import LimiteGrupo
    zona = ZonaSemNegociacao(vol_diaria=pd.Series({"A": 1e-4, "B": 1e-4, "C": 1e-4}))
    regras = RegrasRebalanceio(peso_maximo=1.0, custo_bps=0.0, quantidade_minima=1,
                               grupos=(LimiteGrupo("S", frozenset({"A", "B"}), 0.5),))
    plano = plano_de_rebalanceio(pd.Series({"A": 0.25, "B": 0.25, "C": 0.5}),
                                 {"A": Posicao(52, 10.0), "B": Posicao(52, 10.0), "C": Posicao(96, 10.0)},
                                 pd.Series({"A": 10.0, "B": 10.0, "C": 10.0}), 0.0, regras, um_por_cento, zona=zona)
    assert plano.ordens["delta"].to_dict() == {"A": -2, "B": -2, "C": 0}
    assert plano.ordens.loc[["A", "B"], "situacao"].tolist() == ["forçada: regra de risco"] * 2


def test_the_zone_counts_the_marginal_ir_so_a_resize_that_would_end_the_exemption_is_skipped():
    # R$ 19,500 already sold this month: trimming A by R$ 1,000 (gain R$ 500) costs R$ 75 of IR against a risk gain
    # of 0.02^2 x 63 x 0.05^2 x 20,000 = R$ 1.26; without the sale there is no cash for B's buy either
    zona = ZonaSemNegociacao(vol_diaria=pd.Series({"A": 0.02, "B": 0.02}))
    plano = plano_de_rebalanceio(pd.Series({"A": 0.5, "B": 0.5}), {"A": Posicao(1_100, 5.0), "B": Posicao(900, 10.0)},
                                 pd.Series({"A": 10.0, "B": 10.0}), 0.0, SEM_TETO, sem_custo,
                                 mes=MesFiscal(vendas_de_acoes=19_500.0), zona=zona)
    assert plano.ordens["delta"].to_dict() == {"A": 0, "B": 0}
    assert plano.ordens["situacao"].to_dict() == {"A": "pular: custo acima do ganho", "B": ""}
    assert plano.ir_do_mes == 0.0
