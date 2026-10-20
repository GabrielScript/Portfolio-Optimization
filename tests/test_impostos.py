"""IR sobre ganho em bolsa de pessoa física (S4)."""

import pytest

from core.impostos import apurar_mes


def test_mes_com_vendas_ate_20_mil_e_isento_e_nao_gasta_o_prejuizo_acumulado():
    # ganho de R$ 3 mil com R$ 20 mil vendidos: isento, e os R$ 1 mil de prejuízo de antes ficam para depois
    apuracao = apurar_mes(vendas=20_000.0, resultado=3_000.0, prejuizo_anterior=1_000.0)
    assert (apuracao.imposto, apuracao.prejuizo_acumulado) == (0.0, 1_000.0)


def test_mes_acima_de_20_mil_paga_15_por_cento_do_ganho_depois_do_prejuizo():
    apuracao = apurar_mes(vendas=20_000.01, resultado=3_000.0, prejuizo_anterior=1_000.0)
    assert (apuracao.imposto, apuracao.prejuizo_acumulado) == (pytest.approx(0.15 * 2_000.0), 0.0)


def test_prejuizo_do_mes_acumula_mesmo_em_mes_isento():
    assert apurar_mes(vendas=5_000.0, resultado=-700.0, prejuizo_anterior=300.0).prejuizo_acumulado == 1_000.0
    assert apurar_mes(vendas=50_000.0, resultado=-700.0, prejuizo_anterior=300.0).prejuizo_acumulado == 1_000.0


def test_etf_paga_sobre_qualquer_venda_sem_isencao():
    # ETF de ações: a isenção de R$ 20 mil não vale (isencao = 0)
    apuracao = apurar_mes(vendas=1_000.0, resultado=200.0, prejuizo_anterior=0.0, isencao=0.0)
    assert apuracao.imposto == pytest.approx(30.0)
