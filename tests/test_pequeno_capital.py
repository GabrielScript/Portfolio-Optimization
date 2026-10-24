"""Small capital (user's decision of 2026-10-23: start now, below the minimum capital): the held core ETF stays, the new
cash buys the Conservadora's names one share at a time, the caps measured on the whole patrimony
(``core.rebalanceio.plano_com_nucleo``)."""

import pandas as pd
import pytest

from core.rebalanceio import Posicao, RegrasRebalanceio, pesos_finais, plano_com_nucleo

COTACOES = pd.Series({"BOVA11": 203.0, "VALE3": 66.36, "PETR4": 58.0, "FIQE3": 6.22, "ITSA4": 15.45})
ALVO = pd.Series({"VALE3": 0.25, "PETR4": 0.25, "FIQE3": 0.25, "ITSA4": 0.25})
REGRAS = RegrasRebalanceio(peso_maximo=0.20, custo_bps=0.0, quantidade_minima=1)


def sem_custo(ticker: str, valor: float) -> float:
    return 0.0


def _plano(caixa: float, posicoes=None):
    return plano_com_nucleo(ALVO, posicoes if posicoes is not None else {"BOVA11": Posicao(1, 200.0)}, COTACOES, caixa,
                            REGRAS, sem_custo, nucleo=frozenset({"BOVA11"}))


def test_the_held_core_etf_is_kept_and_not_sold():
    ordens = _plano(100.0).ordens
    assert ordens.loc["BOVA11", "lado"] == ""
    assert ordens.loc["BOVA11", "alvo"] == 1


def test_the_20_percent_cap_is_measured_on_the_whole_patrimony_core_included():
    # 1 BOVA11 (203) + 120 cash = 323 -> cap 64.6: PETR4 (58) fits, VALE3 (66.36) does not; on the sleeve's
    # 120 alone the cap would be 24 and neither would fit
    ordens = _plano(120.0).ordens
    assert ordens.loc["PETR4", "alvo"] == 1
    assert ordens.loc["VALE3", "alvo"] == 0


def test_more_capital_lets_the_dearest_name_in():
    # 203 + 140 = 343 -> cap 68.6 >= 66.36
    assert _plano(140.0).ordens.loc["VALE3", "alvo"] == 1


def test_the_final_weights_for_the_risk_rules_leave_the_core_out_and_divide_by_the_whole_patrimony():
    plano = _plano(120.0)
    patrimonio = 203.0 + 120.0
    pesos = pesos_finais(plano.ordens, patrimonio, fora=frozenset({"BOVA11"}))
    assert "BOVA11" not in pesos.index
    assert pesos["PETR4"] == pytest.approx(58.0 / patrimonio)
    assert (pesos > 0).all()
