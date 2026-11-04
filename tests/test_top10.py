"""Attempt 55's selection (docs/superpowers/specs/2026-11-04-top10-design.md)."""

import numpy as np
import pandas as pd
import pytest

from core.top10 import carteira_com_topo, nomes_do_topo


def _treino(retornos_diarios: dict[str, list[float] | float], sessoes: int = 300) -> pd.DataFrame:
    datas = pd.bdate_range("2020-01-01", periods=sessoes)
    return pd.DataFrame({nome: (np.full(sessoes, valor) if np.isscalar(valor) else valor)
                         for nome, valor in retornos_diarios.items()}, index=datas)


def test_the_top_ten_are_the_best_252_session_returns_among_scored_names():
    retornos = {f"A{i:02d}3": 0.0001 * i for i in range(14)}
    retornos["SEMN3"] = 0.01                                   # the best return, but no Composite score
    nota = pd.Series(1.0, index=[f"A{i:02d}3" for i in range(14)])
    topo = nomes_do_topo(_treino(retornos), nota)
    assert set(topo) == {f"A{i:02d}3" for i in range(4, 14)}


def test_a_top_name_15_percent_below_its_63_session_high_is_left_out():
    sessoes = 300
    subiu_e_caiu = [0.004] * (sessoes - 20) + [-0.009] * 20      # about -16.5% from the high in the last 20 sessions
    caiu_pouco = [0.004] * (sessoes - 20) + [-0.005] * 20        # about -9.5%: still in
    retornos = {"PICO3": subiu_e_caiu, "LEVE3": caiu_pouco, **{f"B{i:02d}3": 0.0 for i in range(12)}}
    nota = pd.Series(1.0, index=list(retornos))
    topo = nomes_do_topo(_treino(retornos, sessoes), nota)
    assert "PICO3" not in topo and "LEVE3" in topo


def test_the_portfolio_takes_up_to_4_top_names_in_score_order_then_fills_with_the_composite():
    nota = pd.Series({f"C{i:02d}3": 100.0 - i for i in range(20)} | {"T1AA3": 1.0, "T2BB3": 2.0, "T3CC3": 3.0,
                                                                     "T4DD3": 4.0, "T5EE3": 5.0})
    setores = {papel: f"Setor {i % 6}" for i, papel in enumerate(nota.index)}
    pesos = carteira_com_topo(nota, ["T1AA3", "T2BB3", "T3CC3", "T4DD3", "T5EE3"], setores, nomes=15, por_setor=4,
                              do_topo=4)
    assert len(pesos) == 15 and pesos.sum() == pytest.approx(1.0)
    assert {"T5EE3", "T4DD3", "T3CC3", "T2BB3"} <= set(pesos.index) and "T1AA3" not in pesos.index
    assert all(contagem <= 4 for contagem in pd.Series(setores).reindex(pesos.index).value_counts())


def test_without_qualifying_top_names_the_portfolio_is_the_composites():
    from core.sinais import carteira_da_nota
    nota = pd.Series({f"C{i:02d}3": 100.0 - i for i in range(20)})
    setores = {papel: f"Setor {i % 6}" for i, papel in enumerate(nota.index)}
    pd.testing.assert_series_equal(carteira_com_topo(nota, [], setores, 15, 4, 4), carteira_da_nota(nota, setores, 15, 4))


def test_a_top_name_of_an_already_chosen_issuer_does_not_enter_twice():
    nota = pd.Series({"PETR4": 9.0, "PETR3": 8.0, **{f"C{i:02d}3": 5.0 - i * 0.1 for i in range(20)}})
    setores = {papel: f"Setor {i % 6}" for i, papel in enumerate(nota.index)}
    pesos = carteira_com_topo(nota, ["PETR4", "PETR3"], setores, 15, 4, 4)
    assert "PETR4" in pesos.index and "PETR3" not in pesos.index
