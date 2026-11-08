"""The joint B3/BDR ranking (docs/superpowers/specs/2026-11-08-bdr-conjunto-design.md, sections 3 and 4)."""

import numpy as np
import pandas as pd
import pytest


def _precos(colunas: list[str], sessoes: int = 300, semente: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(semente)
    retornos = rng.normal(0.0005, 0.02, size=(sessoes, len(colunas))) * np.linspace(0.5, 2.0, len(colunas))
    return pd.DataFrame(100.0 * np.cumprod(1.0 + retornos, axis=0), index=pd.bdate_range("2020-01-01", periods=sessoes),
                        columns=colunas)


def test_the_score_is_the_stouffer_sum_of_pool_wide_price_scores_and_the_b3_only_composite_score():
    from core.bdr_conjunto import nota_conjunta
    from core.mix_bdr import momentum_12_1
    from core.views import escore_normal
    precos = _precos(["AAAA3", "BBBB3", "CCCC3", "NVDC34", "MSFT34"])
    data = precos.index[-1]
    composto = pd.Series({"AAAA3": 0.3, "BBBB3": -0.1, "CCCC3": 0.9})
    nota = nota_conjunta(composto, precos, data)
    momento = escore_normal(momentum_12_1(precos, data))
    risco = escore_normal(-precos.pct_change().iloc[-252:].std())
    fundamento = escore_normal(composto)
    esperado_b3 = (momento + risco + fundamento).dropna() / np.sqrt(3)
    esperado_eua = (momento + risco)[["NVDC34", "MSFT34"]] / np.sqrt(2)
    assert nota[["AAAA3", "BBBB3", "CCCC3"]].to_numpy() == pytest.approx(esperado_b3[["AAAA3", "BBBB3", "CCCC3"]].to_numpy())
    assert nota[["NVDC34", "MSFT34"]].to_numpy() == pytest.approx(esperado_eua.to_numpy())


def test_a_name_without_a_price_component_keeps_the_ones_it_has():
    from core.bdr_conjunto import nota_conjunta
    precos = _precos(["AAAA3", "BBBB3", "NVDC34"])
    precos.loc[precos.index[:100], "NVDC34"] = np.nan            # 200 sessions: no 12-1 momentum, still a volatility
    nota = nota_conjunta(pd.Series({"AAAA3": 0.1, "BBBB3": 0.2}), precos, precos.index[-1])
    from core.views import escore_normal
    risco = escore_normal(-precos.pct_change().iloc[-252:].std())
    assert nota["NVDC34"] == pytest.approx(risco["NVDC34"] / 1.0)


def test_the_simulator_rebalances_to_target_pays_costs_on_traded_weight_and_drifts_with_carried_prices():
    from core.bdr_conjunto import simular
    dias = pd.bdate_range("2021-01-04", periods=6)
    precos = pd.DataFrame({"A": [10.0, 11.0, 12.0, 12.0, 6.0, 6.0], "B": [5.0, 5.0, 5.0, np.nan, np.nan, np.nan],
                           "C": [1.0, 1.0, 1.0, 2.0, 2.0, 4.0]}, index=dias)
    alvos = {dias[0]: pd.Series({"A": 0.5, "B": 0.5}), dias[3]: pd.Series({"A": 0.5, "C": 0.5})}
    valores = simular(alvos, precos, custo_bps=10.0)
    custo = 10.0 / 1e4
    v0 = 1.0 - custo * 1.0                                         # buys 100% of the portfolio
    v2 = v0 * (0.5 * 12.0 / 10.0 + 0.5)                            # B flat
    assert valores.iloc[2] == pytest.approx(v2)
    pesos_antes = np.array([0.5 * 12.0 / 10.0, 0.5]) * v0 / v2      # A, B at the 4th session (B carried at 5.0)
    negociado = abs(0.5 - pesos_antes[0]) + pesos_antes[1] + 0.5     # A trimmed, B sold, C bought
    v3 = v2 * (1.0 - custo * negociado)
    assert valores.iloc[3] == pytest.approx(v3)
    assert valores.iloc[5] == pytest.approx(v3 * (0.5 * 6.0 / 12.0 + 0.5 * 4.0 / 2.0))
    assert valores.index[0] == dias[0] and len(valores) == 6
