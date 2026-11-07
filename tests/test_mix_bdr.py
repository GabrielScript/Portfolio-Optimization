"""The fixed BR/BDR mixes (core/mix_bdr.py), section 2 of docs/superpowers/specs/2026-11-07-mix-bdr-design.md."""

import numpy as np
import pandas as pd
import pytest

from core.mix_bdr import curva_mista, elegiveis_no_mes, escolher_eua, momentum_12_1

DIAS = pd.bdate_range("2019-01-01", periods=400)


def _precos() -> pd.DataFrame:
    passos = np.arange(len(DIAS))
    return pd.DataFrame({"A": 1.0 + 0.002 * passos, "B": 1.0 + 0.001 * passos, "C": np.ones(len(DIAS))}, index=DIAS)


def test_momentum_12_1_is_the_return_from_252_to_21_sessions_back_and_ignores_later_prices():
    precos = _precos()
    data = DIAS[300]
    esperado = precos.iloc[300 - 21] / precos.iloc[300 - 252] - 1.0
    pd.testing.assert_series_equal(momentum_12_1(precos, data), esperado, check_names=False)
    depois = precos.copy()
    depois.iloc[301:] *= 7.0
    pd.testing.assert_series_equal(momentum_12_1(depois, data), momentum_12_1(precos, data))


def test_a_name_is_eligible_only_with_enough_sessions_and_volume_in_the_previous_12_months():
    volume = pd.DataFrame({"A": 2e6, "B": 5e5, "C": 2e6}, index=DIAS)
    volume.loc[DIAS[200:300], "C"] = np.nan                            # C skipped a third of the window
    assert elegiveis_no_mes(volume, pd.Timestamp("2020-05-01")) == ["A"]


def test_the_us_sleeve_takes_the_k_best_eligible_by_momentum_or_fewer():
    precos = _precos()
    volume = pd.DataFrame(2e6, index=DIAS, columns=precos.columns)
    data = DIAS[300]
    assert escolher_eua(precos, volume, data, 2) == ["A", "B"]
    volume["A"] = 1e5
    assert escolher_eua(precos, volume, data, 2) == ["B", "C"]
    assert escolher_eua(precos, volume, data, 5) == ["B", "C"]          # only two eligible


def test_the_mix_resets_the_weights_at_each_rebalance_and_charges_the_traded_share():
    dias = pd.bdate_range("2021-01-04", periods=5)
    br = pd.Series([100.0, 100.0, 100.0, 100.0, 100.0], index=dias)     # the BR sleeve stands still
    eua = pd.DataFrame({nome: [10.0, 20.0, 20.0, 20.0, 20.0] for nome in "XYZ"}, index=dias)  # all double on day 1
    tres = ["X", "Y", "Z"]
    valores = curva_mista(br, eua, {dias[0]: tres, dias[2]: tres}, k=3, custo_bps=0.0)
    # 12/15 BR + 3/15 US: the US fifth doubles, so 1.2 times the start
    assert valores.iloc[1] == pytest.approx(1.2 * valores.iloc[0])
    assert valores.iloc[-1] == pytest.approx(valores.iloc[1])
    com_custo = curva_mista(br, eua, {dias[0]: tres, dias[2]: tres}, k=3, custo_bps=10.0)
    # day 0 buys only the US 3/15 (the BR sleeve is a running curve); day 2 resets the US share from 1/3 to 1/5
    assert com_custo.iloc[0] == pytest.approx(valores.iloc[0] * (1 - 0.001 * 3 / 15))
    assert com_custo.iloc[2] < com_custo.iloc[1]


def test_a_rebalance_with_fewer_us_names_gives_their_slots_to_the_br_sleeve():
    dias = pd.bdate_range("2021-01-04", periods=3)
    br = pd.Series([100.0, 110.0, 110.0], index=dias)
    eua = pd.DataFrame({"X": [10.0, 10.0, 10.0]}, index=dias)
    valores = curva_mista(br, eua, {dias[0]: []}, k=2, custo_bps=0.0)
    assert valores.iloc[1] == pytest.approx(1.1 * valores.iloc[0])     # all in the BR sleeve
