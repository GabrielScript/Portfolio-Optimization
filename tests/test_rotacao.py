"""H2 rotation (docs/superpowers/specs/2026-10-29-h2-rotacao-design.md)."""

import numpy as np
import pandas as pd
import pytest

from core.rotacao import CAIXA, JANELA, escolha, rotacao, sleeve_em_reais

T = pd.Timestamp


def test_a_b3_session_uses_the_us_close_strictly_before_it_times_that_days_ptax():
    us = pd.Series([100.0, 110.0], index=[T("2020-01-02"), T("2020-01-03")])
    ptax = pd.Series([4.0, 5.0], index=[T("2020-01-02"), T("2020-01-03")])
    sessoes = pd.DatetimeIndex([T("2020-01-03"), T("2020-01-06")])
    valores = sleeve_em_reais(us, ptax, sessoes, taxa_anual=0.0)
    assert valores.tolist() == [pytest.approx(100.0 * 5.0), pytest.approx(110.0 * 5.0)]   # PTAX ffilled to 06/01


def test_the_fee_drags_one_252th_of_the_yearly_rate_per_session():
    datas = pd.bdate_range("2020-01-01", periods=253)
    us, ptax = pd.Series(1.0, index=datas), pd.Series(1.0, index=datas)
    valores = sleeve_em_reais(us, ptax, datas[1:], taxa_anual=0.0025)
    assert valores.iloc[-1] / valores.iloc[0] == pytest.approx((1 - 0.0025) ** (251 / 252))


def _historico(retornos: dict[str, float], cdi: float) -> tuple[pd.DataFrame, pd.Series]:
    datas = pd.bdate_range("2020-01-01", periods=JANELA + 1)
    valores = pd.DataFrame({nome: (1 + r) ** np.linspace(0, 1, len(datas)) for nome, r in retornos.items()},
                           index=datas)
    return valores, pd.Series((1 + cdi) ** (1 / JANELA) - 1, index=datas)


def test_the_best_twelve_month_sleeve_wins_when_it_beats_the_cdi():
    valores, cdi = _historico({"BR": 0.10, "US": 0.25, "Ouro": 0.05}, cdi=0.12)
    assert escolha(valores, cdi, valores.index[-1]) == "US"


def test_cash_when_no_sleeve_beats_the_cdi():
    valores, cdi = _historico({"BR": 0.10, "US": 0.08, "Ouro": 0.05}, cdi=0.12)
    assert escolha(valores, cdi, valores.index[-1]) == CAIXA


def test_no_choice_without_252_sessions_of_every_sleeve():
    valores, cdi = _historico({"BR": 0.10, "US": 0.25}, cdi=0.0)
    valores.iloc[:5, 1] = np.nan
    assert escolha(valores, cdi, valores.index[-1]) is None


def test_rotation_holds_the_choice_from_the_next_session_and_pays_10_bps_a_side_per_switch():
    datas = pd.bdate_range("2019-01-01", "2020-03-31")
    passo = pd.Series(np.arange(len(datas)), index=datas)
    sleeves = pd.DataFrame({"BR": 1.001 ** passo, "US": 1.0005 ** passo}, index=datas)  # BR always ahead
    cdi = pd.Series(0.0, index=datas)
    valores, escolhas = rotacao(sleeves, cdi, custo=0.001)
    primeira = escolhas.index[0]
    assert escolhas.iloc[0] == "BR" and set(escolhas) == {"BR"}
    assert valores.index[0] == primeira and valores.iloc[0] == pytest.approx(1.0)
    seguinte = datas[datas.get_loc(primeira) + 1]
    assert valores[seguinte] == pytest.approx((1 - 0.001) * 1.001)       # bought BR at the close: one side paid


def test_switching_between_two_risky_sleeves_pays_both_sides():
    datas = pd.bdate_range("2019-01-01", "2020-06-30")
    n = len(datas)
    br = np.r_[1.001 ** np.arange(300), 1.001 ** 299 * 0.995 ** np.arange(1, n - 299)]
    sleeves = pd.DataFrame({"BR": br, "US": 1.0003 ** np.arange(n)}, index=datas)
    valores, escolhas = rotacao(sleeves, pd.Series(0.0, index=datas), custo=0.001)
    troca = escolhas[escolhas != escolhas.shift()].index[1]
    depois = datas[datas.get_loc(troca) + 1]
    esperado = valores[troca] * (1 - 0.001) ** 2 * sleeves.loc[depois, "US"] / sleeves.loc[troca, "US"]
    assert escolhas[troca] == "US" and valores[depois] == pytest.approx(esperado)


def test_keeping_the_same_sleeve_at_a_month_end_costs_nothing():
    datas = pd.bdate_range("2019-01-01", "2020-03-31")
    passo = pd.Series(np.arange(len(datas)), index=datas)
    sleeves = pd.DataFrame({"BR": 1.001 ** passo, "US": 1.0005 ** passo}, index=datas)
    valores, escolhas = rotacao(sleeves, pd.Series(0.0, index=datas), custo=0.001)
    assert len(escolhas) > 2
    crescimento = sleeves["BR"].iloc[-1] / sleeves.loc[escolhas.index[0], "BR"]
    assert valores.iloc[-1] == pytest.approx((1 - 0.001) * crescimento)
