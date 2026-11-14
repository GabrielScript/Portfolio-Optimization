"""The Composite with a BDR sleeve (spec 2026-11-14-composto-bdr): eligibility, score and the mixed book."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from core import manga_bdr as mb

SESSOES = pd.bdate_range("2020-01-01", periods=300)


def _precos(**colunas) -> pd.DataFrame:
    return pd.DataFrame(colunas, index=SESSOES[: len(next(iter(colunas.values())))])


def test_a_thin_or_rarely_traded_bdr_is_not_eligible():
    n = 260
    precos = pd.DataFrame({"AAAA34": np.linspace(10, 12, n), "BBBB34": np.linspace(10, 12, n),
                           "CCCC34": np.linspace(10, 12, n)}, index=SESSOES[:n])
    volumes = pd.DataFrame({"AAAA34": 2e6, "BBBB34": 5e5, "CCCC34": 2e6}, index=SESSOES[:n])
    volumes.iloc[: n // 5, 2] = np.nan            # CCCC34 traded on 80% of the sessions only
    assert mb.elegiveis(precos, volumes, SESSOES[n - 1]) == ["AAAA34"]


def test_the_score_is_the_return_over_the_volatility_of_the_last_252_sessions():
    n = 260
    t = np.arange(n)
    calmo = 10 * 1.001 ** t * (1 + 0.002 * (-1.0) ** t)       # the same growth, a tenth of the swing
    agitado = 10 * 1.001 ** t * (1 + 0.02 * (-1.0) ** t)
    precos = pd.DataFrame({"CALM34": calmo, "WILD34": agitado}, index=SESSOES[:n])
    nota = mb.nota(precos, SESSOES[n - 1])
    janela = precos.iloc[-253:]
    r = janela.pct_change().dropna()
    esperado = (janela.iloc[-1] / janela.iloc[0] - 1) / (r.std() * np.sqrt(252))
    pd.testing.assert_series_equal(nota.sort_index(), esperado.sort_index(), check_names=False)
    assert nota["CALM34"] > nota["WILD34"]


def test_the_top_k_are_the_best_scores_among_the_eligible_ties_by_ticker():
    nota = pd.Series({"BBBB34": 2.0, "AAAA34": 2.0, "CCCC34": 1.0, "DDDD34": 3.0})
    assert mb.top(nota, ["AAAA34", "BBBB34", "CCCC34"], 2) == ["AAAA34", "BBBB34"]


def _composto(n: int, retorno: float = 0.0) -> pd.Series:
    return pd.Series((1 + retorno) ** np.arange(n), index=SESSOES[:n], dtype=float)


def test_without_a_sleeve_the_book_is_the_composite():
    composto = _composto(40, 0.001)
    precos = pd.DataFrame({"AAAA34": 10.0}, index=composto.index)
    liquido, bruto = mb.simular(composto, precos, {}, peso=0.2)
    pd.testing.assert_series_equal(liquido, composto / composto.iloc[0], check_names=False)
    pd.testing.assert_series_equal(bruto, liquido)


def test_a_sleeve_bdr_that_rises_20_percent_adds_its_weight_times_the_rise_and_pays_tax_on_the_sale():
    # flat Composite; the first session decides 20% in AAAA34; it rises 20% in January; on February's first
    # session the book goes back to 80/20, selling 0.2 * 1.2 - 0.2 * 1.04 of it at a gain
    datas = pd.bdate_range("2021-01-04", "2021-02-03")
    composto = pd.Series(1.0, index=datas)
    precos = pd.Series(10.0, index=datas)
    precos[datas > pd.Timestamp("2021-01-15")] = 12.0
    precos = precos.to_frame("AAAA34")
    fevereiro = datas[datas.month == 2][0]
    escolhas = {datas[0]: ["AAAA34"], fevereiro: ["AAAA34"]}
    liquido, bruto = mb.simular(composto, precos, escolhas, peso=0.2, custo_bdr=0.0, custo_composto=0.0)
    assert bruto.iloc[-1] == pytest.approx(1.04)
    vendido = 0.24 - 0.2 * 1.04
    ganho = vendido * (1 - 10.0 / 12.0)
    assert liquido.iloc[-1] == pytest.approx(1.04 - 0.15 * ganho)


def test_a_bdr_sold_at_a_loss_pays_no_tax():
    datas = pd.bdate_range("2021-01-04", "2021-02-03")
    composto = pd.Series(1.0, index=datas)
    precos = pd.Series(10.0, index=datas)
    precos[datas > pd.Timestamp("2021-01-15")] = 8.0
    precos = precos.to_frame("AAAA34")
    fevereiro = datas[datas.month == 2][0]
    liquido, bruto = mb.simular(composto, precos, {datas[0]: ["AAAA34"], fevereiro: []}, peso=0.2,
                                custo_bdr=0.0, custo_composto=0.0)
    pd.testing.assert_series_equal(liquido, bruto)
    assert liquido.iloc[-1] == pytest.approx(0.96)


def test_costs_are_charged_on_the_traded_value_of_each_leg():
    datas = pd.bdate_range("2021-01-04", periods=3)
    composto = pd.Series(1.0, index=datas)
    precos = pd.DataFrame({"AAAA34": 10.0}, index=datas)
    liquido, _ = mb.simular(composto, precos, {datas[0]: ["AAAA34"]}, peso=0.2, custo_bdr=0.005,
                            custo_composto=0.001)
    assert liquido.iloc[-1] == pytest.approx(1.0 - 0.2 * 0.005 - 0.2 * 0.001)


def test_decisions_are_each_months_first_session():
    datas = pd.bdate_range("2021-01-04", "2021-03-10")
    assert list(mb.decisoes(datas)) == [pd.Timestamp("2021-01-04"), pd.Timestamp("2021-02-01"),
                                       pd.Timestamp("2021-03-01")]
