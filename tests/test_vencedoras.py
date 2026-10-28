"""Extreme winners confirmed by profit growth (docs/superpowers/specs/2026-10-28-vencedoras-crescimento-design.md)."""

import numpy as np
import pandas as pd
import pytest


def _mensal(valores: dict[str, list[float]], inicio: str = "2020-01-01") -> pd.DataFrame:
    meses = pd.date_range(inicio, periods=len(next(iter(valores.values()))), freq="MS")
    return pd.DataFrame(valores, index=meses)


def test_the_12_month_return_compounds_months_m_minus_12_to_m_minus_1_and_needs_all_twelve():
    from core.vencedoras import retorno_12m
    retornos = _mensal({"AAAA3": [0.5] + [0.1] * 12 + [9.0],            # Jan/2020 outside, Feb/2020-Jan/2021 inside
                        "BBBB3": [0.0] * 6 + [np.nan] + [0.0] * 7})       # a gap inside the window
    resultado = retorno_12m(retornos, pd.Timestamp("2021-02-01"))
    assert resultado.to_dict() == {"AAAA3": pytest.approx(1.1 ** 12 - 1.0)}


def test_picks_are_winners_above_100pct_with_growth_strictly_above_the_median_of_every_scored_candidate():
    from core.vencedoras import escolhas
    r12 = pd.Series({"AAAA3": 2.0, "BBBB3": 1.5, "CCCC3": 0.5, "DDDD3": 1.2, "EEEE3": 3.0, "GGGG3": 0.1,
                     "HHHH3": 1.0})
    crescimento = pd.Series({"AAAA3": 3.5, "BBBB3": 0.5, "CCCC3": 5.0, "DDDD3": 4.0, "GGGG3": 1.0, "HHHH3": 2.0})
    setores = dict.fromkeys(r12.index, "Industrial")       # median 2.75; EEEE3 unscored; HHHH3 exactly 100%
    assert escolhas(r12, crescimento, setores) == ["DDDD3", "AAAA3"]


def _cov(vols: dict[str, float], correlacao: float = 0.0) -> pd.DataFrame:
    v = pd.Series(vols)
    matriz = np.full((len(v), len(v)), correlacao)
    np.fill_diagonal(matriz, 1.0)
    return pd.DataFrame(matriz * np.outer(v, v), index=v.index, columns=v.index)


def test_each_pick_gets_one_over_n_capped_at_20pct_and_a_sector_above_30pct_is_scaled_down_the_rest_is_cash():
    from core.vencedoras import pesos_das_escolhas
    setores = {"AAAA3": "Industrial", "BBBB3": "Industrial", "CCCC3": "Energia"}
    pesos = pesos_das_escolhas(["AAAA3", "BBBB3", "CCCC3"], setores, _cov(dict.fromkeys(setores, 0.3)))
    assert pesos.to_dict() == {"AAAA3": pytest.approx(0.15), "BBBB3": pytest.approx(0.15), "CCCC3": pytest.approx(0.2)}


def test_stock_weights_with_ex_ante_vol_above_40pct_are_scaled_to_40pct():
    from core.vencedoras import pesos_das_escolhas
    nomes = ["AAAA3", "BBBB3", "CCCC3", "DDDD3", "EEEE3"]
    setores = dict(zip(nomes, ["S1", "S2", "S3", "S4", "S5"]))
    pesos = pesos_das_escolhas(nomes, setores, _cov(dict.fromkeys(nomes, 0.6), correlacao=1.0))  # vol 60% at 20% each
    assert pesos.to_dict() == {nome: pytest.approx(0.2 * 0.4 / 0.6) for nome in nomes}


def _treino(nomes: list[str], fim: str = "2021-02-26", pregoes: int = 300) -> pd.DataFrame:
    gerador = np.random.default_rng(0)
    datas = pd.bdate_range(end=fim, periods=pregoes)
    return pd.DataFrame(gerador.normal(0.0, 0.01, (pregoes, len(nomes))), index=datas, columns=nomes)


def test_the_strategy_holds_the_picks_and_puts_the_rest_in_the_cash_column_using_only_the_engines_eligible_names():
    from core.vencedoras import CAIXA, estrategia_vencedoras
    retornos = _mensal({"AAAA3": [0.1] * 14, "BBBB3": [0.0] * 14, "CCCC3": [0.2] * 14})   # 213%, 0%, 792%
    crescimento = pd.Series({"AAAA3": 2.0, "BBBB3": 0.0, "CCCC3": 9.0})
    vistos = []

    def nota(mes, candidatos):
        vistos.append((mes, list(candidatos)))
        return crescimento.reindex(list(candidatos)).dropna()

    setores = dict.fromkeys(crescimento.index, "Industrial")
    pesos = estrategia_vencedoras(retornos, nota, setores)(_treino(["AAAA3", "BBBB3"]))   # CCCC3 not eligible
    assert vistos == [(pd.Timestamp("2021-02-01"), ["AAAA3", "BBBB3"])]
    assert pesos.to_dict() == {"AAAA3": pytest.approx(0.2), CAIXA: pytest.approx(0.8)}


def test_a_rebalance_with_no_pick_is_all_cash_never_the_old_portfolio():
    from core.vencedoras import CAIXA, estrategia_vencedoras
    retornos = _mensal({"AAAA3": [0.0] * 14, "BBBB3": [0.0] * 14})
    nota = lambda mes, candidatos: pd.Series({"AAAA3": 1.0, "BBBB3": 0.0})   # noqa: E731
    pesos = estrategia_vencedoras(retornos, nota, dict.fromkeys(["AAAA3", "BBBB3"], "Industrial"))(
        _treino(["AAAA3", "BBBB3"]))
    assert pesos.to_dict() == {CAIXA: 1.0}


def test_the_cash_column_is_a_cdi_total_return_index_on_the_engines_sessions_flat_before_the_first_rate():
    from core.vencedoras import indice_do_caixa
    cdi = pd.Series([0.001, 0.002, 0.003], index=pd.to_datetime(["2021-01-04", "2021-01-05", "2021-01-06"]))
    sessoes = pd.to_datetime(["2021-01-01", "2021-01-04", "2021-01-06"])
    indice = indice_do_caixa(cdi, sessoes)
    assert indice.index.equals(sessoes)
    assert indice.pct_change().iloc[1:].to_list() == pytest.approx([0.001, 1.002 * 1.003 - 1.0])
