"""Unit tests for the G3 pieces (no lake, no model): control sources, the equal-weight book, the selectors."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from core.backtest import CarteiraRecusada
from scripts.qml_g3_avaliar import (AntiSinal, Embaralhada, SemSinal, estrategia_de_pesos_iguais,
                                    seletor_por_probabilidade)

DATA = pd.Timestamp("2020-03-02")
SETORES = {f"A{i:02d}3": ("Financeiro", "Energia", "Saúde", "Indústria", "Consumo")[i % 5] for i in range(20)}


class _Fonte:
    def __init__(self, probabilidades: pd.Series) -> None:
        self.probabilidades = probabilidades

    def predict(self, _date: pd.Timestamp) -> pd.Series:
        return self.probabilidades


def _fonte() -> _Fonte:
    return _Fonte(pd.Series(np.linspace(0.30, 0.70, 20), index=list(SETORES)))


def _treino(colunas: list[str], n: int = 300) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    return pd.DataFrame(rng.normal(0.0003, 0.01, (n, len(colunas))), columns=colunas,
                        index=pd.bdate_range(end=DATA, periods=n))


def test_control_sources_keep_the_names_and_change_only_what_they_say() -> None:
    base = _fonte()
    original = base.predict(DATA)
    assert (SemSinal(base).predict(DATA) == 0.5).all()
    assert list(SemSinal(base).predict(DATA).index) == list(original.index)
    pd.testing.assert_series_equal(AntiSinal(base).predict(DATA), 1.0 - original)


def test_shuffled_source_is_a_permutation_fixed_by_seed_and_date() -> None:
    base = _fonte()
    original = base.predict(DATA)
    embaralhada = Embaralhada(base, seed=1).predict(DATA)
    assert sorted(embaralhada) == sorted(original)                 # the same values...
    assert not embaralhada.equals(original)                        # ...on other names
    assert embaralhada.equals(Embaralhada(base, seed=1).predict(DATA))
    assert not embaralhada.equals(Embaralhada(base, seed=2).predict(DATA))
    assert not embaralhada.equals(Embaralhada(base, seed=1).predict(DATA + pd.Timedelta(days=1)))


def test_probability_selector_takes_the_highest_p_under_the_sector_limit() -> None:
    treino = _treino(list(SETORES)[:18])                           # the two highest-p names are not eligible
    nomes = seletor_por_probabilidade(_fonte(), SETORES, nomes=15, por_setor=4)(DATA, treino)
    assert len(nomes) == 15
    assert set(nomes) <= set(treino.columns)
    assert nomes[0] == "A173"                                      # the highest P(up) among the eligible
    assert max(pd.Series(nomes).map(SETORES).value_counts()) <= 4


def test_equal_weight_book_sums_to_one_and_drops_names_outside_the_window() -> None:
    treino = _treino(list(SETORES)[:15])
    estrategia = estrategia_de_pesos_iguais(lambda *_: list(SETORES)[:10] + ["ZZZZ3"], SETORES)
    pesos = estrategia(treino)
    assert list(pesos.index) == list(SETORES)[:10]
    assert pesos.to_numpy() == pytest.approx(0.1)


def test_equal_weight_book_is_refused_when_it_breaks_a_fixed_rule() -> None:
    treino = _treino(list(SETORES)[:15])
    um_setor = [papel for papel, setor in SETORES.items() if setor == "Financeiro" and papel in treino.columns]
    with pytest.raises(CarteiraRecusada):                          # three names: 33% each, 100% in one sector
        estrategia_de_pesos_iguais(lambda *_: um_setor, SETORES)(treino)
    with pytest.raises(ValueError):
        estrategia_de_pesos_iguais(lambda *_: [], SETORES)(treino)
