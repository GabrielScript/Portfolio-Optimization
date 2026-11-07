"""The periphery tilt (core/periferia.py), sections 2-3 of docs/superpowers/specs/2026-11-07-periferia-design.md."""

import numpy as np
import pandas as pd
import pytest

from core.periferia import JANELA, centralidades_na_arvore, estrategia_periferia, periferia_da_janela


def _estrela() -> np.ndarray:
    # hop distances of a star: node 0 is the hub of 4 leaves
    d = np.full((5, 5), 2.0)
    d[0, :] = d[:, 0] = 1.0
    np.fill_diagonal(d, 0.0)
    return d


def test_star_tree_measures():
    medidas = centralidades_na_arvore(_estrela())
    assert medidas["grau"].tolist() == [4, 1, 1, 1, 1]
    assert medidas["intermediacao"].tolist() == [6, 0, 0, 0, 0]          # C(4, 2) pairs cross the hub
    assert medidas["excentricidade"].tolist() == [1, 2, 2, 2, 2]
    assert medidas["proximidade"].iloc[0] == pytest.approx(1.0)
    assert medidas["proximidade"].iloc[1] == pytest.approx(4 / 7)       # distances 1, 2, 2, 2
    assert medidas["autovetor"].iloc[0] > medidas["autovetor"].iloc[1]


def test_path_tree_betweenness_counts_the_pairs_through_each_node():
    # path 0-1-2-3: node 1 lies on (0,2), (0,3); node 2 on (0,3), (1,3)
    d = np.abs(np.subtract.outer(np.arange(4), np.arange(4))).astype(float)
    assert centralidades_na_arvore(d)["intermediacao"].tolist() == [0, 2, 2, 0]


def _retornos(linhas: int = 300, semente: int = 0) -> pd.DataFrame:
    # four names driven by one market factor, one independent name: the independent one is peripheral
    rng = np.random.default_rng(semente)
    mercado = rng.standard_normal(linhas)
    dados = {f"M{i}": mercado + 0.3 * rng.standard_normal(linhas) for i in range(4)}
    dados["X"] = rng.standard_normal(linhas)
    return pd.DataFrame(dados, index=pd.bdate_range("2020-01-01", periods=linhas)) * 0.01


def test_an_independent_stock_is_more_peripheral_than_the_market_names():
    periferia = periferia_da_janela(_retornos())
    assert periferia.between(0.0, 1.0).all()
    assert periferia["X"] >= periferia.drop("X").median()


def test_only_the_last_252_sessions_count():
    retornos = _retornos(600)
    inicio = retornos.iloc[:-JANELA] * 50.0                             # scramble everything before the window
    alterado = pd.concat([inicio.sample(frac=1.0, axis=1, random_state=1).set_axis(retornos.columns, axis=1),
                          retornos.iloc[-JANELA:]])
    pd.testing.assert_series_equal(periferia_da_janela(retornos), periferia_da_janela(alterado))


def test_the_arm_averages_the_two_percentiles_and_keeps_out_names_without_a_composite_score():
    retornos = _retornos()
    vistos = {}

    def nota(mes, candidatos):
        return pd.Series({"M0": 4.0, "M1": 3.0, "M2": 2.0, "M3": 1.0, "X": np.nan}).reindex(candidatos)

    import core.periferia as modulo

    def falsa(nota_do_mes, setores, **_):
        def _estrategia(treino):
            vistos["nota"] = nota_do_mes(pd.Timestamp("2020-12-01"), list(treino.columns))
            return pd.Series(dtype=float)
        return _estrategia

    original = modulo.estrategia_por_nota
    modulo.estrategia_por_nota = falsa
    try:
        estrategia_periferia(nota, {})(retornos)
    finally:
        modulo.estrategia_por_nota = original
    inclinada = vistos["nota"]
    assert np.isnan(inclinada["X"])
    periferia = periferia_da_janela(retornos)[["M0", "M1", "M2", "M3"]].rank(pct=True)
    esperado = 0.5 * pd.Series({"M0": 1.0, "M1": 0.75, "M2": 0.5, "M3": 0.25}) + 0.5 * periferia
    pd.testing.assert_series_equal(inclinada.drop("X"), esperado, check_names=False)
