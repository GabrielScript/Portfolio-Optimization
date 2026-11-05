"""Three optimizers fed the Composite's alpha (docs/superpowers/specs/2026-11-05-lote3-composto-design.md, section 1)."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from core.baixo_posto import densa
from core.qpo import IC0, qubo_alphas
from core.risco import LIMITES

NOMES = ("Meio-Kelly-alfa", "Mean-CVaR-alfa", "Robust-MV-alfa")


def _treino(linhas: int = 160, colunas: int = 30) -> pd.DataFrame:
    gerador = np.random.default_rng(0)
    comum = gerador.normal(0.0008, 0.008, linhas)[:, None] * gerador.uniform(0.5, 1.5, colunas)
    dados = comum + gerador.normal(0.0004, 0.012, (linhas, colunas))
    return pd.DataFrame(dados, columns=[f"N{i:03d}" for i in range(colunas)], index=pd.bdate_range("2019-01-01", periods=linhas))


def _setores(treino: pd.DataFrame) -> dict[str, str]:
    return {papel: f"S{i % 6}" for i, papel in enumerate(treino.columns)}


def _nota_decrescente(mes: pd.Timestamp, candidatos: list[str]) -> pd.Series:
    return pd.Series(np.arange(len(candidatos), 0, -1, dtype=float), index=candidatos)


def _nota_crescente(mes: pd.Timestamp, candidatos: list[str]) -> pd.Series:
    return pd.Series(np.arange(1, len(candidatos) + 1, dtype=float), index=candidatos)


def _bracos(nota=_nota_decrescente):
    from core.estrategias_lote3 import bracos_lote3
    return bracos_lote3(nota)


def test_the_arms_are_the_three_preregistered_rows():
    linhas = Path("docs/tentativas.csv").read_text(encoding="utf-8").splitlines()
    assert tuple(_bracos()) == NOMES
    assert all(sum(f'"{nome}:' in linha for linha in linhas) == 1 for nome in NOMES)


@pytest.mark.parametrize("nome", NOMES)
def test_each_arm_gives_a_portfolio_inside_the_rules(nome):
    treino = _treino()
    setores = _setores(treino)
    pesos = _bracos()[nome](setores, 252)(treino)
    assert pesos.sum() == pytest.approx(1.0, abs=1e-6)
    assert pesos.min() >= 0.0
    assert pesos.max() <= LIMITES.peso_maximo_acao + 1e-6
    assert pesos.groupby(pd.Series(setores)).sum().max() <= LIMITES.teto_setor + 1e-6
    assert int((pesos > 1e-9).sum()) <= 10


def test_the_decision_reads_the_grinold_alpha_and_rows_recentred_on_it():
    from core.estrategias_lote3 import decisao_com_alfa
    treino = _treino()
    setores = _setores(treino)
    nota = _nota_decrescente(treino.index[-1], list(treino.columns))
    decisao = decisao_com_alfa(treino, nota, setores, 252, LIMITES)
    esperado = qubo_alphas(nota[decisao.alfa.index], densa(decisao.sigma), IC0)
    pd.testing.assert_series_equal(decisao.alfa, esperado, check_names=False)
    pd.testing.assert_series_equal(decisao.media, esperado / 252, check_names=False)
    pd.testing.assert_series_equal(decisao.retornos.mean(), esperado / 252, check_names=False, atol=1e-12, rtol=0)
    pd.testing.assert_frame_equal(decisao.retornos - decisao.retornos.mean(),
                                  treino[decisao.alfa.index] - treino[decisao.alfa.index].mean(), atol=1e-12, rtol=0)


def test_names_without_a_score_or_a_sector_and_second_classes_stay_out():
    from core.estrategias_lote3 import decisao_com_alfa
    treino = _treino().rename(columns={"N001": "ABCD3", "N002": "ABCD4"})
    setores = _setores(treino)
    del setores["N003"]
    nota = _nota_decrescente(treino.index[-1], list(treino.columns))
    nota["N004"] = np.nan
    nomes = set(decisao_com_alfa(treino, nota, setores, 252, LIMITES).alfa.index)
    assert "N003" not in nomes and "N004" not in nomes
    assert len(nomes & {"ABCD3", "ABCD4"}) == 1


@pytest.mark.parametrize("nome", NOMES)
def test_the_score_moves_the_portfolio_toward_the_names_it_ranks_first(nome):
    treino = _treino()
    setores = _setores(treino)
    primeira_metade = list(treino.columns[:15])
    favorecida = _bracos(_nota_decrescente)[nome](setores, 252)(treino)[primeira_metade].sum()
    preterida = _bracos(_nota_crescente)[nome](setores, 252)(treino)[primeira_metade].sum()
    assert favorecida > preterida + 0.05        # with the score ignored the two portfolios would be the same


def test_the_half_kelly_utility_is_crra_gamma_2_with_consistent_derivatives():
    from core.estrategias_lote3 import MEIO_KELLY
    y = np.array([0.5, 1.0, 1.7])
    passo = 1e-6
    assert MEIO_KELLY.valor(y) == pytest.approx(-1.0 / y)
    assert MEIO_KELLY.primeira(y) == pytest.approx((MEIO_KELLY.valor(y + passo) - MEIO_KELLY.valor(y - passo)) / (2 * passo), rel=1e-6)
    assert MEIO_KELLY.segunda(y) == pytest.approx((MEIO_KELLY.primeira(y + passo) - MEIO_KELLY.primeira(y - passo)) / (2 * passo), rel=1e-5)
    assert np.all(-y * MEIO_KELLY.segunda(y) / MEIO_KELLY.primeira(y) == pytest.approx(2.0))
