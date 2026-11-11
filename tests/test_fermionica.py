"""Fermionic (determinantal) selection: the objective's identities, the swap ascent, the caps and the strategy."""

import itertools

import numpy as np
import pandas as pd
import pytest

from core.fermionica import THETA, estrategia_fermionica, fermionic_picks, objective
from core.qubo_universo import GroupCaps


def _corr(n: int, seed: int) -> np.ndarray:
    fatores = np.random.default_rng(seed).normal(size=(4 * n, n))
    fatores[:, 1:] += 0.8 * fatores[:, [0]]
    return np.corrcoef(fatores, rowvar=False)


def test_objective_of_a_pair_is_the_scores_plus_log_one_minus_rho_squared():
    c = np.array([[1.0, 0.6], [0.6, 1.0]])
    assert objective(np.array([0.5, -0.2]), c, (0, 1), 1.0) == pytest.approx(2 * 0.3 + np.log(1 - 0.36))
    assert objective(np.array([0.5, -0.2]), c, (0, 1), 3.0) == pytest.approx(6 * 0.3 + np.log(1 - 0.36))


def test_log_det_is_never_positive_and_zero_only_without_correlation():
    c = _corr(9, 1)
    zeros = np.zeros(9)
    assert all(objective(zeros, c, s, 1.0) < 0 for s in itertools.combinations(range(9), 4))
    assert objective(zeros, np.eye(5), (0, 2, 4), 1.0) == pytest.approx(0.0)


def test_log_det_is_the_sum_of_log_unexplained_variance_in_any_order():
    c = _corr(7, 2)
    for ordem in ((0, 3, 5, 6), (6, 0, 5, 3)):
        soma = 0.0
        for posicao, j in enumerate(ordem):
            antes = list(ordem[:posicao])
            r2 = c[j, antes] @ np.linalg.solve(c[np.ix_(antes, antes)], c[antes, j]) if antes else 0.0
            soma += np.log(1.0 - r2)
        assert objective(np.zeros(7), c, ordem, 1.0) == pytest.approx(soma)


def test_a_near_duplicate_is_left_out_for_a_slightly_lower_independent_name():
    c = np.eye(4)
    c[0, 1] = c[1, 0] = 0.95
    z = np.array([1.0, 0.9, 0.5, -1.0])
    assert fermionic_picks(z, c, 2, None, THETA) == (0, 2)      # 2(1.5) + 0 > 2(1.9) + log(1 - 0.9025)


def test_picks_match_exhaustive_search_on_small_instances():
    for seed in range(6):
        c, z = _corr(9, seed), np.random.default_rng(100 + seed).normal(size=9)
        melhor = max(itertools.combinations(range(9), 4), key=lambda s: objective(z, c, s, THETA))
        assert objective(z, c, fermionic_picks(z, c, 4, None, THETA), THETA) == pytest.approx(
            objective(z, c, melhor, THETA))


def test_no_single_swap_improves_the_picks_and_the_caps_hold():
    c, z = _corr(30, 7), np.random.default_rng(8).normal(size=30)
    caps = GroupCaps(tuple("abcde"[i % 5] for i in range(30)), 2)
    picks = fermionic_picks(z, c, 8, caps, THETA)
    assert len(picks) == 8 and max(pd.Series([caps.groups[i] for i in picks]).value_counts()) <= 2
    atual = objective(z, c, picks, THETA)
    for sai, entra in itertools.product(picks, set(range(30)) - set(picks)):
        troca = tuple(sorted(set(picks) - {sai} | {entra}))
        if max(pd.Series([caps.groups[i] for i in troca]).value_counts()) <= 2:
            assert objective(z, c, troca, THETA) <= atual + 1e-9


def test_a_huge_theta_is_the_ranking_by_score_under_the_caps():
    c, z = _corr(20, 3), np.random.default_rng(4).normal(size=20)
    caps = GroupCaps(tuple("ab"[i % 2] for i in range(20)), 3)
    esperado, contagem = [], {"a": 0, "b": 0}
    for i in np.argsort(-z, kind="stable"):
        if contagem[caps.groups[i]] < 3:
            esperado.append(int(i))
            contagem[caps.groups[i]] += 1
    assert fermionic_picks(z, c, 5, caps, 1e6) == tuple(sorted(esperado[:5]))


def _treino(nomes: list[str], seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    comum = rng.normal(0, 0.01, size=(300, 1))
    return pd.DataFrame(comum + rng.normal(0, 0.015, size=(300, len(nomes))), columns=nomes,
                        index=pd.bdate_range("2020-01-01", periods=300))


def test_strategy_gives_equal_weights_on_k_names_of_the_scored_candidates():
    nomes = [f"AAA{i}3" if i < 10 else f"BB{i}3" for i in range(20)]
    setores = {nome: f"s{i % 6}" for i, nome in enumerate(nomes)}
    nota = lambda mes, candidatos: pd.Series(np.linspace(1, -1, len(candidatos)), index=candidatos)  # noqa: E731
    pesos = estrategia_fermionica(nota, setores, nomes=10, por_setor=2)(_treino(nomes))
    assert len(pesos) == 10 and pesos.to_numpy() == pytest.approx(1 / 10)
    assert pd.Series([setores[p] for p in pesos.index]).value_counts().max() <= 2


def test_strategy_takes_every_candidate_when_there_are_no_more_than_k():
    nomes = [f"{letra * 4}3" for letra in "ABCDEF"]
    setores = {nome: f"s{i}" for i, nome in enumerate(nomes)}
    nota = lambda mes, candidatos: pd.Series(np.linspace(1, -1, len(candidatos)), index=candidatos)  # noqa: E731
    pesos = estrategia_fermionica(nota, setores, nomes=10, por_setor=2)(_treino(nomes))
    assert sorted(pesos.index) == nomes and pesos.to_numpy() == pytest.approx(1 / 6)
