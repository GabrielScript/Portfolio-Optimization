"""Fermionic and entropic selection: the swap ascent, the caps, score ties and the strategy."""

import itertools

import numpy as np
import pandas as pd
import pytest

from core.diversidade import log_det, von_neumann
from core.fermionica import THETA, covariance_correlation, estrategia_fermionica, fermionic_picks, objective
from core.qubo_universo import GroupCaps
from core.sinais import carteira_da_nota

FUNCIONAIS = [log_det, von_neumann]


def _corr(n: int, seed: int) -> np.ndarray:
    fatores = np.random.default_rng(seed).normal(size=(4 * n, n))
    fatores[:, 1:] += 0.8 * fatores[:, [0]]
    return np.corrcoef(fatores, rowvar=False)


def _within_caps(picks, caps: GroupCaps) -> bool:
    return max(pd.Series([caps.groups[i] for i in picks]).value_counts()) <= caps.max_per_group


def test_objective_of_a_pair_is_the_scores_plus_the_diversity():
    c = np.array([[1.0, 0.6], [0.6, 1.0]])
    z = np.array([0.5, -0.2])
    assert objective(z, c, (0, 1), 1.0) == pytest.approx(2 * 0.3 + np.log(1 - 0.36))
    assert objective(z, c, (0, 1), 3.0) == pytest.approx(6 * 0.3 + np.log(1 - 0.36))
    assert objective(z, c, (0, 1), 1.0, von_neumann) == pytest.approx(0.6 - 1.6 * np.log(1.6) - 0.4 * np.log(0.4))


def test_a_near_duplicate_is_left_out_for_a_slightly_lower_independent_name():
    c = np.eye(4)
    c[0, 1] = c[1, 0] = 0.95
    z = np.array([1.0, 0.9, 0.5, -1.0])
    assert fermionic_picks(z, c, 2, None, THETA) == (0, 2)      # 2(1.5) + 0 > 2(1.9) + log(1 - 0.9025)


@pytest.mark.parametrize("funcional", FUNCIONAIS)
def test_picks_match_exhaustive_search_on_small_instances(funcional):
    for seed in range(6):
        c, z = _corr(9, seed), np.random.default_rng(100 + seed).normal(size=9)
        melhor = max(itertools.combinations(range(9), 4), key=lambda s: objective(z, c, s, THETA, funcional))
        assert objective(z, c, fermionic_picks(z, c, 4, None, THETA, funcional), THETA, funcional) == pytest.approx(
            objective(z, c, melhor, THETA, funcional))


@pytest.mark.parametrize("funcional", FUNCIONAIS)
def test_picks_match_exhaustive_search_under_caps(funcional):
    caps = GroupCaps(tuple("abc"[i % 3] for i in range(10)), 2)
    for seed in range(6):
        c, z = _corr(10, seed), np.random.default_rng(200 + seed).normal(size=10)
        viaveis = [s for s in itertools.combinations(range(10), 5) if _within_caps(s, caps)]
        melhor = max(viaveis, key=lambda s: objective(z, c, s, 0.25, funcional))
        picks = fermionic_picks(z, c, 5, caps, 0.25, funcional)
        assert _within_caps(picks, caps)
        assert objective(z, c, picks, 0.25, funcional) == pytest.approx(objective(z, c, melhor, 0.25, funcional))


@pytest.mark.parametrize("funcional", FUNCIONAIS)
def test_no_single_swap_improves_the_picks_and_the_caps_hold(funcional):
    c, z = _corr(30, 7), np.random.default_rng(8).normal(size=30)
    caps = GroupCaps(tuple("abcde"[i % 5] for i in range(30)), 2)
    picks = fermionic_picks(z, c, 8, caps, THETA, funcional)
    assert len(picks) == 8 and _within_caps(picks, caps)
    atual = objective(z, c, picks, THETA, funcional)
    for sai, entra in itertools.product(picks, set(range(30)) - set(picks)):
        troca = tuple(sorted(set(picks) - {sai} | {entra}))
        if _within_caps(troca, caps):
            assert objective(z, c, troca, THETA, funcional) <= atual + 1e-7


def test_a_huge_theta_is_the_ranking_by_score_under_the_caps():
    c, z = _corr(20, 3), np.random.default_rng(4).normal(size=20)
    caps = GroupCaps(tuple("ab"[i % 2] for i in range(20)), 3)
    esperado, contagem = [], {"a": 0, "b": 0}
    for i in np.argsort(-z, kind="stable"):
        if contagem[caps.groups[i]] < 3:
            esperado.append(int(i))
            contagem[caps.groups[i]] += 1
    assert fermionic_picks(z, c, 5, caps, 1e6) == tuple(sorted(esperado[:5]))


def test_exact_score_ties_at_a_huge_theta_end_on_the_same_total_score():
    c = _corr(12, 9)
    z = np.repeat([2.0, 1.0, 0.0, -1.0], 3)                    # ties inside every block of three
    picks = fermionic_picks(z, c, 5, None, 1e6)
    assert z[list(picks)].sum() == pytest.approx(2.0 * 3 + 1.0 * 2)


def _treino(nomes: list[str], seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    comum = rng.normal(0, 0.01, size=(300, 1))
    return pd.DataFrame(comum + rng.normal(0, 0.015, size=(300, len(nomes))), columns=nomes,
                        index=pd.bdate_range("2020-01-01", periods=300))


def _descending(mes, candidatos):
    return pd.Series(np.linspace(1, -1, len(candidatos)), index=candidatos)


@pytest.mark.parametrize("funcional", FUNCIONAIS)
def test_strategy_gives_equal_weights_on_k_names_of_the_scored_candidates(funcional):
    nomes = [f"AAA{i}3" if i < 10 else f"BB{i}3" for i in range(20)]
    setores = {nome: f"s{i % 6}" for i, nome in enumerate(nomes)}
    registro = {}
    pesos = estrategia_fermionica(_descending, setores, nomes=10, por_setor=2, diversity=funcional,
                                  registro=registro)(_treino(nomes))
    assert len(pesos) == 10 and pesos.to_numpy() == pytest.approx(1 / 10)
    assert pd.Series([setores[p] for p in pesos.index]).value_counts().max() <= 2
    (candidatos, picks, z, corr), = registro.values()
    assert [candidatos[i] for i in picks] == list(pesos.index)
    assert z.mean() == pytest.approx(0.0) and z.std() == pytest.approx(1.0) and np.diag(corr) == pytest.approx(1.0)


def test_strategy_with_the_first_reading_correlation_still_runs():
    nomes = [f"AAA{i}3" if i < 10 else f"BB{i}3" for i in range(20)]
    setores = {nome: f"s{i % 6}" for i, nome in enumerate(nomes)}
    pesos = estrategia_fermionica(_descending, setores, nomes=10, por_setor=2,
                                  correlation=covariance_correlation)(_treino(nomes))
    assert len(pesos) == 10


def test_strategy_with_few_candidates_is_the_ranking_under_the_cap():
    nomes = [f"{letra * 4}3" for letra in "ABCDEFGH"]
    setores = dict(zip(nomes, ["a", "a", "a", "b", "b", "c", "d", "e"]))
    pesos = estrategia_fermionica(_descending, setores, nomes=10, por_setor=2)(_treino(nomes))
    esperado = carteira_da_nota(_descending(None, nomes), setores, 10, 2)
    assert list(pesos.index) == list(esperado.index) and len(pesos) == 7        # the 3rd name of sector a is out
    assert pesos.to_numpy() == pytest.approx(1 / 7)


def test_strategy_when_the_caps_admit_fewer_than_k_is_the_ranking_under_the_cap():
    nomes = [f"{letra * 4}3" for letra in "ABCDEFGHIJKL"]
    setores = {nome: "abcd"[i % 4] for i, nome in enumerate(nomes)}           # 4 sectors x 3 names, 2 per sector
    pesos = estrategia_fermionica(_descending, setores, nomes=10, por_setor=2)(_treino(nomes))
    assert list(pesos.index) == list(carteira_da_nota(_descending(None, nomes), setores, 10, 2).index)
    assert len(pesos) == 8
