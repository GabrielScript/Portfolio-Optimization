"""MARS on the B3: the environment's accounting (cash earns the CDI, holdings follow prices, a delisting is sold)."""

from __future__ import annotations

import numpy as np
import pytest

from qpo_kaggle.mars import Carteira, Execucao, Painel, passo, trocar_nomes

CFG = Execucao()


def _painel(precos: list[list[float]], cdi: float = 0.0) -> Painel:
    tabela = np.array(precos, dtype=float)
    sessoes, nomes = tabela.shape
    return Painel(np.array([f"N{i}" for i in range(nomes)]), np.arange(sessoes), tabela,
                  np.zeros((sessoes, nomes, 5), np.float32), np.full(sessoes, cdi), np.array([0, 0, 0, sessoes]))


def test_cash_earns_the_next_sessions_cdi():
    carteira = Carteira.em_caixa(1)
    recompensa, *_ = passo(_painel([[10.0], [10.0]], cdi=0.001), carteira, 0, np.zeros(1), CFG)
    assert carteira.valor == pytest.approx(1.001)
    assert recompensa == pytest.approx(0.001)


def test_a_holding_follows_its_price():
    carteira = Carteira(np.array([0.5]), 0.5, [1.0])
    passo(_painel([[10.0], [11.0]]), carteira, 0, np.zeros(1), CFG)
    assert carteira.valor == pytest.approx(1.05)


def test_a_name_whose_price_ends_is_sold_the_session_before():
    painel = _painel([[10.0, 5.0], [10.0, 5.0], [10.0, np.nan]])
    carteira = Carteira(np.array([0.0, 0.3]), 0.7, [1.0])
    passo(painel, carteira, 1, np.zeros(2), CFG)
    assert carteira.x[1] == 0.0


def test_at_the_year_change_a_name_that_stays_keeps_its_value_and_one_that_leaves_is_sold():
    carteira = Carteira(np.array([0.3, 0.2]), 0.5, [1.0])
    nova = trocar_nomes(carteira, np.array(["A", "B"]), np.array(["B", "C"]), CFG)
    assert list(nova.x) == [0.2, 0.0]
    assert nova.caixa == pytest.approx(0.5 + 0.3 * (1 - CFG.custo))


def test_the_precomputed_covariance_matches_numpys_over_the_last_60_sessions():
    rng = np.random.default_rng(3)
    painel = _painel(list(np.exp(np.cumsum(rng.normal(0, 0.02, (90, 3)), axis=0))))
    assert np.allclose(painel.cov(80), np.cov(painel.retornos[21:81].T))
