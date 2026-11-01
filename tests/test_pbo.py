"""Bailey, Borwein, López de Prado and Zhu's probability of backtest overfitting, by combinatorially symmetric cross-validation."""

from itertools import combinations

import numpy as np
import pandas as pd
import pytest
from scipy.stats import rankdata


def _sharpe(linhas: np.ndarray) -> np.ndarray:
    return linhas.mean(axis=0) / linhas.std(axis=0, ddof=1)


def _pbo_a_mao(retornos: np.ndarray, blocos: int) -> float:
    """The paper's steps with loops, on the rows themselves."""
    partes = np.array_split(retornos, blocos)
    caem = 0
    total = 0
    for dentro in combinations(range(blocos), blocos // 2):
        treino = np.vstack([partes[i] for i in dentro])
        teste = np.vstack([partes[i] for i in range(blocos) if i not in dentro])
        melhor = int(np.argmax(_sharpe(treino)))
        posicao = rankdata(_sharpe(teste))[melhor] / (retornos.shape[1] + 1)
        caem += int(np.log(posicao / (1.0 - posicao)) <= 0.0)
        total += 1
    return caem / total


def _quadro(dados: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame(dados, columns=[f"A{i}" for i in range(dados.shape[1])])


def test_pbo_matches_the_papers_steps_done_by_hand():
    from core.pbo import cscv

    dados = np.random.default_rng(0).normal(0.0005, 0.01, (240, 7))
    assert cscv(_quadro(dados), blocos=8).pbo == pytest.approx(_pbo_a_mao(dados, 8))


def test_pure_noise_overfits_about_half_the_time():
    from core.pbo import cscv

    resultados = [cscv(_quadro(np.random.default_rng(semente).normal(0.0, 0.01, (480, 20))), blocos=10).pbo for semente in range(12)]
    assert np.mean(resultados) == pytest.approx(0.5, abs=0.12)


def test_a_strategy_truly_better_in_every_period_does_not_overfit():
    from core.pbo import cscv

    dados = np.random.default_rng(1).normal(0.0, 0.01, (480, 10))
    dados[:, 3] += 0.01
    resultado = cscv(_quadro(dados), blocos=10)
    assert resultado.pbo == 0.0
    assert resultado.mais_escolhida == "A3"


def test_it_counts_every_split():
    from core.pbo import cscv

    dados = np.random.default_rng(2).normal(0.0, 0.01, (120, 4))
    assert cscv(_quadro(dados), blocos=6).divisoes == 20


def test_it_refuses_an_odd_number_of_blocks_and_rows_with_gaps():
    from core.pbo import cscv

    dados = _quadro(np.random.default_rng(3).normal(0.0, 0.01, (120, 4)))
    with pytest.raises(ValueError, match="even"):
        cscv(dados, blocos=5)
    dados.iloc[3, 1] = np.nan
    with pytest.raises(ValueError, match="gap"):
        cscv(dados, blocos=6)
