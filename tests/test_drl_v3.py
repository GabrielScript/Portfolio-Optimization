"""DRL v3 (docs/superpowers/specs/2026-11-12-drl-v3-design.md): what the second review changed in the environment."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from drl.ambiente import EnvConfig, PortfolioEnv
from tests.drl_fixtures import panel_of, random_prices


def _env(precos: pd.DataFrame, start: int = 0, stop: int | None = None, cdi: float = 0.0, history: int = 1,
         **config) -> PortfolioEnv:
    painel = panel_of(precos, cdi, history)
    return PortfolioEnv(painel, np.arange(precos.shape[1]), start, len(precos) - 1 if stop is None else stop,
                        EnvConfig(**{"hold": 1, "cost_bps": 0.0, "lookback": 5, **config}))


def _all_cash(env: PortfolioEnv) -> np.ndarray:
    return np.r_[np.zeros(env.n), 1.0]


# --- defect 1: the reward's optimum was 100% cash ---------------------------------------------------------------------


def test_the_raw_differential_sharpe_pays_a_book_that_only_holds_cash() -> None:
    env = _env(random_prices(sessions=400, names=3), start=300, cdi=0.0004, reward="dsr", dsr_warmup=0)
    env.reset()
    recompensas = [env.apply(_all_cash(env)) for _ in range(40)]
    assert sum(recompensas[2:]) > 0.0                               # the defect, kept reproducible under "dsr"


def test_the_reward_over_the_cdi_pays_cash_nothing() -> None:
    env = _env(random_prices(sessions=400, names=3), start=300, cdi=0.0004, reward="dsr_cdi")
    env.reset()
    assert [env.apply(_all_cash(env)) for _ in range(40)] == pytest.approx([0.0] * 40, abs=1e-9)


# --- defect 2: the reward's cold start --------------------------------------------------------------------------------


def _first_and_late_reward(**config) -> tuple[float, float]:
    """The reward of one session of the same size (+2 standard deviations) on the first row and 200 sessions later."""
    precos = random_prices(sessions=900, names=4, seed=1)
    retornos = precos.pct_change().to_numpy()
    desvio = retornos[1:].mean(axis=1).std()
    for linha in (501, 701):                                        # the sessions earned from rows 500 and 700
        precos.iloc[linha:] *= (1.0 + 2.0 * desvio) / (1.0 + retornos[linha].mean())
    env = _env(precos, start=500, reward="dsr_cdi", **config)
    env.reset()
    iguais = np.r_[np.full(4, 0.25), 0.0]
    recompensas = [env.apply(iguais) for _ in range(201)]
    return recompensas[0], recompensas[200]


def test_a_cold_start_does_not_reward_the_first_session_at_all() -> None:
    primeira, tardia = _first_and_late_reward(dsr_warmup=0)
    assert primeira == 0.0 and tardia > 0.0                         # no variance estimate yet: the ratio is undefined


def test_a_warm_start_gives_the_first_session_the_scale_of_a_late_one() -> None:
    primeira, tardia = _first_and_late_reward(dsr_warmup=252)
    assert 0.5 < primeira / tardia < 2.0


def test_the_warm_start_reads_only_sessions_before_the_first_row() -> None:
    precos = random_prices(sessions=600, names=3, seed=2)
    mexido = precos.copy()
    mexido.iloc[401:] *= 1.3
    a, b = (_env(p, start=400, reward="dsr_cdi", dsr_warmup=252) for p in (precos, mexido))
    assert (a._a, a._b) == (b._a, b._b) and a._b > 0.0


# --- defects 3 and 4: one replayed path, and the end of the tape as a terminal state ----------------------------------


def test_the_end_of_a_stretch_is_a_truncation_not_a_terminal_state() -> None:
    env = _env(random_prices(sessions=30, names=2), hold=5)
    env.reset()
    sinais = [env.step(np.zeros(3))[2:4] for _ in range(6)]
    assert sinais[:-1] == [(False, False)] * 5 and sinais[-1] == (False, True)


def test_a_training_episode_starts_on_a_random_row_and_lasts_its_decisions() -> None:
    painel = panel_of(random_prices(sessions=400, names=2))
    env = PortfolioEnv(painel, np.arange(2), 100, 399, EnvConfig(hold=5, cost_bps=0.0), episode=12)
    primeiras = set()
    for semente in range(40):
        env.reset(seed=semente)
        primeiras.add(env.t)
        passos = 0
        while not env.done:
            env.step(np.zeros(3))
            passos += 1
        assert passos == 12 and env.t <= 399
    assert len(primeiras) > 20 and min(primeiras) >= 100 and max(primeiras) <= 399 - 12 * 5
    with pytest.raises(ValueError):
        PortfolioEnv(painel, np.arange(2), 100, 120, EnvConfig(hold=5), episode=12)   # does not fit the stretch


def test_an_evaluation_stretch_still_starts_on_its_first_row() -> None:
    env = _env(random_prices(sessions=60, names=2), start=10)
    env.reset(seed=7)
    assert env.t == 10


# --- fully invested, and the names a stretch may trade ----------------------------------------------------------------


def test_a_fully_invested_book_holds_cash_only_when_nothing_is_tradeable() -> None:
    precos = random_prices(sessions=40, names=3)
    env = _env(precos, start=5, invest_all=True)
    pesos = env.project(np.array([0.0, 0.0, 0.0, 9.0]))            # the cash logit is ignored
    assert pesos[-1] == 0.0 and pesos[:-1] == pytest.approx(1.0 / 3.0)
    vazio = _env(precos, start=5, history=500, invest_all=True)    # no name has 500 closes: nothing is tradeable
    assert vazio.project(np.zeros(4)) == pytest.approx([0.0, 0.0, 0.0, 1.0])
    assert _env(precos, start=5).project(np.array([0.0, 0.0, 0.0, 9.0]))[-1] > 0.99   # the papers' action keeps cash


def test_a_stretch_trades_only_the_names_it_is_allowed() -> None:
    painel = panel_of(random_prices(sessions=40, names=3))
    env = PortfolioEnv(painel, np.arange(3), 5, 39, EnvConfig(hold=1, invest_all=True),
                       allowed=np.array([True, False, True]))
    assert env.tradeable().tolist() == [True, False, True]
    assert env.project(np.zeros(4)) == pytest.approx([0.5, 0.0, 0.5, 0.0])
    with pytest.raises(ValueError):
        PortfolioEnv(painel, np.arange(3), 5, 39, EnvConfig(), allowed=np.array([True, False]))


# --- defect 6: raw inputs ---------------------------------------------------------------------------------------------


def _cross_env(precos: pd.DataFrame, start: int, names: list[str] | None = None, scores: np.ndarray | None = None,
               observation: str = "cross") -> PortfolioEnv:
    from dataclasses import replace

    painel = panel_of(precos, names=names)
    if scores is not None:
        painel = replace(painel, scores=scores)
    return PortfolioEnv(painel, np.arange(precos.shape[1]), start, len(precos) - 1,
                        EnvConfig(hold=1, cost_bps=0.0, observation=observation, invest_all=True))


def test_the_cross_observation_standardizes_each_input_across_the_tradeable_names() -> None:
    precos = random_prices(sessions=320, names=6, seed=3)
    precos.iloc[300] = precos.iloc[299] * [1.0, 1.0, 1.0, 1.0, 1.0, 5.0]      # one name jumps 400% on the last session
    env = _cross_env(precos, start=300, names=["N00", "N01", "N02", "N03", "N05"])
    quadro = env.observe().reshape(7, 6)
    negociaveis = quadro[[0, 1, 2, 3, 5]]
    assert quadro[:, 0] == pytest.approx([0, 0, 0, 0, 0, 0, 1])                # the weights are passed as they are
    assert (quadro[4] == 0.0).all()                                            # not tradeable: no inputs
    assert negociaveis[:, 1:].mean(axis=0) == pytest.approx(np.zeros(5), abs=1e-6)
    assert negociaveis[:, 1:].std(axis=0) == pytest.approx(np.ones(5), abs=1e-6)
    assert quadro[5, 1] == pytest.approx(2.0, abs=0.02)             # one outlier among five names cannot exceed 2
    assert quadro[6, 3:].tolist() == [0.0, 0.0, 0.0]                           # the cash row: weight and the market


def test_the_cross_observation_reads_nothing_after_the_decision() -> None:
    precos = random_prices(sessions=320, names=5, seed=4)
    mexido = precos.copy()
    mexido.iloc[301:] *= np.linspace(1.2, 2.0, 5)
    assert _cross_env(precos, 300).observe() == pytest.approx(_cross_env(mexido, 300).observe())


def test_the_score_enters_as_one_more_standardized_input_and_a_missing_score_as_zero() -> None:
    precos = random_prices(sessions=320, names=4, seed=5)
    notas = np.tile([0.1, 0.5, np.nan, 0.9], (320, 1))
    quadro = _cross_env(precos, 300, scores=notas, observation="cross_score").observe().reshape(5, 7)
    assert quadro[:4, 6] == pytest.approx([-1.2247449, 0.0, 0.0, 1.2247449])
    with pytest.raises(ValueError, match="scores"):
        _cross_env(precos, 300, observation="cross_score")
