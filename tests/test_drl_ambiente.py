"""The DRL simulator: timing, cost, drift, cash, eligibility, rewards and the absence of look-ahead."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from drl.ambiente import ETA, EnvConfig, PortfolioEnv, differential_sharpe, simulate, softmax
from tests.drl_fixtures import panel_of, random_prices


def _env(precos: pd.DataFrame, start: int = 0, stop: int | None = None, cdi: float = 0.0, **config) -> PortfolioEnv:
    painel = panel_of(precos, cdi)
    return PortfolioEnv(painel, np.arange(precos.shape[1]), start, len(precos) - 1 if stop is None else stop,
                        EnvConfig(**{"hold": 1, "cost_bps": 0.0, "lookback": 5, **config}))


def _two_names() -> pd.DataFrame:
    datas = pd.bdate_range("2020-01-01", periods=5)
    return pd.DataFrame({"A": [10.0, 11.0, 11.0, 12.1, 12.1], "B": [20.0, 20.0, 18.0, 18.0, 18.0]}, index=datas)


def test_all_cash_earns_the_cdi_and_pays_no_cost() -> None:
    env = _env(_two_names(), cdi=0.001, cost_bps=10.0)
    env.reset()
    env.apply(np.array([0.0, 0.0, 1.0]))
    assert env.value == pytest.approx(1.001) and env.turnover == [0.0]


def test_cash_earns_nothing_when_the_papers_convention_is_asked_for() -> None:
    env = _env(_two_names(), cdi=0.001, cash_earns_cdi=False)
    env.reset()
    env.apply(np.array([0.0, 0.0, 1.0]))
    assert env.value == pytest.approx(1.0)


def test_the_book_earns_the_weighted_return_of_the_next_session() -> None:
    env = _env(_two_names())
    env.reset()
    env.apply(np.array([0.5, 0.5, 0.0]))
    assert env.value == pytest.approx(1.05)                         # A +10%, B flat


def test_the_cost_is_charged_on_the_notional_traded_against_the_drifted_book() -> None:
    env = _env(_two_names(), cost_bps=100.0)
    env.reset()
    env.apply(np.array([0.5, 0.5, 0.0]))                            # from cash: turnover 1
    assert env.value == pytest.approx(1.05 - 0.01)
    deriva = env.weights.copy()
    assert deriva[:2] == pytest.approx([0.55 / 1.05, 0.5 / 1.05])
    env.apply(deriva)                                               # holding the drifted book trades nothing
    assert env.turnover[1] == pytest.approx(0.0)


def test_one_step_holds_the_book_for_the_whole_period_without_trading() -> None:
    precos = random_prices(sessions=60, names=3)
    diario = _env(precos, hold=1, cost_bps=10.0)
    mensal = _env(precos, hold=21, cost_bps=10.0)
    diario.reset(), mensal.reset()
    alvo = np.array([0.5, 0.3, 0.2, 0.0])
    recompensa = mensal.apply(alvo)
    soma = diario.apply(alvo)
    for _ in range(20):
        soma += diario.apply(diario.weights)                        # buy and hold, session by session
    assert mensal.t == 21 and len(mensal.turnover) == 1
    assert mensal.value == pytest.approx(diario.value)
    assert recompensa == pytest.approx(soma)


def test_the_last_hold_stops_at_the_end_of_the_stretch() -> None:
    env = _env(random_prices(sessions=30, names=2), start=0, stop=25, hold=21)
    simulate(env, lambda e: np.array([0.5, 0.5, 0.0]))
    assert env.t == 25 and len(env.values) == 26 and len(env.turnover) == 2


def test_an_action_never_puts_weight_on_a_name_that_is_not_tradeable() -> None:
    precos = random_prices(sessions=40, names=3)
    painel = panel_of(precos, names=["N00", "N02"])                 # N01 is not on the list
    env = PortfolioEnv(painel, np.arange(3), 0, 39, EnvConfig(hold=1))
    env.reset()
    pesos = env.project(np.array([0.0, 9.0, 0.0, 0.0]))
    assert pesos[1] == 0.0 and pesos.sum() == pytest.approx(1.0)
    with pytest.raises(ValueError, match="not tradeable"):
        env.apply(np.array([0.0, 1.0, 0.0, 0.0]))


def test_a_target_off_the_simplex_is_refused() -> None:
    env = _env(_two_names())
    env.reset()
    with pytest.raises(ValueError):
        env.apply(np.array([0.7, 0.7, 0.0]))


def test_the_differential_sharpe_matches_the_recursion_by_hand() -> None:
    a = b = 0.0
    valores = []
    for retorno in (0.01, -0.02, 0.015):
        valor, a, b = differential_sharpe(retorno, a, b)
        valores.append(valor)
    a1, b1 = ETA * 0.01, ETA * 0.01 ** 2
    esperado = (b1 * (-0.02 - a1) - 0.5 * a1 * ((-0.02) ** 2 - b1)) / (b1 - a1 ** 2) ** 1.5
    assert valores[0] == 0.0 and valores[1] == pytest.approx(esperado)


def test_the_drawdown_reward_subtracts_the_distance_from_the_peak() -> None:
    simples, com_queda = _env(_two_names(), reward="dsr"), _env(_two_names(), reward="dsr_drawdown")
    for env in (simples, com_queda):
        env.reset()
        env.apply(np.array([0.0, 1.0, 0.0]))                        # B: flat
    r_simples, r_queda = simples.apply(np.array([0.0, 1.0, 0.0])), com_queda.apply(np.array([0.0, 1.0, 0.0]))  # -10%
    assert r_queda == pytest.approx(r_simples - 0.1)


@pytest.mark.parametrize("observation", ["matrix", "compact"])
def test_the_observation_does_not_move_when_the_return_about_to_be_earned_changes(observation: str) -> None:
    precos = random_prices(sessions=300, names=4)
    mexido = precos.copy()
    mexido.iloc[201:] *= 1.5                                        # the close of row 201 on: the future of row 200
    a, b = (_env(p, start=200, lookback=20, observation=observation) for p in (precos, mexido))
    assert np.array_equal(a.reset()[0], b.reset()[0])
    passado = precos.copy()
    passado.iloc[200:] *= 1.5                                       # the close of row 200 itself: already known
    assert not np.array_equal(a.reset()[0], _env(passado, start=200, lookback=20, observation=observation).reset()[0])


def test_the_matrix_observation_puts_the_newest_return_first() -> None:
    env = _env(_two_names(), start=3, stop=4, lookback=4)
    quadro = env.reset()[0].reshape(3, 4)
    assert quadro[0, 1:] == pytest.approx(np.log([12.1 / 11.0, 1.0, 1.1]))     # r(t-1), r(t-2), r(t-3) of A
    assert quadro[:, 0] == pytest.approx([0.0, 0.0, 1.0])                       # the weights: all cash


def test_softmax_gives_zero_to_minus_infinity() -> None:
    assert softmax(np.array([0.0, -np.inf, 0.0])) == pytest.approx([0.5, 0.0, 0.5])


def test_consecutive_stretches_earn_every_session_exactly_once() -> None:
    precos = random_prices(sessions=120, names=1)
    painel = panel_of(precos)
    tudo = PortfolioEnv(painel, np.arange(1), 10, 110, EnvConfig(hold=7, cost_bps=0.0))
    partes = [PortfolioEnv(painel, np.arange(1), a, b, EnvConfig(hold=7, cost_bps=0.0)) for a, b in ((10, 60), (60, 110))]
    for env in (tudo, *partes):
        simulate(env, lambda e: np.array([1.0, 0.0]))
    assert tudo.value == pytest.approx(partes[0].value * partes[1].value)
    assert tudo.value == pytest.approx(precos.iloc[110, 0] / precos.iloc[10, 0])


def test_an_invalid_configuration_or_stretch_is_refused() -> None:
    with pytest.raises(ValueError):
        EnvConfig(reward="profit")
    with pytest.raises(ValueError):
        EnvConfig(observation="image")
    with pytest.raises(ValueError):
        _env(_two_names(), start=0, stop=5)                         # the last session earns nothing
