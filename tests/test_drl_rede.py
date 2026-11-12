"""DRL v3: the network every name shares, the PPO policy built on it and the training through the simulator."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

from drl import agentes, direto
from drl.ambiente import EnvConfig, PortfolioEnv, observation_width, simulate
from drl.rede import Scorer
from tests.drl_fixtures import panel_of, random_prices

CORRECTED = EnvConfig(hold=5, cost_bps=10.0, observation="cross", reward="dsr_cdi", invest_all=True, dsr_warmup=252)


# --- defect 5: a policy that cannot leave equal weight ----------------------------------------------------------------


def test_the_scorer_gives_a_name_the_same_logit_wherever_it_sits() -> None:
    torch.manual_seed(0)
    scorer = Scorer(width=6)
    quadros = torch.randn(2, 9, 6)                                   # 8 names and the cash row
    troca = torch.tensor([3, 0, 7, 1, 2, 6, 5, 4])
    trocados = torch.cat([quadros[:, troca], quadros[:, -1:]], dim=1)
    logits, logits_trocados = scorer(quadros), scorer(trocados)
    assert logits.shape == (2, 9)
    assert torch.allclose(logits[:, troca], logits_trocados[:, :-1], atol=1e-6)
    assert (logits[:, -1] == 0).all()                               # the cash logit is a fixed reference


def test_the_scorer_starts_close_to_equal_weights_and_can_be_moved_far_from_them() -> None:
    torch.manual_seed(0)
    scorer = Scorer(width=6)
    quadros = torch.randn(1, 41, 6)
    assert scorer(quadros).abs().max() < 0.1
    with torch.no_grad():
        scorer.head.weight *= 3000.0
    pesos = torch.softmax(scorer(quadros)[0, :-1], dim=0)
    assert pesos.max() > 5.0 / 40.0                                 # one set of weights reaches every name's logit


# --- training through the simulator -----------------------------------------------------------------------------------


def _drifting_prices(sessions: int = 700, names: int = 8, seed: int = 0) -> pd.DataFrame:
    """Half the names drift up 0.15% a session and half down, under 1% of noise: the past return tells them apart."""
    rng = np.random.default_rng(seed)
    deriva = np.where(np.arange(names) % 2 == 0, 0.0015, -0.0015)
    retornos = rng.normal(deriva, 0.01, size=(sessions, names))
    return pd.DataFrame(100.0 * np.cumprod(1.0 + retornos, axis=0), index=pd.bdate_range("2015-01-02", periods=sessions),
                        columns=[f"N{i:02d}" for i in range(names)])


def test_the_differentiable_simulator_earns_what_the_environment_earns() -> None:
    torch.manual_seed(1)
    painel = panel_of(random_prices(sessions=420, names=6, seed=6), cdi=0.0004,
                      names=["N00", "N01", "N02", "N04", "N05"])   # one name is never tradeable
    env = PortfolioEnv(painel, np.arange(6), 260, 400, CORRECTED)
    scorer = Scorer(observation_width(CORRECTED))
    with torch.no_grad():
        scorer.head.weight *= 200.0                                 # far from equal weights, so the trades are large
    simulate(env, direto.policy_of([scorer]))
    do_ambiente = np.asarray(env.values[1:]) / np.asarray(env.values[:-1]) - 1.0 - env.rates()
    with torch.no_grad():
        da_fita = direto.rollout(scorer, direto.tape_of(env), torch.tensor([0]), decisions=28)[0].numpy()
    assert len(do_ambiente) == 140 and da_fita == pytest.approx(do_ambiente, abs=1e-6)
    assert np.abs(np.asarray(env.decisions)[:, :-1] - 0.2).max() > 0.1          # the comparison is not of equal weights


def test_the_gradient_through_the_simulator_finds_a_planted_signal() -> None:
    painel = panel_of(_drifting_prices())
    treino = PortfolioEnv(painel, np.arange(8), 260, 560, CORRECTED)
    validacao = PortfolioEnv(painel, np.arange(8), 560, 630, CORRECTED)
    teste = PortfolioEnv(painel, np.arange(8), 630, 699, CORRECTED)
    ajustado = direto.fit(treino, validacao, seed=0, params=direto.DirectParams(steps=60, batch=16, episode=12))
    simulate(teste, direto.policy_of([ajustado.scorer]))
    nos_que_sobem = np.asarray(teste.decisions)[:, [0, 2, 4, 6]].sum(axis=1)
    assert nos_que_sobem.mean() > 0.8 and 0 < ajustado.step <= 60    # equal weights would hold 0.5 there
    assert np.isfinite(ajustado.validation_sharpe)


def test_the_test_book_is_the_mean_of_the_seeds_books() -> None:
    painel = panel_of(random_prices(sessions=320, names=4, seed=7))
    env = PortfolioEnv(painel, np.arange(4), 300, 319, CORRECTED)
    scorers = []
    for semente in (0, 1):
        torch.manual_seed(semente)
        scorer = Scorer(observation_width(CORRECTED))
        with torch.no_grad():
            scorer.head.weight *= 200.0
        scorers.append(scorer)
    env.reset()
    um, outro, media = (direto.policy_of(grupo)(env) for grupo in ([scorers[0]], [scorers[1]], scorers))
    assert media == pytest.approx((um + outro) / 2.0) and not np.allclose(um, outro, atol=0.01)
    with pytest.raises(ValueError):
        direto.policy_of([])


# --- the PPO arms -----------------------------------------------------------------------------------------------------


def _agent_envs():
    painel = panel_of(random_prices(sessions=420, names=3, seed=8))

    def treino() -> PortfolioEnv:
        return PortfolioEnv(painel, np.arange(3), 260, 380, CORRECTED, episode=4)

    return treino, PortfolioEnv(painel, np.arange(3), 380, 419, CORRECTED)


def test_the_shared_policy_trains_under_ppo_and_its_action_is_the_scorers_logits() -> None:
    treino, teste = _agent_envs()
    modelo = agentes.train("ppo_shared", treino, timesteps=96, seed=0, n_envs=1)
    observacao, _ = teste.reset()
    acao, _ = modelo.predict(observacao, deterministic=True)
    scorer = modelo.policy.features_extractor.scorer
    with torch.no_grad():
        logits = scorer(torch.from_numpy(observacao).view(1, 4, 6))[0].numpy()
    assert acao == pytest.approx(logits, abs=1e-6)
    env, recompensa = agentes.evaluate(modelo, teste)
    assert len(env.values) == 40 and np.isfinite(recompensa) and env.weights[-1] == 0.0
    assert sum(p.numel() for p in modelo.policy.parameters()) < 20_000      # it does not grow with the names


def test_the_flat_ppo_policy_still_trains_on_the_corrected_environment() -> None:
    treino, teste = _agent_envs()
    modelo = agentes.train("ppo", treino, timesteps=96, seed=0, n_envs=1)
    env, recompensa = agentes.evaluate(modelo, teste)
    assert len(env.values) == 40 and np.isfinite(recompensa) and env.weights[-1] == 0.0
