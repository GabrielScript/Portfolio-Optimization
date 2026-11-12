"""The agents: Stable-Baselines3 with the papers' networks and, for PPO, P1's Table 1 hyperparameters.

What the first version got wrong, fixed here:
* PPO ran on SB3's defaults (gamma 0.99, GAE 0.95, clip 0.2, batch 64, 10 epochs, constant rate), not on P1's table.
* SAC could never run: it was given the on-policy ``vf`` head and the default one-million-transition replay buffer.
* The validation year of the protocol was never used. ``select`` trains one agent per seed and keeps the one with the
  highest validation reward, which is P1's rule; the test year is touched once, by the agent chosen.

``ppo_shared`` (v3) is PPO on the same table with ``drl.politica.SharedPolicy``: one network scores every name.

SB3 and Torch are imported inside the functions, so the tape, the environment and the baselines run without them.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from drl.ambiente import PortfolioEnv, observation_width

PPO_FAMILY = ("ppo", "ppo_shared")
ON_POLICY = (*PPO_FAMILY, "a2c")
OFF_POLICY = ("ddpg", "td3", "sac")
ALGORITHMS = ON_POLICY + OFF_POLICY
HIDDEN = (64, 64)                # P1: two layers of 64, tanh
LOG_STD_INIT = -1.0              # P1
REPLAY_BUFFER = 50_000           # SB3's default of 1,000,000 does not fit this observation in RAM


@dataclass(frozen=True)
class PPOParams:
    """P1, Table 1. ``n_envs`` is the paper's 10 where the machine allows; the rollout stays 756 steps per copy."""

    n_steps: int = 756
    batch_size: int = 1260
    n_epochs: int = 16
    gamma: float = 0.9
    gae_lambda: float = 0.9
    clip_range: float = 0.25
    learning_rate_start: float = 3e-4
    learning_rate_end: float = 1e-5


TABLE_1 = PPOParams()


def annealed(start: float, end: float) -> Callable[[float], float]:
    """SB3 schedule: ``progress`` runs from 1 (first update) to 0 (last); the rate goes linearly from start to end."""
    return lambda progress: end + (start - end) * progress


def policy_kwargs(algorithm: str, width: int | None = None) -> dict[str, Any]:
    """Network arguments by family: on-policy agents have a value head and a Gaussian to initialise; off-policy ones
    have a Q head, and only they would reject the other family's arguments. ``width`` (numbers per row of the
    observation) is what the shared policy needs to see the names in the flat vector."""
    from torch import nn

    if algorithm not in ALGORITHMS:
        raise ValueError(f"algorithm must be one of {ALGORITHMS}: {algorithm!r}")
    if algorithm == "ppo_shared":
        if width is None:
            raise ValueError("the shared policy needs the observation's width")
        return {"width": width, "log_std_init": LOG_STD_INIT}
    if algorithm in ON_POLICY:
        return {"net_arch": {"pi": list(HIDDEN), "vf": list(HIDDEN)}, "activation_fn": nn.Tanh,
                "log_std_init": LOG_STD_INIT}
    return {"net_arch": {"pi": list(HIDDEN), "qf": list(HIDDEN)}, "activation_fn": nn.Tanh}


def algorithm_kwargs(algorithm: str, rollout_cap: int, ppo: PPOParams = TABLE_1) -> dict[str, Any]:
    """Constructor arguments besides the policy. ``rollout_cap``: transitions one environment copy can give before
    the run ends, so a short smoke run still performs an update."""
    if algorithm in PPO_FAMILY:
        return {"n_steps": max(2, min(ppo.n_steps, rollout_cap)), "batch_size": ppo.batch_size,
                "n_epochs": ppo.n_epochs, "gamma": ppo.gamma, "gae_lambda": ppo.gae_lambda,
                "clip_range": ppo.clip_range,
                "learning_rate": annealed(ppo.learning_rate_start, ppo.learning_rate_end)}
    if algorithm == "a2c":
        return {}
    return {"buffer_size": REPLAY_BUFFER}


def train(algorithm: str, make_env: Callable[[], PortfolioEnv], timesteps: int, seed: int, n_envs: int = 1):
    """One trained agent. ``make_env`` builds a fresh training environment; on-policy agents get ``n_envs`` copies in
    subprocesses (rollout collection is the only part that parallelises on a small CPU), off-policy agents get one."""
    from stable_baselines3 import A2C, DDPG, PPO, SAC, TD3
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

    classes = {"ppo": PPO, "ppo_shared": PPO, "a2c": A2C, "ddpg": DDPG, "td3": TD3, "sac": SAC}
    if algorithm not in classes:
        raise ValueError(f"algorithm must be one of {ALGORITHMS}: {algorithm!r}")
    politica: Any = "MlpPolicy"
    if algorithm == "ppo_shared":
        from drl.politica import SharedPolicy
        politica = SharedPolicy
    copias = n_envs if algorithm in ON_POLICY else 1
    venv = (SubprocVecEnv([make_env] * copias) if copias > 1 else DummyVecEnv([make_env]))
    try:
        largura = observation_width(venv.get_attr("config")[0])
        modelo = classes[algorithm](politica, venv, policy_kwargs=policy_kwargs(algorithm, largura), seed=seed,
                                    verbose=0, **algorithm_kwargs(algorithm, max(2, timesteps // copias)))
        modelo.learn(total_timesteps=timesteps)
    finally:
        venv.close()
    return modelo


def evaluate(model, env: PortfolioEnv) -> tuple[PortfolioEnv, float]:
    """The deterministic policy over one episode of ``env``: the environment (curve, weights, turnover) and the
    episode's total reward."""
    observacao, _ = env.reset()
    total = 0.0
    while not env.done:
        acao, _ = model.predict(observacao, deterministic=True)
        observacao, recompensa, _, _, _ = env.step(acao)
        total += recompensa
    return env, total


@dataclass(frozen=True)
class Selection:
    """The agent kept for the test year and the validation reward of every seed tried."""

    model: Any
    seed: int
    validation_rewards: dict[int, float]


def select(seeds: Sequence[int], fit: Callable[[int], Any], validation_reward: Callable[[Any], float]) -> Selection:
    """P1's rule: one agent per seed, keep the highest validation reward (the lowest seed wins a tie)."""
    if not seeds:
        raise ValueError("at least one seed is required")
    modelos = {seed: fit(seed) for seed in seeds}
    recompensas = {seed: float(validation_reward(modelo)) for seed, modelo in modelos.items()}
    if not all(np.isfinite(valor) for valor in recompensas.values()):
        raise ValueError(f"a validation reward is not finite: {recompensas}")
    melhor = max(sorted(recompensas), key=lambda seed: recompensas[seed])
    return Selection(modelos[melhor], melhor, recompensas)
