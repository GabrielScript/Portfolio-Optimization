"""The agents, straight from Stable-Baselines3 with the papers' networks (P1: [64, 64] with tanh, log_std_init -1;
P2: the FinRL defaults of their Table 5). SB3 is imported inside the functions so the rest of the package — the data,
the environment, the baselines and their tests — runs without torch installed."""

from __future__ import annotations

from typing import Any

ALGORITMOS = ("ppo", "a2c", "ddpg", "td3", "sac")
REDE = [64, 64]
ATIVACAO = "tanh"
LOG_STD_INIT = -1.0


DETERMINISTICOS = ("ddpg", "td3")
BUFFER_OFF_POLICY = 50_000    # P2's Table 5 value; SB3's default of 1,000,000 cannot fit this observation in RAM


def politica(nome: str = "ppo") -> dict[str, Any]:
    """The MLP the five algorithms share: two hidden layers of 64 with tanh, the papers' shape.

    The kwargs are not the same for all five. PPO, A2C and SAC are stochastic: their actor emits a Gaussian and takes
    ``log_std_init`` plus a value head. DDPG and TD3 are deterministic — no standard deviation to initialise, and the
    critic head is ``qf``, not ``vf``; passing the stochastic kwargs makes SB3 raise ``TypeError``."""
    if nome not in ALGORITMOS:
        raise ValueError(f"algoritmo must be one of {ALGORITMOS}")
    if nome in DETERMINISTICOS:
        return {"net_arch": {"pi": REDE, "qf": REDE}, "activation_fn": _tanh()}
    return {"net_arch": {"pi": REDE, "vf": REDE}, "log_std_init": LOG_STD_INIT,
            "activation_fn": _tanh()}


def _tanh():
    from torch import nn

    return nn.Tanh


def treinar(nome: str, env, total_timesteps: int = 200_000, seed: int = 0, n_envs: int = 1,
            verbose: int = 0):
    """Train one agent on ``env``. On-policy algorithms (PPO, A2C) accept several parallel copies of the environment,
    which is the only cheap parallelism on a four-core CPU: the policy network itself gains nothing from more cores."""
    from stable_baselines3 import A2C, DDPG, PPO, SAC, TD3
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

    classes = {"ppo": PPO, "a2c": A2C, "ddpg": DDPG, "td3": TD3, "sac": SAC}
    if nome not in classes:
        raise ValueError(f"algoritmo must be one of {ALGORITMOS}")
    if n_envs > 1 and nome in ("ppo", "a2c"):
        venv = SubprocVecEnv([lambda: env for _ in range(n_envs)])
        modelo = classes[nome]("MlpPolicy", venv, policy_kwargs=politica(nome), seed=seed, verbose=verbose,
                               n_steps=756 if nome == "ppo" else 5)
    else:
        venv = DummyVecEnv([lambda: env])
        extra = {}
        if nome in DETERMINISTICOS:                      # off-policy: the replay buffer is the memory constraint
            extra["buffer_size"] = BUFFER_OFF_POLICY
        modelo = classes[nome]("MlpPolicy", venv, policy_kwargs=politica(nome), seed=seed, verbose=verbose, **extra)
    modelo.learn(total_timesteps=total_timesteps)
    return modelo, venv


def avaliar(modelo, env) -> dict[str, list]:
    """Run the trained policy deterministically over one full episode and return its value curve, the weights it held
    and the turnover it generated (the report's inputs)."""
    import numpy as np

    obs, _ = env.reset()
    curva, giros, pesos, valores = [env.valor], [], [], []
    terminou = False
    while not terminou:
        acao, _ = modelo.predict(obs, deterministic=True)
        obs, _, terminou, _, info = env.step(acao)
        curva.append(info["valor"])
        giros.append(info["giro"])
        valores.append(info["retorno"])
        pesos.append(env.historico_pesos[-1])
    return {"curva": np.array(curva), "giros": np.array(giros), "retornos": np.array(valores), "pesos": pesos}
