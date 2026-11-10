"""The per-algorithm policy kwargs. These run without Stable-Baselines3 — only a network shape is inspected — but they
do need torch, so the module skips where it is absent (the project venv), and runs in `.venv-drl`."""

from __future__ import annotations

import pytest

pytest.importorskip("torch")

from drl import agentes


def test_a_deterministic_policy_gets_no_log_std_init():
    """DDPG and TD3 are deterministic: their policy has no standard deviation, and SB3 rejects the keyword."""
    for nome in ("ddpg", "td3"):
        kwargs = agentes.politica(nome)
        assert "log_std_init" not in kwargs


def test_a_stochastic_policy_keeps_the_papers_log_std_init():
    for nome in ("ppo", "a2c", "sac"):
        assert agentes.politica(nome)["log_std_init"] == agentes.LOG_STD_INIT


def test_a_deterministic_policy_takes_the_critic_head_not_a_value_head():
    """TD3Policy expects ``qf``; a ``vf`` key is not part of its signature."""
    kwargs = agentes.politica("ddpg")
    assert kwargs["net_arch"] == {"pi": agentes.REDE, "qf": agentes.REDE}
    assert kwargs["activation_fn"].__name__ == "Tanh"


def test_a_stochastic_policy_keeps_the_actor_and_value_heads():
    assert agentes.politica("ppo")["net_arch"] == {"pi": agentes.REDE, "vf": agentes.REDE}


def test_an_unknown_algorithm_is_rejected():
    with pytest.raises(ValueError):
        agentes.politica("reinforce")
