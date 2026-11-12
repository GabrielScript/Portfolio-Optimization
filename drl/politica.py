"""PPO's actor-critic over the shared network (DRL v3, arm ``ppo_shared``).

Stable-Baselines3 builds a policy as features -> (actor latent, critic latent) -> (action mean, value). Here the
features are the names' logits followed by the mean of their hidden vectors; the actor latent is the logits
themselves (the action mean, so SB3's linear action head is replaced by the identity) and the critic reads the mean
hidden vector, which does not depend on the order of the names either.
"""

from __future__ import annotations

import torch
from gymnasium import spaces
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch import nn

from drl.rede import HIDDEN, Scorer

VALUE_HIDDEN = 64


class SharedExtractor(BaseFeaturesExtractor):
    """Flat observation ``(batch, rows * width)`` -> ``(batch, rows + hidden)``: the logits, then the pooled names."""

    def __init__(self, observation_space: spaces.Box, width: int) -> None:
        linhas, resto = divmod(observation_space.shape[0], width)
        if resto:
            raise ValueError(f"an observation of {observation_space.shape[0]} numbers is not rows of {width}")
        super().__init__(observation_space, features_dim=linhas + HIDDEN[-1])
        self.rows = linhas
        self.scorer = Scorer(width)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        oculto = self.scorer.embed(observations.view(len(observations), self.rows, -1))
        return torch.cat([self.scorer.logits(oculto), oculto.mean(dim=1)], dim=1)


class Split(nn.Module):
    """SB3's ``mlp_extractor`` contract: the first ``rows`` features go to the actor untouched, the rest to the critic."""

    def __init__(self, rows: int, pooled: int) -> None:
        super().__init__()
        self.rows = rows
        self.latent_dim_pi, self.latent_dim_vf = rows, VALUE_HIDDEN
        self.value_body = nn.Sequential(nn.Linear(pooled, VALUE_HIDDEN), nn.Tanh())

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.forward_actor(features), self.forward_critic(features)

    def forward_actor(self, features: torch.Tensor) -> torch.Tensor:
        return features[:, :self.rows]

    def forward_critic(self, features: torch.Tensor) -> torch.Tensor:
        return self.value_body(features[:, self.rows:])


class SharedPolicy(ActorCriticPolicy):
    """``width``: numbers per row of the observation. SB3's orthogonal initialisation is off: it would overwrite the
    scorer's small last layer."""

    def __init__(self, observation_space: spaces.Box, action_space: spaces.Box, lr_schedule, *args, width: int,
                 **kwargs) -> None:
        kwargs |= {"features_extractor_class": SharedExtractor, "features_extractor_kwargs": {"width": width},
                   "ortho_init": False}
        super().__init__(observation_space, action_space, lr_schedule, *args, **kwargs)

    def _build_mlp_extractor(self) -> None:
        self.mlp_extractor = Split(self.action_space.shape[0], HIDDEN[-1])

    def _build(self, lr_schedule) -> None:
        super()._build(lr_schedule)
        self.action_net = nn.Identity()                              # the logits are the action mean
        self.optimizer = self.optimizer_class(self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs)
