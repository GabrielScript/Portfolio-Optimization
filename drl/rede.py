"""The network every name shares (DRL v3).

P1's policy is one flat MLP from about 900 inputs to about 151 logits. Each logit has its own output weights, a
window gives about 60 distinct states to fit them, and nothing learned about one name carries to another: the softmax
never left 1/n. Here one small network scores a name from that name's own row and the cash row (the market's state
and the cash held), and the same weights score every name. What is learned on one name is known for all, the policy
does not depend on the order or on the number of names, and a window's five years give names x decisions examples.

Torch is imported here: the tape, the environment and the rules do not import this module.
"""

from __future__ import annotations

import torch
from torch import nn

HIDDEN = (64, 64)                # P1's two layers of 64, tanh
HEAD_GAIN = 0.01                 # a small last layer: the first policy is close to equal weights


class Scorer(nn.Module):
    """Frames ``(batch, n + 1, width)``, the last row being cash, to logits ``(batch, n + 1)``; the cash logit is 0."""

    def __init__(self, width: int, hidden: tuple[int, ...] = HIDDEN) -> None:
        super().__init__()
        camadas: list[nn.Module] = []
        entrada = 2 * width                                          # the name's row and the cash row
        for saida in hidden:
            camadas += [nn.Linear(entrada, saida), nn.Tanh()]
            entrada = saida
        self.body = nn.Sequential(*camadas)
        self.head = nn.Linear(entrada, 1)
        nn.init.orthogonal_(self.head.weight, HEAD_GAIN)
        nn.init.zeros_(self.head.bias)
        self.width, self.hidden = width, entrada

    def embed(self, frames: torch.Tensor) -> torch.Tensor:
        # frames: (batch, n + 1, width) -> (batch, n, hidden)
        nomes = frames[:, :-1]
        contexto = frames[:, -1:].expand(-1, nomes.shape[1], -1)
        return self.body(torch.cat([nomes, contexto], dim=-1))

    def logits(self, hidden: torch.Tensor) -> torch.Tensor:
        # hidden: (batch, n, hidden) -> (batch, n + 1)
        por_nome = self.head(hidden).squeeze(-1)
        return torch.cat([por_nome, por_nome.new_zeros(len(por_nome), 1)], dim=1)

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        return self.logits(self.embed(frames))
