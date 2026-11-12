"""Training through the simulator (DRL v3, arms ``direct`` and ``direct_score``).

The agent's trades do not move prices, so the portfolio's return is an explicit, differentiable function of the
weights it chooses. PPO estimates the gradient of that function by adding noise to about 150 logits and watching the
reward; here it is computed exactly (Moody and Saffell's direct reinforcement; the "deterministic policy gradient" of
Jiang, Xu and Liang's EIIE). ``rollout`` is ``PortfolioEnv.apply`` written in Torch for a batch of episodes, and a
test holds the two to the same returns. Evaluation never uses it: validation and test run in the environment itself.

The loss is minus the mean annualized Sharpe ratio over the CDI of the episodes: the differential Sharpe ratio of the
PPO arms is the online derivative of this same quantity.
"""

from __future__ import annotations

import copy
import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import torch

from drl.ambiente import LOGIT_BOUND, Policy, PortfolioEnv, observation_width, simulate
from drl.rede import Scorer
from drl.relatorio import SESSIONS, metrics

EPSILON = 1e-8                   # keeps the ratio finite for a book whose excess return does not move
MAX_GRADIENT_NORM = 1.0


@dataclass(frozen=True)
class DirectParams:
    """Fixed by the pre-registration (docs/superpowers/specs/2026-11-12-drl-v3-design.md)."""

    steps: int = 300
    batch: int = 32
    learning_rate: float = 1e-3
    check_every: int = 10
    episode: int = 12            # decisions per training episode, the length of a test year at ``hold`` 21


PRE_REGISTERED = DirectParams()


@dataclass(frozen=True)
class Tape:
    """One stretch as tensors; row 0 is the stretch's first row."""

    frames: torch.Tensor         # (rows, n + 1, width) float32, the weights column at 0
    tradeable: torch.Tensor      # (rows, n) bool
    growth: torch.Tensor         # (rows, n + 1) float64: 1 + the return of each name and of cash to the next row
    rate: torch.Tensor           # (rows,) float64: the CDI to the next row
    hold: int
    cost: float                  # per unit of notional traded


def tape_of(env: PortfolioEnv) -> Tape:
    if not env.config.invest_all:
        raise ValueError("the differentiable simulator is written for a fully invested book (invest_all)")
    trecho = env.arrays()
    return Tape(torch.tensor(trecho.frames, dtype=torch.float32), torch.tensor(trecho.tradeable),
                torch.tensor(1.0 + np.column_stack([trecho.forward, trecho.cash])), torch.tensor(trecho.rate),
                env.config.hold, env.config.cost_bps / 1e4)


def rollout(scorer: Scorer, tape: Tape, starts: torch.Tensor, decisions: int) -> torch.Tensor:
    """Episodes from cash at rows ``starts`` (batch,) of the tape: the session returns over the CDI,
    ``(batch, decisions * hold)``, differentiable in the scorer's parameters. Mirrors ``PortfolioEnv.apply``."""
    lote, ativos = len(starts), tape.growth.shape[1]
    pesos = torch.zeros(lote, ativos, dtype=torch.float64)
    pesos[:, -1] = 1.0
    passos = torch.arange(tape.hold)
    excessos = []
    for decisao in range(decisions):
        t = starts + decisao * tape.hold
        quadros = torch.cat([pesos.unsqueeze(-1).float(), tape.frames[t][:, :, 1:]], dim=-1)
        negociavel = tape.tradeable[t]
        mascara = torch.cat([negociavel, ~negociavel.any(dim=1, keepdim=True)], dim=1)
        logits = scorer(quadros).double().clamp(-LOGIT_BOUND, LOGIT_BOUND)
        alvo = torch.softmax(logits.masked_fill(~mascara, -math.inf), dim=1)
        custo = tape.cost * (alvo[:, :-1] - pesos[:, :-1]).abs().sum(dim=1)
        linhas = t[:, None] + passos                                             # (batch, hold)
        acumulado = torch.cumprod(tape.growth[linhas], dim=1)                    # (batch, hold, n + 1)
        valor = (alvo[:, None, :] * acumulado).sum(dim=-1)                       # (batch, hold), from 1 at the trade
        anterior = torch.cat([torch.ones(lote, 1, dtype=torch.float64), valor[:, :-1]], dim=1)
        retorno = valor / anterior - 1.0
        retorno = torch.cat([retorno[:, :1] - custo[:, None], retorno[:, 1:]], dim=1)
        excessos.append(retorno - tape.rate[linhas])
        pesos = alvo * acumulado[:, -1] / valor[:, -1:]
    return torch.cat(excessos, dim=1)


def sharpe(excess: torch.Tensor) -> torch.Tensor:
    """(batch, sessions) session returns over the CDI -> (batch,) annualized Sharpe ratios."""
    return excess.mean(dim=1) / (excess.std(dim=1) + EPSILON) * math.sqrt(SESSIONS)


def policy_of(scorers: Sequence[Scorer]) -> Policy:
    """The rule that holds the mean of the scorers' books: each scorer's logits go through the environment's own
    projection, so the mean is on the simplex and holds nothing that is not tradeable."""
    if not scorers:
        raise ValueError("at least one scorer is required")

    def policy(env: PortfolioEnv) -> np.ndarray:
        quadro = torch.from_numpy(env.observe()).view(1, env.n + 1, -1)
        with torch.no_grad():
            livros = [env.project(scorer(quadro)[0].numpy()) for scorer in scorers]
        return np.mean(livros, axis=0)
    return policy


def stretch_sharpe(env: PortfolioEnv, policy: Policy) -> float:
    """The Sharpe ratio over the CDI of ``policy`` run once over ``env``'s stretch, from cash."""
    simulate(env, policy)
    return metrics(env.values, env.rates()[:len(env.values) - 1])["sharpe"]


@dataclass(frozen=True)
class Fitted:
    """One seed's scorer at its best validation check, and when that was."""

    scorer: Scorer
    validation_sharpe: float
    step: int


def fit(train_env: PortfolioEnv, validation_env: PortfolioEnv, seed: int,
        params: DirectParams = PRE_REGISTERED) -> Fitted:
    """Gradient ascent on the mean Sharpe of random training episodes; the scorer kept is the one with the highest
    validation Sharpe among the checks (every ``check_every`` steps). The test year plays no part."""
    fita = tape_of(train_env)
    ultimo_inicio = len(fita.rate) - params.episode * fita.hold
    if ultimo_inicio < 0:
        raise ValueError(f"an episode of {params.episode} decisions does not fit the training stretch")
    torch.manual_seed(seed)
    gerador = torch.Generator().manual_seed(seed)
    scorer = Scorer(observation_width(train_env.config))
    otimizador = torch.optim.Adam(scorer.parameters(), lr=params.learning_rate)
    melhor, melhor_sharpe, melhor_passo = None, -math.inf, 0
    for passo in range(1, params.steps + 1):
        inicios = torch.randint(0, ultimo_inicio + 1, (params.batch,), generator=gerador)
        scorer.train()
        otimizador.zero_grad(set_to_none=True)
        perda = -sharpe(rollout(scorer, fita, inicios, params.episode)).mean()
        perda.backward()
        torch.nn.utils.clip_grad_norm_(scorer.parameters(), MAX_GRADIENT_NORM)
        otimizador.step()
        if passo % params.check_every == 0 or passo == params.steps:
            scorer.eval()
            nota = stretch_sharpe(validation_env, policy_of([scorer]))
            if nota > melhor_sharpe:
                melhor, melhor_sharpe, melhor_passo = copy.deepcopy(scorer.state_dict()), nota, passo
    scorer.load_state_dict(melhor)
    scorer.eval()
    return Fitted(scorer, melhor_sharpe, melhor_passo)
