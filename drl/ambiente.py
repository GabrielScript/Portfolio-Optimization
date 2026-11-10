"""The portfolio environment of the three papers on our tape, and the one simulator everything is measured in: an
agent's action and a classical rule's weights go through the same ``apply`` (same timing, cost, drift, eligibility and
cash), so a difference between two curves is a difference between two strategies and nothing else.

Papers: P1 = Sood et al., "Deep RL for optimal portfolio allocation: a comparative study with mean-variance
optimization" (PPO, differential Sharpe reward, the ``(n + 1) x lookback`` observation); P2 = Sattar et al., "RMS-driven
deep RL" (A2C / DDPG / PPO, drawdown-based reward); P3 = Rezaei and Nezamabadi-Pour's review (A2C, DDPG, PPO, SAC, TD3).

Timing. A decision at the close of session ``t`` sees returns realized up to that close, trades at it, and earns the
return from ``t`` to ``t + 1``. With ``hold = h`` one step is one decision: the book is bought, then drifts untouched for
``h`` sessions, and the step's reward is the sum of the ``h`` session rewards. (The first version kept daily steps and
silently replaced 20 of every 21 actions by "hold", while still crediting those ignored actions with rewards.)

Cash earns the CDI. The first version paid it nothing, which on B3 hands every strategy that holds cash a penalty of
ten points a year or more.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import ClassVar

import numpy as np
from gymnasium import Env, spaces

from drl.dados import Panel

ETA = 1.0 / 252.0                # the differential Sharpe ratio's time scale (P1: about a year of sessions)
REWARDS = ("value", "dsr", "dsr_drawdown")
OBSERVATIONS = ("matrix", "compact")
COMPACT_WIDTH = 6
LOGIT_BOUND = 10.0               # SB3 needs a bounded box; a softmax needs far less range than this


@dataclass(frozen=True)
class EnvConfig:
    """What the environment does besides replaying the tape; one frozen object per run, recorded with its result."""

    reward: str = "dsr"
    cost_bps: float = 10.0               # per unit of notional traded, each side (the house engine's convention)
    hold: int = 21                       # sessions between decisions
    lookback: int = 60                   # P1's T
    observation: str = "matrix"
    cash_earns_cdi: bool = True
    drawdown_weight: float = 1.0         # P2 gives no value for its drawdown term; fixed here, never scanned

    def __post_init__(self) -> None:
        if self.reward not in REWARDS:
            raise ValueError(f"reward must be one of {REWARDS}: {self.reward!r}")
        if self.observation not in OBSERVATIONS:
            raise ValueError(f"observation must be one of {OBSERVATIONS}: {self.observation!r}")
        if self.hold < 1 or self.lookback < 2 or self.cost_bps < 0:
            raise ValueError("hold >= 1, lookback >= 2 and cost_bps >= 0 are required")


def softmax(logits: np.ndarray) -> np.ndarray:
    """P1's projection of the raw action onto the simplex; ``-inf`` entries get exactly zero."""
    deslocado = logits - np.max(logits)
    expoente = np.exp(deslocado)
    return expoente / expoente.sum()


def differential_sharpe(retorno: float, a: float, b: float) -> tuple[float, float, float]:
    """One step of Moody and Saffell's differential Sharpe ratio and the updated ``(A, B)``. While the variance
    estimate is not positive the ratio is undefined and the reward is 0."""
    delta_a, delta_b = retorno - a, retorno * retorno - b
    variancia = b - a * a
    valor = 0.0 if variancia <= 1e-12 else (b * delta_a - 0.5 * a * delta_b) / variancia ** 1.5
    return float(valor), a + ETA * delta_a, b + ETA * delta_b


class PortfolioEnv(Env):
    """
    One stretch of the tape: decisions at rows ``start, start + hold, ...`` while the row is below ``stop``; the last
    return earned is the one from ``stop - 1`` to ``stop``. ``columns`` are the panel positions of the names this
    environment may hold (fixed for its life, so the action vector never changes shape); the last weight is cash.
    """

    metadata: ClassVar[dict] = {"render_modes": []}

    def __init__(self, panel: Panel, columns: np.ndarray, start: int, stop: int, config: EnvConfig) -> None:
        super().__init__()
        ultimo = len(panel.sessions) - 1
        if not 0 <= start < stop <= ultimo:
            raise ValueError(f"need 0 <= start < stop <= {ultimo}: start {start}, stop {stop}")
        self.config = config
        self.start, self.stop = int(start), int(stop)
        self._forward = panel.forward[:, columns]                    # (sessions - 1, n)
        self._eligible = panel.eligible[:, columns]
        self._cdi = panel.cdi if config.cash_earns_cdi else np.zeros_like(panel.cdi)
        self._market = panel.market
        self._log = np.log1p(self._forward)
        self._cumulative = np.vstack([np.zeros(len(columns)), np.cumsum(self._log, axis=0)])   # row t: sum up to t
        self.n = len(columns)
        largura = config.lookback if config.observation == "matrix" else COMPACT_WIDTH
        self.action_space = spaces.Box(-LOGIT_BOUND, LOGIT_BOUND, shape=(self.n + 1,), dtype=np.float32)
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=((self.n + 1) * largura,), dtype=np.float32)
        self._restart()

    # ------------------------------------------------------------------ state
    def _restart(self) -> None:
        self.t = self.start
        self.value = 1.0
        self.peak = 1.0
        self.weights = np.zeros(self.n + 1)
        self.weights[-1] = 1.0                       # every backtest starts in cash, as P1's do
        self._a = self._b = 0.0
        self.values: list[float] = [1.0]             # one point per session, the first at ``start``
        self.turnover: list[float] = []              # one number per decision
        self.decisions: list[np.ndarray] = []        # the weights bought at each decision

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self._restart()
        return self.observe(), {}

    @property
    def done(self) -> bool:
        return self.t >= self.stop

    def tradeable(self) -> np.ndarray:
        """Which of the names may be bought at the current row (cash always may)."""
        return self._eligible[self.t]

    def past_returns(self, sessions: int) -> np.ndarray:
        """The last ``sessions`` holder returns realized by the current close: (rows, n), oldest first."""
        return self._forward[max(0, self.t - sessions):self.t]

    # ------------------------------------------------------------------ observation
    def observe(self) -> np.ndarray:
        """Built from rows strictly before ``t`` of the forward returns: nothing the next action will earn."""
        t = min(self.t, len(self._market) - 1)
        if self.config.observation == "matrix":
            quadro = np.zeros((self.n + 1, self.config.lookback))
            recentes = self._log[max(0, t - self.config.lookback + 1):t][::-1]         # r(t-1), r(t-2), ...
            quadro[:-1, 1:1 + len(recentes)] = recentes.T
        else:
            quadro = np.zeros((self.n + 1, COMPACT_WIDTH))
            for coluna, janela in enumerate((1, 5, 21, 252), start=1):
                quadro[:-1, coluna] = np.expm1(self._cumulative[t] - self._cumulative[max(0, t - janela)])
            trecho = self._log[max(0, t - 63):t]
            quadro[:-1, 5] = trecho.std(axis=0) if len(trecho) > 1 else 0.0
        quadro[:, 0] = self.weights
        quadro[-1, 1:3] = self._market[t]
        return quadro.ravel().astype(np.float32)

    # ------------------------------------------------------------------ dynamics
    def project(self, action: np.ndarray) -> np.ndarray:
        """Raw action -> weights: softmax over the tradeable names and cash."""
        logits = np.clip(np.asarray(action, dtype=float).ravel(), -LOGIT_BOUND, LOGIT_BOUND)
        return softmax(np.where(np.append(self.tradeable(), True), logits, -np.inf))

    def apply(self, target: np.ndarray) -> float:
        """
        Trade to ``target`` (n names + cash, on the simplex) at the current close, hold for ``hold`` sessions and
        return the summed reward. The cost is charged on the notional traded against the drifted book and comes out of
        the first session's return.
        """
        if self.done:
            raise RuntimeError("the episode is over: reset the environment")
        target = np.asarray(target, dtype=float)
        if target.shape != (self.n + 1,) or (target < -1e-9).any() or abs(target.sum() - 1.0) > 1e-6:
            raise ValueError("target must be n + 1 non-negative weights summing to 1")
        if (target[:-1][~self.tradeable()] > 1e-9).any():
            raise ValueError("target puts weight on a name that is not tradeable at this session")
        giro = float(np.abs(target[:-1] - self.weights[:-1]).sum())
        custo = self.config.cost_bps / 1e4 * giro
        self.turnover.append(giro)
        self.decisions.append(target.copy())
        recompensa, livro = 0.0, target
        for _ in range(self.config.hold):
            if self.done:
                break
            crescido = livro * np.append(1.0 + self._forward[self.t], 1.0 + self._cdi[self.t])
            retorno = float(crescido.sum()) - 1.0 - custo
            custo = 0.0
            self.value *= 1.0 + retorno
            self.peak = max(self.peak, self.value)
            self.values.append(self.value)
            recompensa += self._reward(retorno)
            livro = crescido / crescido.sum()
            self.t += 1
        self.weights = livro
        return recompensa

    def _reward(self, retorno: float) -> float:
        if self.config.reward == "value":
            return float(np.log1p(retorno))
        valor, self._a, self._b = differential_sharpe(retorno, self._a, self._b)
        if self.config.reward == "dsr":
            return valor
        return valor - self.config.drawdown_weight * (1.0 - self.value / self.peak)

    def step(self, action):
        recompensa = self.apply(self.project(action))
        return self.observe(), float(recompensa), self.done, False, {"value": self.value}


Policy = Callable[[PortfolioEnv], np.ndarray]
"""A rule: the environment at a decision -> the n + 1 weights to hold."""


def simulate(env: PortfolioEnv, policy: Policy) -> PortfolioEnv:
    """Run ``policy`` over one episode of ``env`` and return the environment with its curve, weights and turnover."""
    env.reset()
    while not env.done:
        env.apply(policy(env))
    return env
