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
REWARDS = ("value", "dsr", "dsr_drawdown", "dsr_cdi")
OBSERVATIONS = ("matrix", "compact", "cross", "cross_score")
COMPACT_WIDTH = 6
RETURN_WINDOWS = (1, 5, 21, 252)  # sessions of each return the compact and cross observations carry
VOL_WINDOW = 63
Z_CLIP = 3.0
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
    dsr_warmup: int = 0                  # sessions behind the first (A, B); 0 = the papers' cold start
    invest_all: bool = False             # the action holds cash only when no name is tradeable

    def __post_init__(self) -> None:
        if self.reward not in REWARDS:
            raise ValueError(f"reward must be one of {REWARDS}: {self.reward!r}")
        if self.observation not in OBSERVATIONS:
            raise ValueError(f"observation must be one of {OBSERVATIONS}: {self.observation!r}")
        if self.hold < 1 or self.lookback < 2 or self.cost_bps < 0 or self.dsr_warmup < 0:
            raise ValueError("hold >= 1, lookback >= 2, cost_bps >= 0 and dsr_warmup >= 0 are required")


def observation_width(config: EnvConfig) -> int:
    """Numbers per row (one row per name and one for cash) of the observation."""
    return {"matrix": config.lookback, "compact": COMPACT_WIDTH, "cross": COMPACT_WIDTH,
            "cross_score": COMPACT_WIDTH + 1}[config.observation]


def standardize(values: np.ndarray, rows: np.ndarray) -> np.ndarray:
    """Each column of ``values`` (names x inputs) as a z-score across the names of ``rows`` that have a number there,
    clipped to +/- ``Z_CLIP``; 0 for every other name and for a column without dispersion. A return of 0.5% in a
    session and one of 50% in a year reach the network on one scale, and what it sees is the cross-section."""
    usados = rows[:, None] & np.isfinite(values)
    contagem = np.maximum(usados.sum(axis=0), 1)
    limpos = np.where(usados, values, 0.0)
    media = limpos.sum(axis=0) / contagem
    desvio = np.sqrt((np.where(usados, values - media, 0.0) ** 2).sum(axis=0) / contagem)
    z = (limpos - media) / np.where(desvio > 0, desvio, 1.0)
    return np.where(usados & (desvio > 0), np.clip(z, -Z_CLIP, Z_CLIP), 0.0)


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


@dataclass(frozen=True)
class Stretch:
    """An environment's stretch as arrays, one row per session from ``start`` to ``stop - 1``: what a simulator
    written elsewhere (``drl.direto``) needs to replay it."""

    frames: np.ndarray           # (rows, n + 1, width) the observation of each row, weights column at 0
    tradeable: np.ndarray        # (rows, n) bool
    forward: np.ndarray          # (rows, n) each name's return to the next row
    cash: np.ndarray             # (rows,) what cash earns to the next row
    rate: np.ndarray             # (rows,) the CDI to the next row


class PortfolioEnv(Env):
    """
    One stretch of the tape: decisions at rows ``start, start + hold, ...`` while the row is below ``stop``; the last
    return earned is the one from ``stop - 1`` to ``stop``. ``columns`` are the panel positions of the names this
    environment may hold (fixed for its life, so the action vector never changes shape); the last weight is cash.

    ``episode``: decisions per episode, each episode starting on a row drawn from the stretch (training). None runs
    the stretch once from ``start`` (validation, test, the rules). Replaying one fixed path of about 60 decisions
    thousands of times is how an agent memorizes a tape.

    ``allowed``: which of ``columns`` this stretch may trade at all, besides each session's eligibility. Training
    sees every name that was eligible in its years; the test trades the names known when it starts.
    """

    metadata: ClassVar[dict] = {"render_modes": []}

    def __init__(self, panel: Panel, columns: np.ndarray, start: int, stop: int, config: EnvConfig, *,
                 episode: int | None = None, allowed: np.ndarray | None = None) -> None:
        super().__init__()
        ultimo = len(panel.sessions) - 1
        if not 0 <= start < stop <= ultimo:
            raise ValueError(f"need 0 <= start < stop <= {ultimo}: start {start}, stop {stop}")
        if episode is not None and not 1 <= episode * config.hold <= stop - start:
            raise ValueError(f"an episode of {episode} decisions of {config.hold} sessions does not fit rows "
                             f"{start} to {stop}")
        if allowed is not None and np.shape(allowed) != (len(columns),):
            raise ValueError(f"allowed must have one flag per column: {np.shape(allowed)} for {len(columns)}")
        self.config = config
        self.start, self.stop, self.episode = int(start), int(stop), episode
        self._forward = panel.forward[:, columns]                    # (sessions - 1, n)
        self._eligible = panel.eligible[:, columns] & (True if allowed is None else np.asarray(allowed, dtype=bool))
        self._cdi = panel.cdi if config.cash_earns_cdi else np.zeros_like(panel.cdi)
        self._rate = panel.cdi                                       # what "dsr_cdi" measures the return against
        self._market = panel.market
        contagem = self._eligible[:-1].sum(axis=1)
        self._market_return = np.where(contagem > 0, (self._forward * self._eligible[:-1]).sum(axis=1)
                                       / np.maximum(contagem, 1), 0.0)       # equal-weighted, row t: from t to t + 1
        self._log = np.log1p(self._forward)
        self._cumulative = np.vstack([np.zeros(len(columns)), np.cumsum(self._log, axis=0)])   # row t: sum up to t
        self.n = len(columns)
        if config.observation == "cross_score" and panel.scores is None:
            raise ValueError("the cross_score observation needs a panel with scores")
        self._scores = None if panel.scores is None else panel.scores[:, columns]
        largura = observation_width(config)
        self.action_space = spaces.Box(-LOGIT_BOUND, LOGIT_BOUND, shape=(self.n + 1,), dtype=np.float32)
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=((self.n + 1) * largura,), dtype=np.float32)
        self._restart(self.start)

    # ------------------------------------------------------------------ state
    def _restart(self, first: int) -> None:
        self.t = first
        self._end = self.stop if self.episode is None else first + self.episode * self.config.hold
        self.value = 1.0
        self.peak = 1.0
        self.weights = np.zeros(self.n + 1)
        self.weights[-1] = 1.0                       # every backtest starts in cash, as P1's do
        self._a, self._b = self._warm_start()
        self.values: list[float] = [1.0]             # one point per session, the first at ``start``
        self.turnover: list[float] = []              # one number per decision
        self.decisions: list[np.ndarray] = []        # the weights bought at each decision

    def _warm_start(self) -> tuple[float, float]:
        """The first ``(A, B)`` of the differential Sharpe ratio: the mean and mean square of the equal-weighted market
        of this environment's names over the ``dsr_warmup`` sessions realized by the first row, measured as the reward
        measures the book. From ``(0, 0)`` the variance estimate grows from nothing, and the reward of session ``k``
        is about ``1 / sqrt(k * ETA)`` times its steady-state size."""
        inicio = max(0, self.t - self.config.dsr_warmup)
        mercado = self._market_return[inicio:self.t]
        if self.config.reward == "dsr_cdi":
            mercado = mercado - self._rate[inicio:self.t]
        if len(mercado) < 2:
            return 0.0, 0.0
        return float(mercado.mean()), float((mercado * mercado).mean())

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        primeira = self.start
        if self.episode is not None:
            primeira = int(self.np_random.integers(self.start, self.stop - self.episode * self.config.hold + 1))
        self._restart(primeira)
        return self.observe(), {}

    @property
    def done(self) -> bool:
        return self.t >= self._end

    def tradeable(self) -> np.ndarray:
        """Which of the names may be bought at the current row (cash always may)."""
        return self._eligible[self.t]

    def rates(self) -> np.ndarray:
        """The CDI of each session of the stretch: entry ``k`` is earned from row ``start + k`` to the next."""
        return self._rate[self.start:self.stop]

    def arrays(self) -> Stretch:
        trecho = slice(self.start, self.stop)
        return Stretch(np.stack([self.frame(t) for t in range(self.start, self.stop)]), self._eligible[trecho],
                       self._forward[trecho], self._cdi[trecho], self._rate[trecho])

    def past_returns(self, sessions: int) -> np.ndarray:
        """The last ``sessions`` holder returns realized by the current close: (rows, n), oldest first."""
        return self._forward[max(0, self.t - sessions):self.t]

    # ------------------------------------------------------------------ observation
    def observe(self) -> np.ndarray:
        """The frame of the current row with the weights held in its first column, flattened."""
        quadro = self.frame(self.t)
        quadro[:, 0] = self.weights
        return quadro.ravel().astype(np.float32)

    def frame(self, t: int) -> np.ndarray:
        """Row ``t``'s observation but for the weights (first column, left at 0): ``(n + 1, width)``. Built from rows
        strictly before ``t`` of the forward returns: nothing the next action will earn."""
        quadro = np.zeros((self.n + 1, observation_width(self.config)))
        if self.config.observation == "matrix":
            recentes = self._log[max(0, t - self.config.lookback + 1):t][::-1]         # r(t-1), r(t-2), ...
            quadro[:-1, 1:1 + len(recentes)] = recentes.T
        elif self.config.observation == "compact":
            quadro[:-1, 1:] = self._price_inputs(t)
        else:
            entradas = self._price_inputs(t)
            if self.config.observation == "cross_score":
                entradas = np.column_stack([entradas, self._scores[t]])
            quadro[:-1, 1:] = standardize(entradas, self._eligible[t])
        quadro[-1, 1:3] = self._market[t]
        return quadro

    def _price_inputs(self, t: int) -> np.ndarray:
        """(n, 5): the returns over ``RETURN_WINDOWS`` and the volatility of the last ``VOL_WINDOW`` sessions."""
        retornos = [np.expm1(self._cumulative[t] - self._cumulative[max(0, t - janela)]) for janela in RETURN_WINDOWS]
        trecho = self._log[max(0, t - VOL_WINDOW):t]
        return np.column_stack([*retornos, trecho.std(axis=0) if len(trecho) > 1 else np.zeros(self.n)])

    # ------------------------------------------------------------------ dynamics
    def project(self, action: np.ndarray) -> np.ndarray:
        """Raw action -> weights: softmax over the tradeable names and cash. With ``invest_all`` the cash logit counts
        only when no name is tradeable: under a Sharpe over the CDI the cash share is not identified (scaling the
        risky book scales the mean and the deviation alike), so it is not left for the agent to drift on."""
        logits = np.clip(np.asarray(action, dtype=float).ravel(), -LOGIT_BOUND, LOGIT_BOUND)
        tradeable = self.tradeable()
        cash = not (self.config.invest_all and tradeable.any())
        return softmax(np.where(np.append(tradeable, cash), logits, -np.inf))

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
            recompensa += self._reward(retorno, float(self._rate[self.t]))
            livro = crescido / crescido.sum()
            self.t += 1
        self.weights = livro
        return recompensa

    def _reward(self, retorno: float, rate: float) -> float:
        """The session's reward. ``dsr_cdi`` is the differential Sharpe ratio of the return over the CDI: with the raw
        return and cash earning the CDI, a book of cash alone has the highest Sharpe there is, and the agent was
        being trained towards a book the gate scores 0."""
        if self.config.reward == "value":
            return float(np.log1p(retorno))
        medido = retorno - rate if self.config.reward == "dsr_cdi" else retorno
        valor, self._a, self._b = differential_sharpe(medido, self._a, self._b)
        if self.config.reward == "dsr_drawdown":
            return valor - self.config.drawdown_weight * (1.0 - self.value / self.peak)
        return valor

    def step(self, action):
        """The end of an episode is the end of the data, not of the problem: it is reported as a truncation, so the
        learner bootstraps the value of the last state instead of taking it as 0."""
        recompensa = self.apply(self.project(action))
        return self.observe(), float(recompensa), False, self.done, {"value": self.value}


Policy = Callable[[PortfolioEnv], np.ndarray]
"""A rule: the environment at a decision -> the n + 1 weights to hold."""


def simulate(env: PortfolioEnv, policy: Policy) -> PortfolioEnv:
    """Run ``policy`` over one episode of ``env`` and return the environment with its curve, weights and turnover."""
    env.reset()
    while not env.done:
        env.apply(policy(env))
    return env
