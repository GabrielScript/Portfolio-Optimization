"""The portfolio environment the three papers run in, as a Gymnasium ``Env`` over our own close panel: the agent picks
weights on the eligible names plus cash, and the next session's returns turn them into the portfolio's return. Three
rewards reproduce the papers (``valor``: change in portfolio value, P2; ``dsr``: Moody-Saffell differential Sharpe, P1;
``dsr_drawdown``: the same plus the drawdown penalty of P3), and ``custo_bps`` charges B3's round trip on the turnover
the action implies (all three papers assume zero cost; ours must show both).

No look-ahead: at row ``t`` the observation is built from closes up to and including ``t``, and the weights chosen there
earn the close-to-close return of ``t+1``. A name that stops trading is settled at its last close into cash and never
breaks a return again."""

from __future__ import annotations

from typing import ClassVar

import numpy as np
import pandas as pd
from gymnasium import Env, spaces

from drl import dados

ETA = 1.0 / 252.0            # the DSR's exponential time scale (P1: about one year of sessions)
PENALIDADE_MDD = 1.0         # weight of the drawdown term in ``dsr_drawdown`` (P3 gives no value; documented choice)
RECOMPENSAS = ("valor", "dsr", "dsr_drawdown")
LOGIT_MAX = 10.0             # the bounded raw action SB3 requires; softmax needs far less range than this


def softmax(x: np.ndarray) -> np.ndarray:
    """The paper's projection of the raw policy output onto the simplex (P1 enforces its constraints this way)."""
    z = x - np.max(x)
    e = np.exp(z)
    return e / e.sum()


def dsr(retorno: float, a: float, b: float) -> tuple[float, float, float]:
    """One step of Moody-Saffell's differential Sharpe ratio and the updated ``(A, B)``. The first steps, before the
    variance is positive, return 0: the denominator has no meaning yet and the reward must not be an infinity."""
    da = retorno - a
    db = retorno * retorno - b
    variancia = b - a * a
    if variancia <= 1e-12:
        return 0.0, a + ETA * da, b + ETA * db
    d = (b * da - 0.5 * a * db) / variancia ** 1.5
    return float(d), a + ETA * da, b + ETA * db


class AmbienteAcoes(Env):
    """One walk-forward window of the panel. ``fechamentos``: wide closes; ``elegiveis``: the point-in-time mask from
    ``dados.elegiveis``; ``inicio``/``fim``: the row range the episode walks. ``modo`` ``"matriz"`` reproduces P1's
    ``(N+1) x lookback`` observation (use it with few names); ``"compacto"`` gives six numbers per name so the full
    universe stays trainable on a CPU."""

    metadata: ClassVar[dict] = {"render_modes": []}

    def __init__(self, fechamentos: pd.DataFrame, elegiveis: pd.DataFrame, inicio: int, fim: int, *,
                 recompensa: str = "dsr", custo_bps: float = 0.0, lookback: int = 60, modo: str = "matriz",
                 vol_mercado: pd.Series | None = None, capital: float = 1_000_000.0):
        super().__init__()
        if recompensa not in RECOMPENSAS:
            raise ValueError(f"recompensa must be one of {RECOMPENSAS}")
        if modo not in ("matriz", "compacto"):
            raise ValueError("modo must be 'matriz' or 'compacto'")
        self.fechamentos, self.elegiveis = fechamentos, elegiveis.astype(bool)
        self.inicio = int(inicio)
        self.fim = min(int(fim), len(fechamentos) - 1)      # the last session earns no return: nothing to step
        self.recompensa, self.custo_bps, self.lookback, self.modo = recompensa, custo_bps, lookback, modo
        self.capital = capital
        self.nomes = list(fechamentos.columns)
        self.transacoes = self.fechamentos[self.nomes].to_numpy(dtype=float)
        self.mascara = self.elegiveis[self.nomes].to_numpy(dtype=bool)
        self.ret = self.transacoes[1:] / self.transacoes[:-1] - 1.0
        self.vol_mercado = (vol_mercado.reindex(fechamentos.index).to_numpy(dtype=float) if vol_mercado is not None
                            else np.zeros(len(fechamentos)))
        n = len(self.nomes)
        self.action_space = spaces.Box(-LOGIT_MAX, LOGIT_MAX, shape=(n + 1,), dtype=np.float32)
        largura = lookback if modo == "matriz" else 6
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=((n + 1) * largura,), dtype=np.float32)
        self._ordem = np.arange(n)
        self.reiniciar_estado()

    # ------------------------------------------------------------------ the episode
    def reiniciar_estado(self) -> None:
        self.t = self.inicio
        self.valor = self.capital
        self.pico = self.capital
        self.pesos = np.zeros(len(self.nomes) + 1)
        self.pesos[-1] = 1.0                     # starts in cash, as P1's backtests do
        self.a = self.b = 0.0
        self.curva: list[float] = []
        self.historico_pesos: list[np.ndarray] = []
        self.giro: list[float] = []

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self.reiniciar_estado()
        return self._observacao(), {}

    def _observacao(self) -> np.ndarray:
        if self.modo == "matriz":
            log = dados.janela_retornos(self.transacoes_pd(), self.t, self.lookback)     # (N, lookback)
            bloco = np.zeros((len(self.nomes) + 1, self.lookback))
            bloco[:-1, 1:] = log[:, 1:]                                               # columns 2..T: the returns
            bloco[:-1, 0] = self.pesos[:-1]                                           # column 1: current allocation
            bloco[-1, 0] = self.pesos[-1]                                             # cash row: its weight, then
            bloco[-1, 1] = self.vol_mercado[self.t] if np.isfinite(self.vol_mercado[self.t]) else 0.0
            bloco[-1, 2] = self.a                                                     # ... the DSR state (P1 has VIX)
            bloco[-1, 3] = self.b
            return np.nan_to_num(bloco, nan=0.0, posinf=0.0, neginf=0.0).ravel().astype(np.float32)
        return self._compacto()

    def transacoes_pd(self) -> pd.DataFrame:
        return pd.DataFrame(self.transacoes, index=self.fechamentos.index, columns=self.nomes)

    def _compacto(self) -> np.ndarray:
        fechado = self.fechamentos.to_numpy(dtype=float)
        t = self.t
        janela = fechado[max(0, t - 252): t + 1]
        with np.errstate(invalid="ignore", divide="ignore"):
            rel252 = np.nan_to_num(janela[-1] / janela[0] - 1.0) if len(janela) > 1 else np.zeros(len(self.nomes))
            r1 = np.nan_to_num(self.ret[t - 1]) if t >= 1 else np.zeros(len(self.nomes))
            r5 = np.nan_to_num(janela[-1] / janela[-6] - 1.0) if len(janela) > 6 else np.zeros(len(self.nomes))
            r21 = np.nan_to_num(janela[-1] / janela[-22] - 1.0) if len(janela) > 22 else np.zeros(len(self.nomes))
        janela_r = self.ret[max(0, t - 63): t]
        vol = np.nan_to_num(np.nanstd(janela_r, axis=0)) if len(janela_r) else np.zeros(len(self.nomes))
        por_ativo = np.stack([self.pesos[:-1], r1, r5, r21, rel252, vol], axis=1)
        linha_caixa = np.array([[self.pesos[-1], self.vol_mercado[t] if np.isfinite(self.vol_mercado[t]) else 0.0,
                                 self.a, self.b, 0.0, 0.0]])
        saida = np.vstack([por_ativo, linha_caixa])
        return np.nan_to_num(saida, nan=0.0, posinf=0.0, neginf=0.0).ravel().astype(np.float32)

    def step(self, action):
        a = np.clip(np.asarray(action, dtype=float).ravel(), -LOGIT_MAX, LOGIT_MAX)
        livres = np.append(self.mascara[self.t], True)            # cash is always available, ineligible names never
        brutos = np.where(livres, a, -np.inf)
        pesos = softmax(brutos)

        deriva = self._deriva()
        giro = float(np.abs(pesos[:-1] - deriva[:-1]).sum())      # the notional traded; cash costs nothing to move
        custo = self.custo_bps / 1e4 * giro
        retorno_ativo = float(np.dot(pesos[:-1], np.nan_to_num(self.ret[self.t])))
        retorno = retorno_ativo - custo
        self.pesos = pesos
        self.valor *= 1.0 + retorno
        self.pico = max(self.pico, self.valor)
        self.curva.append(self.valor)
        self.historico_pesos.append(pesos.copy())
        self.giro.append(giro)

        if self.recompensa == "valor":
            recompensa = retorno
        elif self.recompensa == "dsr":
            recompensa, self.a, self.b = dsr(retorno, self.a, self.b)
        else:
            d, self.a, self.b = dsr(retorno, self.a, self.b)
            recompensa = d - PENALIDADE_MDD * (1.0 - self.valor / self.pico)

        self.t += 1
        terminou = self.t >= self.fim
        return self._observacao(), float(recompensa), terminou, False, {
            "valor": self.valor, "giro": giro, "custo": custo, "retorno": retorno}

    def _deriva(self) -> np.ndarray:
        """The weights the positions drift to after the session they were bought for: what the next action trades from.
        The cash leg does not drift (it earns nothing here), so it keeps its share."""
        if self.t <= self.inicio or not self.historico_pesos:
            return self.pesos.copy()
        anterior = self.historico_pesos[-1]
        ret = np.nan_to_num(self.ret[self.t - 1]) if self.t >= 1 else np.zeros(len(self.nomes))
        valorizado = anterior[:-1] * (1.0 + ret)
        total = valorizado.sum() + anterior[-1]
        return np.append(valorizado / total, anterior[-1] / total) if total > 0 else anterior.copy()
