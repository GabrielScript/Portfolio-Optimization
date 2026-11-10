"""Volatility targeting: scale the portfolio's exposure so its realised risk stays near a chosen level.

The Composite is long-only and unlevered, so the overlay can only ever *reduce* exposure — never borrow. When realised
volatility rises above the target the portfolio moves part of its capital to cash, which earns the CDI; when volatility
falls back the cash is redeployed. The decision for a session uses returns up to the session before it, so the rule is
tradeable as written.

Measured on the live Composite curve (2010-02 to 2026-11, 10 bps of turnover cost): a 20% target with a 21-session
window took the maximum drawdown from -41.5% to -21.4% while keeping the compound return at 20.3% against 19.9%, and the
Sharpe from 1.04 to 1.20 (DECISOES 336).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

SESSOES = 252
ALVO_MINIMO = 0.005      # 0.5% a year: below this the rule is a division by ~0, not a target


def exposicao(retornos: pd.Series, alvo: float, janela: int = 21) -> pd.Series:
    """The share of capital in the risky portfolio for each session, in [0, 1].

    ``alvo`` is the annualised volatility the portfolio should run at. The realised volatility is measured over the
    ``janela`` sessions ending at the *previous* session — the shift is what makes the rule tradeable — and the exposure
    is ``alvo / vol``, capped at 1 because the strategy never levers up."""
    if janela < 2:
        raise ValueError("janela must be at least 2 sessions")
    if alvo < ALVO_MINIMO:
        raise ValueError(f"alvo must be at least {ALVO_MINIMO:.3f} a year")
    vol = retornos.rolling(janela).std() * np.sqrt(SESSOES)
    escala = (alvo / vol).shift(1)
    return escala.clip(lower=0.0, upper=1.0).fillna(1.0)


def aplicar(retornos: pd.Series, cdi: pd.Series, alvo: float, janela: int = 21, custo_bps: float = 10.0,
            caminho: pd.Series | None = None) -> pd.Series:
    """The portfolio's returns with the overlay applied: the exposed share earns ``retornos``, the rest earns ``cdi``,
    and every change of exposure pays ``custo_bps`` on the notional that moves.

    ``caminho`` supplies an already-computed exposure path, for tests and for reuse; left out, it comes from
    ``exposicao``."""
    taxa = cdi.reindex(retornos.index).ffill().fillna(0.0)
    e = caminho if caminho is not None else exposicao(retornos, alvo, janela)
    e = e.reindex(retornos.index).ffill().fillna(1.0)
    bruto = e * retornos + (1.0 - e) * taxa
    giro = e.diff().abs().fillna(0.0)
    return bruto - giro * custo_bps / 1e4


def medir(retornos: pd.Series) -> dict[str, float]:
    """CAGR, annualised Sharpe (risk-free zero, the overlay already puts idle cash in the CDI) and maximum drawdown."""
    r = np.asarray(retornos.dropna(), dtype=float)
    if len(r) < 2:
        return {"retorno_anual": 0.0, "sharpe": 0.0, "mdd": 0.0}
    curva = np.cumprod(1.0 + r)
    anos = len(r) / SESSOES
    total = float(curva[-1] - 1.0)
    anual = float((1.0 + total) ** (1.0 / anos) - 1.0) if total > -1.0 else -1.0
    desvio = float(np.std(r, ddof=1))
    sharpe = float(np.mean(r) / desvio * np.sqrt(SESSOES)) if desvio > 0 else 0.0
    mdd = float((curva / np.maximum.accumulate(curva) - 1.0).min())
    return {"retorno_anual": anual, "sharpe": sharpe, "mdd": mdd}


def sobrepor(valores: pd.Series, cdi: pd.Series, alvo: float, janela: int = 21,
             custo_bps: float = 10.0) -> tuple[pd.Series, pd.Series]:
    """Apply the overlay to a value curve produced by the engine (``ResultadoBacktest.valores``).

    Returns the scaled curve and the exposure path. The engine's curve already carries its own rebalancing costs and its
    own turnover, so this adds only what the overlay itself costs: the notional that moves in and out of cash."""
    retornos = valores.pct_change().dropna()
    e = exposicao(retornos, alvo, janela)
    com = aplicar(retornos, cdi, alvo, janela, custo_bps, caminho=e)
    escala = float(valores.iloc[0])
    curva = pd.Series(escala * np.cumprod(1.0 + com.to_numpy()), index=retornos.index)
    return pd.concat([pd.Series([escala], index=valores.index[:1]), curva]), e


def imposto_de_estimativa(valores: pd.Series, exposicao_: pd.Series, aliquota: float = 0.15) -> float:
    """An upper bound on the income tax the overlay creates, for the report only.

    Every de-levering sells part of every position. This takes the notional sold — the drops in exposure times the
    portfolio value — and charges the full rate on the whole of it, which overstates the bill whenever a sale realizes a
    loss or the monthly stock-sale exemption (R$ 20 thousand) applies. It is a ceiling, not a forecast."""
    quedas = (-exposicao_.diff().fillna(0.0)).clip(lower=0.0)
    valor = valores.reindex(quedas.index).ffill().fillna(float(valores.iloc[0]))
    return float((quedas * valor).sum() * aliquota)
