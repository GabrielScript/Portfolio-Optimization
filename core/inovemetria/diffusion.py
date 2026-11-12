"""
diffusion.py - Four autonomous growth laws of a firm's real revenue, each as the exact one-quarter map of its ODE
(spec "S. Diffusion stage"): exponential, logistic (Verhulst, Fisher-Pry, Bass with p = 0), Bass and Gompertz.
Fitted on consecutive quarter pairs by bounded least squares on log x_{t+1}; the law of smallest AICc is used.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares


def step_exponential(x, r):
    """dx/dt = r x over one quarter."""
    return x * np.exp(r)


def step_logistic(x, q, m):
    """dx/dt = q x (1 - x/m) over one quarter."""
    e = np.exp(q)
    return m * x * e / (m + x * (e - 1.0))


def step_bass(x, p, q, m):
    """dx/dt = (p + q x/m)(m - x) over one quarter: u = (p m + q x)/(m - x) grows by e^{p+q}. At x = m it stays."""
    x = np.asarray(x, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        u = (p * m + q * x) / (m - x) * np.exp(p + q)
        nxt = m * (u - p) / (u + q)
    out = np.where(np.isclose(x, m), m, nxt)
    return out.item() if out.ndim == 0 else out


def step_gompertz(x, r, m):
    """dx/dt = r x log(m/x) over one quarter: log x relaxes to log m at rate r."""
    return np.exp(np.log(m) + np.exp(-r) * (np.log(x) - np.log(m)))


_RATE = (1e-4, 2.0)
LAWS = ("exponential", "logistic", "bass", "gompertz")
_PENALTY = 50.0          # log-residual of a non-finite step: far worse than any fit, keeps the optimizer away


@dataclass(frozen=True)
class Fit:
    law: str
    params: tuple[float, ...]
    aicc: float


def _step(law: str, x, params):
    return {"exponential": step_exponential, "logistic": step_logistic, "bass": step_bass,
            "gompertz": step_gompertz}[law](x, *params)


def _bounds(law: str, x: np.ndarray) -> tuple[list[float], list[float]]:
    """Spec bounds, the ceiling m as log m in [log 0.5 min x, log 100 max x]."""
    lo_m, hi_m = np.log(0.5 * x.min()), np.log(100.0 * x.max())
    return {"exponential": ([-0.5], [0.5]),
            "logistic": ([_RATE[0], lo_m], [_RATE[1], hi_m]),
            "bass": ([0.0, _RATE[0], lo_m], [0.5, _RATE[1], hi_m]),
            "gompertz": ([_RATE[0], lo_m], [_RATE[1], hi_m])}[law]


def _starts(law: str, x: np.ndarray, growth: float) -> list[list[float]]:
    """Deterministic starts: the mean log growth clipped into the rate bounds, ceilings at 1.2, 2 and 10 max x."""
    if law == "exponential":
        return [[float(np.clip(growth, -0.49, 0.49))]]
    rate = float(np.clip(abs(growth), 0.01, 1.9))
    ceilings = [np.log(k * x.max()) for k in (1.2, 2.0, 10.0)]
    if law == "bass":
        return [[0.01, rate, c] for c in ceilings]
    return [[rate, c] for c in ceilings]


def _to_params(law: str, theta: np.ndarray) -> tuple[float, ...]:
    return (float(theta[0]),) if law == "exponential" else (*map(float, theta[:-1]), float(np.exp(theta[-1])))


def _residuals(law: str, before: np.ndarray, after: np.ndarray):
    def _r(theta: np.ndarray) -> np.ndarray:
        with np.errstate(all="ignore"):
            pred = np.asarray(_step(law, before, _to_params(law, theta)), dtype=float)
            res = np.log(pred) - np.log(after)
        return np.where(np.isfinite(res), res, _PENALTY)
    return _r


def aicc(rss: float, n: int, k: int) -> float:
    """n log(RSS/n) + 2k + 2k(k+1)/(n-k-1); k counts the residual variance."""
    return n * np.log(max(rss, 1e-300) / n) + 2 * k + 2 * k * (k + 1) / (n - k - 1)


def fit_laws(before: np.ndarray, after: np.ndarray) -> Fit:
    """The law of smallest AICc on log(after) over the consecutive pairs (before[i], after[i]), all positive."""
    before, after = np.asarray(before, dtype=float), np.asarray(after, dtype=float)
    x = np.concatenate([before, after])
    growth = float(np.mean(np.log(after / before)))
    best: Fit | None = None
    for law in LAWS:
        lower, upper = _bounds(law, x)
        fun = _residuals(law, before, after)
        for start in _starts(law, x, growth):
            start = np.clip(start, lower, upper)
            sol = least_squares(fun, start, bounds=(lower, upper), method="trf", x_scale="jac")
            fit = Fit(law, _to_params(law, sol.x), aicc(float(np.sum(sol.fun ** 2)), len(before), len(start) + 1))
            if best is None or fit.aicc < best.aicc:
                best = fit
    return best


def forecast(fit: Fit, x: float, steps: int = 4) -> float:
    """x after ``steps`` quarters under the fitted law."""
    for _ in range(steps):
        x = float(_step(fit.law, x, fit.params))
    return x


INFLECTION_SHARE = {"logistic": 0.5, "gompertz": float(np.exp(-1.0))}   # u = x/m at the inflection


def _ceiling_share(fit: Fit, x: float) -> tuple[float, float | None]:
    """u = x/m and the inflection share of the fitted law (None when it has no inflection)."""
    m = fit.params[-1]
    if fit.law == "bass":
        p, q, _ = fit.params
        return x / m, ((q - p) / (2 * q) if q > p else None)
    return x / m, INFLECTION_SHARE[fit.law]


def stage(fit: Fit, x: float) -> str:
    """Descriptive stage at revenue x: emergence (u < 0.1), take-off (up to the inflection), maturing (up to 0.9),
    saturation, decline (x > m); exponential law: unbounded growth or decline by the sign of r."""
    if fit.law == "exponential":
        return "unbounded growth" if fit.params[0] > 0 else "decline"
    u, turn = _ceiling_share(fit, x)
    if u > 1.0:
        return "decline"
    if u < 0.1:
        return "emergence"
    if turn is not None and u < turn:
        return "take-off"
    return "maturing" if u < 0.9 else "saturation"


def simulate_bass_agents(n: int, p: float, q: float, steps: int, dt: float, rng: np.random.Generator) -> np.ndarray:
    """Agent model of the Bass law: each non-adopter adopts in a step with probability (p + q adopters/n) dt.
    Returns the adopters' share at t = 0, dt, ..., steps dt."""
    adopted = 0
    shares = [0.0]
    for _ in range(steps):
        chance = min(1.0, (p + q * adopted / n) * dt)
        adopted += rng.binomial(n - adopted, chance)
        shares.append(adopted / n)
    return np.array(shares)
