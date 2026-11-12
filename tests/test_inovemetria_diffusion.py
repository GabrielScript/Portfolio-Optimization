"""Growth laws as exact one-quarter maps of their ODEs, their AICc choice, forecast and stage, and the agent model
that the Bass map is the large-population limit of."""

import numpy as np
import pytest

from core.inovemetria.diffusion import (fit_laws, forecast, simulate_bass_agents, stage, step_bass, step_exponential,
                                        step_gompertz,
                                        step_logistic)


def _euler(rhs, x0: float, steps: int = 200_000) -> float:
    x, h = x0, 1.0 / steps
    for _ in range(steps):
        x += h * rhs(x)
    return x


@pytest.mark.parametrize("x0", [5.0, 40.0, 90.0])
def test_each_step_is_one_quarter_of_its_ode(x0):
    p, q, r, m = 0.03, 0.4, 0.2, 100.0
    assert step_exponential(x0, r) == pytest.approx(_euler(lambda x: r * x, x0), rel=1e-4)
    assert step_logistic(x0, q, m) == pytest.approx(_euler(lambda x: q * x * (1 - x / m), x0), rel=1e-4)
    assert step_bass(x0, p, q, m) == pytest.approx(_euler(lambda x: (p + q * x / m) * (m - x), x0), rel=1e-4)
    assert step_gompertz(x0, r, m) == pytest.approx(_euler(lambda x: r * x * np.log(m / x), x0), rel=1e-4)


def _path(step, x0: float, n: int) -> np.ndarray:
    xs = [x0]
    for _ in range(n):
        xs.append(float(step(xs[-1])))
    return np.array(xs)


def test_fit_recovers_a_noiseless_logistic_and_its_forecast():
    xs = _path(lambda x: step_logistic(x, 0.35, 1000.0), 20.0, 16)
    fit = fit_laws(xs[:-1], xs[1:])
    truth = _path(lambda x: step_logistic(x, 0.35, 1000.0), xs[-1], 4)[-1]
    assert fit.law in ("logistic", "bass")
    assert forecast(fit, xs[-1]) == pytest.approx(truth, rel=1e-3)


def test_a_constant_growth_series_is_exponential():
    xs = 100.0 * np.exp(0.03 * np.arange(14)) * (1 + 0.002 * np.sin(np.arange(14)))
    assert fit_laws(xs[:-1], xs[1:]).law == "exponential"


def test_a_series_above_its_ceiling_forecasts_a_decline():
    xs = _path(lambda x: step_gompertz(x, 0.3, 500.0), 900.0, 14)
    fit = fit_laws(xs[:-1], xs[1:])
    assert forecast(fit, xs[-1]) < xs[-1]
    assert stage(fit, xs[-1]) == "decline"


def test_agents_converge_to_the_bass_curve():
    p, q, dt, steps = 0.02, 0.5, 0.01, 1200
    share = simulate_bass_agents(20_000, p, q, steps, dt, np.random.default_rng(7))   # steps + 1 values from t = 0
    curve = _path(lambda x: step_bass(x, p, q, 1.0), 0.0, steps // 100)                # one unit of time per step
    assert np.max(np.abs(share[::100] - curve)) < 0.02


@pytest.mark.parametrize(("u", "label"), [(0.05, "emergence"), (0.3, "take-off"), (0.7, "maturing"),
                                           (0.95, "saturation"), (1.2, "decline")])
def test_stage_labels_of_a_logistic(u, label):
    from core.inovemetria.diffusion import Fit
    assert stage(Fit("logistic", (0.4, 100.0), 0.0), 100.0 * u) == label
