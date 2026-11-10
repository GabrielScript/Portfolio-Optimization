"""Tests of core.qnn: the point-in-time panel, the purged split, the circuit and the walk-forward boundary."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

from core.qnn import (QNNConfig, QuantumClassifier, WalkForwardPredictor, build_pit_panel, eligible_by_month,
                      holder_returns, logistic_trainer, purged_split, quantum_trainer, screened, trading_month)

SESSIONS = 700
NAMES = [f"N{i:02d}" for i in range(14)]


def _prices(seed: int = 0, sessions: int = SESSIONS, names: list[str] | None = None) -> pd.DataFrame:
    names = NAMES if names is None else names
    rng = np.random.default_rng(seed)
    datas = pd.bdate_range("2015-01-02", periods=sessions)
    retornos = rng.normal(0.0005, 0.02, size=(sessions, len(names)))
    return pd.DataFrame(100.0 * np.cumprod(1.0 + retornos, axis=0), index=datas, columns=names)


def _universe(precos: pd.DataFrame, names: list[str]) -> pd.DataFrame:
    meses = pd.date_range(precos.index[0].to_period("M").to_timestamp(),
                          precos.index[-1].to_period("M").to_timestamp() + pd.offsets.MonthBegin(1), freq="MS")
    return pd.DataFrame([(mes, nome) for mes in meses for nome in names], columns=["month", "security"])


def _config(**changes) -> QNNConfig:
    return QNNConfig(**{"epochs": 2, "min_names": 3, "min_samples": 20, **changes})


def _returns(precos: pd.DataFrame) -> pd.DataFrame:
    return precos.ffill(limit=5).pct_change().iloc[1:]


# --- panel ------------------------------------------------------------------------------------------------------------


def test_panel_holds_only_the_names_eligible_in_the_trading_month() -> None:
    precos = _prices()
    elegiveis = eligible_by_month(_universe(precos, NAMES[:10]))          # four names are never eligible
    painel = build_pit_panel(_returns(precos), holder_returns(precos), elegiveis, _config())
    por_data = pd.Series(1, index=painel.decision_dates).groupby(level=0).sum()
    assert (por_data == 10).all()


def test_a_name_delisted_inside_the_horizon_keeps_its_sample_with_the_holder_return() -> None:
    precos = _prices()
    morte = 400
    precos.iloc[morte:, 0] = np.nan                                         # N00 stops trading for good
    elegiveis = eligible_by_month(_universe(precos, NAMES))
    config = _config()
    painel = build_pit_panel(_returns(precos), holder_returns(precos), elegiveis, config)
    # the last decision before the death still has the dead name (14 names), although its label spans the death
    antes = precos.index[1:][[i for i in range(config.min_history - 1, morte - 1, config.sample_every)][-1]]
    assert int((painel.decision_dates == antes).sum()) == len(NAMES)


def test_labels_are_split_at_the_median_of_the_same_cross_section() -> None:
    precos = _prices()
    painel = build_pit_panel(_returns(precos), holder_returns(precos), eligible_by_month(_universe(precos, NAMES)),
                             _config())
    media = pd.Series(painel.y, index=painel.decision_dates).groupby(level=0).mean()
    assert np.allclose(media, 0.5)                                          # 14 names: exactly 7 above the median


def test_trading_month_is_the_month_of_the_next_session() -> None:
    sessoes = pd.DatetimeIndex(["2020-01-30", "2020-01-31", "2020-02-03"])
    assert trading_month(sessoes, 0) == pd.Timestamp("2020-01-01")
    assert trading_month(sessoes, 1) == pd.Timestamp("2020-02-01")
    assert trading_month(sessoes, 2) == pd.Timestamp("2020-02-01")           # past the end: the next business day


# --- purged split -----------------------------------------------------------------------------------------------------


def test_purged_split_leaves_no_training_label_inside_the_validation_period() -> None:
    precos = _prices()
    painel = build_pit_panel(_returns(precos), holder_returns(precos), eligible_by_month(_universe(precos, NAMES)),
                             _config())
    treino, validacao = purged_split(painel.decision_dates, painel.label_dates, 0.25)
    inicio_validacao = painel.decision_dates[validacao].min()
    assert len(treino) and len(validacao)
    assert (painel.label_dates[treino] < inicio_validacao).all()
    assert (painel.decision_dates[validacao] >= inicio_validacao).all()
    # the purge really drops samples: some decisions before the validation have labels realized inside it
    assert len(treino) + len(validacao) < len(painel)


# --- circuit ----------------------------------------------------------------------------------------------------------


def test_circuit_matches_the_analytic_expectations_with_idle_rotations() -> None:
    """One layer, zero rotations: RY(theta) on wire 0 then the CNOT ring gives <Z_0> = 1 and <Z_i> = cos(theta)."""
    modelo = QuantumClassifier(n_layers=1, seed=0)
    with torch.no_grad():
        modelo.quantum.weights.zero_()
    theta = 0.7
    x = torch.zeros((1, 6), dtype=torch.float64)
    x[0, 0] = theta
    esperado = np.r_[1.0, np.full(5, np.cos(theta))]
    assert np.allclose(modelo.expectations(x).detach().numpy()[0], esperado, atol=1e-10)


def test_backprop_gradient_matches_finite_differences() -> None:
    modelo = QuantumClassifier(n_layers=2, seed=1)
    x = torch.tensor(np.random.default_rng(0).uniform(-1.5, 1.5, size=(5, 6)))
    y = torch.tensor([0.0, 1.0, 1.0, 0.0, 1.0], dtype=torch.float64)

    def perda() -> torch.Tensor:
        return torch.nn.functional.binary_cross_entropy_with_logits(modelo(x), y)

    perda().backward()
    analitico = modelo.quantum.weights.grad.clone()
    passo = 1e-6
    for indice in [(0, 0, 0), (1, 3, 2), (0, 5, 1)]:
        with torch.no_grad():
            modelo.quantum.weights[indice] += passo
            acima = float(perda())
            modelo.quantum.weights[indice] -= 2 * passo
            abaixo = float(perda())
            modelo.quantum.weights[indice] += passo
        assert abs((acima - abaixo) / (2 * passo) - float(analitico[indice])) < 1e-6


def test_same_seed_gives_the_same_initial_model_without_touching_the_global_generator() -> None:
    torch.manual_seed(123)
    antes = torch.rand(1)
    torch.manual_seed(123)
    a = QuantumClassifier(n_layers=2, seed=7)
    depois = torch.rand(1)
    b = QuantumClassifier(n_layers=2, seed=7)
    assert torch.equal(antes, depois)
    assert all(torch.equal(p, q) for p, q in zip(a.parameters(), b.parameters()))
    assert not torch.equal(a.quantum.weights, QuantumClassifier(n_layers=2, seed=8).quantum.weights)


def test_quantum_trainer_learns_a_separable_signal() -> None:
    rng = np.random.default_rng(0)
    x = rng.uniform(-1.5, 1.5, size=(600, 6))
    y = (x[:, 2] > 0).astype(np.float64)
    datas = pd.DatetimeIndex(np.repeat(pd.bdate_range("2015-01-02", periods=60), 10))
    modelo = quantum_trainer(_config(epochs=25, batch_size=128))(x, y, datas, datas + pd.Timedelta(days=1), 0)
    p = modelo.probability_up(x)
    assert ((p > 0.5) == (y == 1.0)).mean() > 0.9


def test_training_is_the_same_whatever_the_global_generator_holds() -> None:
    rng = np.random.default_rng(1)
    x = rng.uniform(-1.5, 1.5, size=(300, 6))
    y = (x[:, 0] + x[:, 4] > 0).astype(np.float64)
    datas = pd.DatetimeIndex(np.repeat(pd.bdate_range("2015-01-02", periods=30), 10))
    treinar = quantum_trainer(_config(epochs=3, batch_size=64))
    torch.manual_seed(1)
    a = treinar(x, y, datas, datas + pd.Timedelta(days=1), 5).probability_up(x)
    torch.manual_seed(2)
    b = treinar(x, y, datas, datas + pd.Timedelta(days=1), 5).probability_up(x)
    assert np.array_equal(a, b)


# --- walk-forward -----------------------------------------------------------------------------------------------------


class _Recorder:
    """A trainer that records what it was given and answers 0.5."""

    def __init__(self) -> None:
        self.calls: list[tuple[pd.Timestamp, int]] = []

    def __call__(self, x, y, decision_dates, label_dates, seed):
        self.calls.append((label_dates.max(), seed))
        return self

    def probability_up(self, x: np.ndarray) -> np.ndarray:
        return np.full(len(x), 0.5)


def _predictor(precos: pd.DataFrame, trainer, **changes) -> WalkForwardPredictor:
    return WalkForwardPredictor(_returns(precos), holder_returns(precos), eligible_by_month(_universe(precos, NAMES)),
                                _config(**changes), trainer)


def test_a_block_is_trained_only_on_labels_realized_before_its_first_day() -> None:
    precos = _prices()
    gravador = _Recorder()
    previsor = _predictor(precos, gravador)
    previsor.predict(pd.Timestamp("2017-06-01"))
    previsor.predict(pd.Timestamp("2017-03-01"))                            # same block: no second fit
    assert len(gravador.calls) == 1
    assert gravador.calls[0][0] < pd.Timestamp("2017-01-01")


def test_block_seed_does_not_depend_on_the_order_the_blocks_are_trained() -> None:
    precos = _prices(sessions=1000)
    primeiro, segundo = _Recorder(), _Recorder()
    a, b = _predictor(precos, primeiro), _predictor(precos, segundo)
    a.predict(pd.Timestamp("2017-12-01")), a.predict(pd.Timestamp("2018-06-01"))
    b.predict(pd.Timestamp("2018-06-01")), b.predict(pd.Timestamp("2017-12-01"))
    assert dict((seed, fim) for fim, seed in primeiro.calls) == dict((seed, fim) for fim, seed in segundo.calls)


def test_predict_ignores_every_row_after_the_decision_date() -> None:
    precos = _prices()
    data = pd.Timestamp("2017-06-01")
    base = _predictor(precos, logistic_trainer()).predict(data)
    mexido = precos.copy()
    mexido.loc[mexido.index > data] *= 1.5
    mexido.loc[mexido.index > data, "N03"] = np.nan
    assert _predictor(mexido, logistic_trainer()).predict(data).equals(base)
    assert len(base) == len(NAMES) and base.between(0.0, 1.0).all()


def test_predict_answers_only_for_the_eligible_names() -> None:
    precos = _prices()
    previsor = WalkForwardPredictor(_returns(precos), holder_returns(precos),
                                    eligible_by_month(_universe(precos, NAMES[:9])), _config(), logistic_trainer())
    assert sorted(previsor.predict(pd.Timestamp("2017-06-01")).index) == NAMES[:9]


# --- the screen -------------------------------------------------------------------------------------------------------


class _Fixed:
    def __init__(self, p: pd.Series) -> None:
        self.p = p

    def predict(self, date: pd.Timestamp) -> pd.Series:
        return self.p


def test_screen_hands_the_base_strategy_only_the_top_share_by_probability() -> None:
    treino = _returns(_prices()).iloc[-100:]
    p = pd.Series(np.linspace(0.3, 0.7, len(NAMES)), index=NAMES)
    visto: list[list[str]] = []

    def base(janela: pd.DataFrame) -> pd.Series:
        visto.append(list(janela.columns))
        return pd.Series(1.0 / janela.shape[1], index=janela.columns)

    pesos = screened(base, _Fixed(p), keep_share=0.5, min_names=4)(treino)
    assert visto == [NAMES[7:]]
    assert np.isclose(pesos.sum(), 1.0)


def test_screen_refuses_a_cross_section_that_is_too_small() -> None:
    treino = _returns(_prices()).iloc[-100:]
    p = pd.Series(0.5, index=NAMES[:5])
    with pytest.raises(ValueError, match="names"):
        screened(lambda janela: pd.Series(dtype=float), _Fixed(p), keep_share=0.5, min_names=4)(treino)


def test_screen_rejects_a_share_outside_the_unit_interval() -> None:
    with pytest.raises(ValueError):
        screened(lambda janela: pd.Series(dtype=float), _Fixed(pd.Series(dtype=float)), keep_share=0.0)
