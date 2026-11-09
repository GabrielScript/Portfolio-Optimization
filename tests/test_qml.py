"""Unit tests for the QML arm (run in the .venv-qpo environment: PennyLane + Torch)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

from core.config import ConfigOtimizacao, Objetivo
from core.qml import (FEATURE_NAMES, QMLConfig, QMLPredictor, build_panel, cross_sectional_z, features_at,
                      qml_strategy)


def _retornos(n: int = 900, k: int = 8, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    datas = pd.bdate_range("2018-01-01", periods=n)
    return pd.DataFrame(rng.normal(0.0003, 0.01, (n, k)), index=datas,
                        columns=[f"A{i:02d}" for i in range(k)])


def test_features_have_the_declared_columns_and_no_nan() -> None:
    feats = features_at(_retornos())
    assert list(feats.columns) == list(FEATURE_NAMES)
    assert len(feats) == 8
    assert not feats.isna().any().any()


def test_features_do_not_read_past_the_decision_row() -> None:
    ret = _retornos()
    base = features_at(ret.iloc[:600])
    alterado = ret.copy()
    alterado.iloc[600:] *= -5.0
    assert features_at(alterado.iloc[:600]).equals(base)


def test_cross_sectional_z_handles_a_constant_column() -> None:
    frame = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [2.0, 2.0, 2.0]})
    z = cross_sectional_z(frame)
    assert np.isfinite(z.to_numpy()).all()
    assert (z["b"] == 0.0).all()


def test_panel_labels_are_realized_after_the_decision() -> None:
    panel = build_panel(_retornos(), QMLConfig(epochs=2, sample_every=63, horizon=21, seed=0))
    assert len(panel) > 0
    assert (panel.label_dates > panel.decision_dates).all()
    assert set(np.unique(panel.y)) <= {0.0, 1.0}


def test_training_boundary_uses_only_realized_labels() -> None:
    predictor = QMLPredictor(_retornos(), QMLConfig(epochs=2, sample_every=63, horizon=21, seed=0))
    linhas = predictor._training_rows(2021)
    assert len(linhas) > 0
    assert (predictor.panel.label_dates[linhas] < pd.Timestamp("2021-01-01")).all()


def test_quantum_model_learns_a_simple_directional_signal() -> None:
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    x = rng.normal(0.0, 0.6, (480, 6))
    y = (x[:, 0] > 0.0).astype(float)
    from core.qml import QuantumDirectionalModel, _fit
    config = QMLConfig(epochs=60, batch_size=240, learning_rate=0.05, patience=20, seed=0)
    model = _fit(QuantumDirectionalModel(config), x, y, config)
    acerto = float(((model.probability_up(x) > 0.5) == (y > 0.5)).mean())
    assert acerto > 0.60, f"accuracy {acerto:.3f} on a trivial signal"


class _PredictorFixo:
    def __init__(self, tickers: list[str]) -> None:
        self.tickers = tickers

    def predict(self, _date: pd.Timestamp) -> pd.Series:
        valores = np.linspace(-0.9, 0.9, len(self.tickers))
        return pd.Series(1.0 / (1.0 + np.exp(-3.0 * valores)), index=self.tickers)


def test_hybrid_strategy_respects_every_fixed_rule() -> None:
    ret = _retornos(n=400, k=8)
    setores = {papel: ("Financeiro", "Energia", "Saúde", "Indústria")[i % 4]
               for i, papel in enumerate(ret.columns)}
    cfg = ConfigOtimizacao(objetivo=Objetivo.MAX_SHARPE, peso_maximo=0.20, n_ativos_max=None, vol_maxima=0.40)
    estrategia = qml_strategy(_PredictorFixo(list(ret.columns)), cfg, setores, None, tilt=1.0)
    pesos = estrategia(ret)
    assert abs(float(pesos.sum()) - 1.0) < 1e-6
    assert float(pesos.max()) <= 0.20 + 1e-6
    por_setor = pesos.groupby(pesos.index.map(setores)).sum()
    assert float(por_setor.max()) <= 0.30 + 1e-6
    assert (pesos > 0).all()


def test_hybrid_strategy_rejects_a_single_name() -> None:
    ret = _retornos(n=400, k=6)
    setores = {papel: "Energia" for papel in ret.columns}
    cfg = ConfigOtimizacao(objetivo=Objetivo.MAX_SHARPE, peso_maximo=0.20, n_ativos_max=None, vol_maxima=0.40)
    estrategia = qml_strategy(_PredictorFixo(list(ret.columns)), cfg, setores, None, selector=lambda *_: ["A00"])
    with pytest.raises(ValueError):
        estrategia(ret)


def test_probability_cache_keeps_the_ticker_columns(tmp_path, monkeypatch) -> None:
    import scripts.qml_avaliar as avaliar

    ret = _retornos(n=1_500, k=6)
    acoes = pd.DataFrame(index=ret.index, columns=ret.columns, dtype=float)
    monkeypatch.setattr(avaliar, "CACHE", tmp_path / "cache.parquet")
    esperadas = []
    for posicao in range(avaliar.CFG_BT.janela_treino, len(acoes), avaliar.CFG_BT.janela_teste):
        esperadas.append(acoes.index[posicao])
        if posicao + 1 < len(acoes):
            esperadas.append(acoes.index[posicao + 1])

    class _Predictor:
        def predict(self, _data: pd.Timestamp) -> pd.Series:
            return pd.Series(np.linspace(0.2, 0.8, len(ret.columns)), index=ret.columns)

    quadro = avaliar._probabilidades(_Predictor(), acoes)
    assert quadro.shape == (len(esperadas), len(ret.columns))
    assert quadro.index.isin(esperadas).all()


def test_frozen_probabilities_use_the_prior_session() -> None:
    from scripts.qml_avaliar import ProbabilidadesCongeladas

    quadro = pd.DataFrame({"A": [0.4, 0.6], "B": [0.5, 0.7]},
                          index=pd.DatetimeIndex(["2010-01-05", "2010-02-05"]))
    fonte = ProbabilidadesCongeladas(quadro)
    assert fonte.predict(pd.Timestamp("2010-03-01"))["A"] == 0.6
    assert fonte.predict(pd.Timestamp("2009-12-01")).empty
