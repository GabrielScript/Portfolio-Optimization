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


def test_predict_does_not_read_rows_after_the_decision_date() -> None:
    """The leakage surface is ``predict``: rewriting the future must not change a past call."""
    ret = _retornos(n=1_500)
    corte = ret.index[1_300]
    config = QMLConfig(epochs=2, sample_every=63, horizon=21, seed=0)
    torch.manual_seed(0)                                   # the circuit's initial weights, before _fit seeds it
    base = QMLPredictor(ret, config).predict(corte)
    alterado = ret.copy()
    alterado.loc[alterado.index > corte] *= -5.0
    assert not base.empty
    torch.manual_seed(0)
    assert QMLPredictor(alterado, config).predict(corte).equals(base)


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
        config = QMLConfig()

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


def _acoes_de_teste(n: int = 1_500, k: int = 6) -> pd.DataFrame:
    ret = _retornos(n=n, k=k)
    return pd.DataFrame(index=ret.index, columns=ret.columns, dtype=float)


def test_cache_from_another_configuration_is_rebuilt(tmp_path, monkeypatch) -> None:
    """A stale P(up) file must never be paired with a new config: the sidecar decides."""
    import json

    import scripts.qml_avaliar as avaliar

    cache = tmp_path / "cache.parquet"
    pd.DataFrame({"A": [0.9, 0.9]}, index=pd.DatetimeIndex(["2010-01-05", "2010-02-05"])).to_parquet(cache)
    outra = avaliar._impressao_da_config(QMLConfig())
    outra["config"]["n_layers"] = 99
    cache.with_suffix(".json").write_text(json.dumps(outra), encoding="utf-8")
    chamadas: list[pd.Timestamp] = []

    class _Predictor:
        config = QMLConfig()

        def predict(self, data: pd.Timestamp) -> pd.Series:
            chamadas.append(data)
            return pd.Series(np.linspace(0.2, 0.8, 6), index=list("ABCDEF"))

    quadro = avaliar._probabilidades(_Predictor(), _acoes_de_teste(), cache=cache)
    assert chamadas, "the cache of another configuration was used"
    assert len(quadro) == len(chamadas)


def test_corrupt_cache_is_rebuilt_instead_of_crashing(tmp_path) -> None:
    import json

    import scripts.qml_avaliar as avaliar

    cache = tmp_path / "cache.parquet"
    cache.write_bytes(b"PAR1truncated")
    cache.with_suffix(".json").write_text(json.dumps(avaliar._impressao_da_config(QMLConfig())), encoding="utf-8")

    class _Predictor:
        config = QMLConfig()

        def predict(self, _data: pd.Timestamp) -> pd.Series:
            return pd.Series(np.linspace(0.2, 0.8, 6), index=list("ABCDEF"))

    quadro = avaliar._probabilidades(_Predictor(), _acoes_de_teste(), cache=cache)
    assert len(quadro) > 0
    assert pd.read_parquet(cache).shape == quadro.shape


def test_phases_flag_without_values_is_rejected() -> None:
    import scripts.qml_avaliar as avaliar

    with pytest.raises(ValueError):
        avaliar._armar(["--fases"])


def test_expected_returns_is_flat_without_conviction_and_scales_with_volatility() -> None:
    from core.qml import expected_returns

    sigma = np.array([0.20, 0.40, 0.40])
    neutro = expected_returns(pd.Series(0.5, index=list("ABC")), sigma, 0.10)
    assert np.allclose(neutro.to_numpy(), 0.10)
    mu = expected_returns(pd.Series({"A": 0.6, "B": 0.6, "C": 0.4}), sigma, 0.10, tilt=2.0)
    assert mu.to_dict() == pytest.approx({"A": 0.10 + 2 * 0.2 * 0.20, "B": 0.10 + 2 * 0.2 * 0.40,
                                          "C": 0.10 - 2 * 0.2 * 0.40})
    prior = pd.Series({"C": 0.30, "A": 0.10, "B": 0.20})                 # a per-name prior is aligned by ticker
    com_prior = expected_returns(pd.Series({"A": 0.5, "B": 0.5, "C": 0.5}), sigma, prior)
    assert com_prior.to_dict() == pytest.approx({"A": 0.10, "B": 0.20, "C": 0.30})


def test_init_seed_makes_a_block_independent_of_the_blocks_trained_before() -> None:
    """With ``init_seed`` the 2024 model is the same whether or not 2022 was trained first in the process."""
    ret = _retornos(n=2_100)
    config = QMLConfig(epochs=1, sample_every=63, horizon=21, seed=0, init_seed=7)
    data = ret.index[1_700]
    assert data.year == 2024
    sozinho = QMLPredictor(ret, config).predict(data)
    depois_de_outro = QMLPredictor(ret, config)
    depois_de_outro.model_for(2022)
    pd.testing.assert_series_equal(depois_de_outro.predict(data), sozinho)


def test_default_config_keeps_the_fingerprint_of_the_stored_caches() -> None:
    """A new optional field must not orphan the P(up) files of the frozen arm."""
    import json
    from pathlib import Path

    import scripts.qml_avaliar as avaliar

    lado = Path(avaliar.CACHE).with_suffix(".json")
    if not lado.exists():
        pytest.skip("no stored cache in this checkout")
    assert json.loads(lado.read_text(encoding="utf-8")) == avaliar._impressao_da_config(QMLConfig())
    assert "init_seed" in avaliar._impressao_da_config(QMLConfig(init_seed=3))["config"]


def test_covariance_estimator_is_fitted_on_every_eligible_name_then_cut_to_the_candidates() -> None:
    from core.estimation import covariancia_ledoit_wolf

    ret = _retornos(n=400, k=8)
    setores = {papel: ("Financeiro", "Energia", "Saúde", "Indústria")[i % 4] for i, papel in enumerate(ret.columns)}
    cfg = ConfigOtimizacao(objetivo=Objetivo.KELLY, peso_maximo=0.20, n_ativos_max=None, vol_maxima=0.40,
                           aversao_risco=2.0)
    vistos: list[list[str]] = []

    def espiao(janela: pd.DataFrame) -> pd.DataFrame:
        vistos.append(list(janela.columns))
        return covariancia_ledoit_wolf(janela)

    candidatos = list(ret.columns[:6])
    pesos = qml_strategy(_PredictorFixo(list(ret.columns)), cfg, setores, None, selector=lambda *_: candidatos,
                         covariance=espiao)(ret)
    assert vistos == [list(ret.columns)]                       # the whole eligible window, not the 6 candidates
    assert set(pesos.index) <= set(candidatos)
    assert abs(float(pesos.sum()) - 1.0) < 1e-6 and float(pesos.max()) <= 0.20 + 1e-6


def test_explicit_ledoit_wolf_on_the_full_universe_matches_the_default_path() -> None:
    """Without a selector the window and the candidates coincide: the hook must reproduce the frozen path."""
    from core.estimation import covariancia_ledoit_wolf

    ret = _retornos(n=400, k=8)
    setores = {papel: ("Financeiro", "Energia", "Saúde", "Indústria")[i % 4] for i, papel in enumerate(ret.columns)}
    cfg = ConfigOtimizacao(objetivo=Objetivo.KELLY, peso_maximo=0.20, n_ativos_max=None, vol_maxima=0.40,
                           aversao_risco=2.0)
    fonte = _PredictorFixo(list(ret.columns))
    padrao = qml_strategy(fonte, cfg, setores, None)(ret)
    explicito = qml_strategy(fonte, cfg, setores, None, covariance=covariancia_ledoit_wolf)(ret)
    pd.testing.assert_series_equal(padrao, explicito)


def test_covariance_estimator_is_refused_with_the_jorion_prior() -> None:
    from core.multivariate import pca_covariance

    cfg = ConfigOtimizacao(objetivo=Objetivo.KELLY, peso_maximo=0.20, n_ativos_max=None, vol_maxima=0.40)
    with pytest.raises(ValueError):
        qml_strategy(_PredictorFixo(["A00"]), cfg, {"A00": "Energia"}, None, base="jorion",
                     covariance=pca_covariance)
