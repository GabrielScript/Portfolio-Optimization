"""
Independent check of the hand-written core against skfolio (DECISOES 168).

skfolio is a test-only oracle: it is not in requirements.txt and the app never imports it. Run with the
separate venv (`../.venv-skfolio`, requirements.txt + requirements-dev.txt + skfolio 1.12.2 + PySCIPOpt 6.3.0);
in the app's `.venv` every test here is skipped.

Both sides get the same annualized mu and Sigma, so a mismatch is a difference in the optimizer, not in the
estimates.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

skfolio = pytest.importorskip("skfolio")

from skfolio import RiskMeasure  # noqa: E402
from skfolio.moments import BaseCovariance, BaseMu  # noqa: E402
from skfolio.optimization import MeanRisk, ObjectiveFunction  # noqa: E402
from skfolio.prior import EmpiricalPrior  # noqa: E402

from core.config import ConfigOtimizacao, LimiteGrupo, Objetivo  # noqa: E402
from core.optimization import otimizar, preparar_covariancia  # noqa: E402

TOLERANCIA_PESO = 1e-4   # = LIMIAR_PESO: below it our optimizer zeroes weights as solver noise


class MuFixo(BaseMu):
    """Returns the given mu whatever X is."""

    def __init__(self, mu: np.ndarray | None = None) -> None:
        self.mu = mu

    def fit(self, X, y=None) -> MuFixo:  # noqa: N803 (sklearn's name)
        self.mu_ = np.asarray(self.mu, dtype=float)
        return self


class CovarianciaFixa(BaseCovariance):
    """Returns the given Sigma whatever X is (no nearest-PD projection)."""

    def __init__(self, covariancia: np.ndarray | None = None) -> None:
        super().__init__(nearest=False)
        self.covariancia = covariancia

    def fit(self, X, y=None, **fit_params) -> CovarianciaFixa:  # noqa: N803
        self._set_covariance(np.asarray(self.covariancia, dtype=float))
        return self


@pytest.fixture
def mercado() -> tuple[pd.Series, pd.DataFrame, pd.DataFrame]:
    """8 assets, annualized mu and a factor-model Sigma, plus returns X that only carry the asset names."""
    gerador = np.random.default_rng(20261029)
    tickers = [f"A{i}" for i in range(8)]
    cargas = gerador.normal(0.0, 0.18, size=(8, 2))
    cov = cargas @ cargas.T + np.diag(gerador.uniform(0.02, 0.08, size=8))
    mu = pd.Series(gerador.uniform(0.04, 0.30, size=8), index=tickers)
    cov_df = pd.DataFrame(cov, index=tickers, columns=tickers)
    datas = pd.bdate_range("2024-01-01", periods=300)
    retornos = pd.DataFrame(gerador.normal(0.0, 0.01, size=(300, 8)), index=datas, columns=tickers)
    return mu, cov_df, retornos


SETORES = {"S1": ("A0", "A1", "A2"), "S2": ("A3", "A4", "A5"), "S3": ("A6", "A7")}


def _grupos(teto: float) -> tuple[LimiteGrupo, ...]:
    return tuple(LimiteGrupo(nome, frozenset(membros), teto) for nome, membros in SETORES.items())


def _skfolio(mu: pd.Series, cov: pd.DataFrame, retornos: pd.DataFrame, cfg: ConfigOtimizacao,
             **parametros) -> pd.Series:
    """skfolio MeanRisk on our mu and Sigma, with our per-asset cap and the SETORES caps of ``cfg.grupos``."""
    grupos = [next(setor for setor, membros in SETORES.items() if t in membros) for t in mu.index]
    restricoes = [f"{g.nome} <= {g.teto}" for g in cfg.grupos]
    modelo = MeanRisk(
        prior_estimator=EmpiricalPrior(mu_estimator=MuFixo(mu.to_numpy()),
                                       covariance_estimator=CovarianciaFixa(preparar_covariancia(cov))),
        max_weights=cfg.peso_maximo,
        groups=[grupos] if cfg.grupos else None,
        linear_constraints=restricoes or None,
        **parametros,
    )
    modelo.fit(retornos)
    return pd.Series(modelo.weights_, index=mu.index)


def test_kelly_igual_a_maximize_utility_com_aversao_pela_metade(mercado) -> None:
    """Our Kelly, mu'w - (lambda/2) w'Sigma w, is skfolio's utility mu'w - risk_aversion * w'Sigma w."""
    mu, cov, retornos = mercado
    cfg = ConfigOtimizacao(objetivo=Objetivo.KELLY, peso_maximo=0.25, n_ativos_max=None,
                           vol_maxima=None, aversao_risco=2.0, grupos=_grupos(0.40))
    nossos = otimizar(mu, cov, cfg).pesos
    deles = _skfolio(mu, cov, retornos, cfg, objective_function=ObjectiveFunction.MAXIMIZE_UTILITY,
                     risk_measure=RiskMeasure.VARIANCE, risk_aversion=cfg.aversao_risco / 2.0)
    assert (nossos - deles).abs().max() < TOLERANCIA_PESO, pd.DataFrame({"nossos": nossos, "skfolio": deles})


def test_min_vol_igual_a_minimize_risk_variance(mercado) -> None:
    mu, cov, retornos = mercado
    cfg = ConfigOtimizacao(objetivo=Objetivo.MIN_VOL, peso_maximo=0.25, n_ativos_max=None,
                           vol_maxima=None, grupos=_grupos(0.40))
    nossos = otimizar(mu, cov, cfg).pesos
    deles = _skfolio(mu, cov, retornos, cfg, objective_function=ObjectiveFunction.MINIMIZE_RISK,
                     risk_measure=RiskMeasure.VARIANCE)
    assert (nossos - deles).abs().max() < TOLERANCIA_PESO, pd.DataFrame({"nossos": nossos, "skfolio": deles})


def test_max_sharpe_charnes_cooper_igual_a_maximize_ratio(mercado) -> None:
    """Our exact Charnes-Cooper QP is skfolio's ratio objective on the standard deviation, same rf."""
    mu, cov, retornos = mercado
    cfg = ConfigOtimizacao(objetivo=Objetivo.MAX_SHARPE, peso_maximo=0.25, n_ativos_max=None,
                           vol_maxima=None, taxa_livre_risco=0.10, grupos=_grupos(0.40))
    nossos = otimizar(mu, cov, cfg)
    assert nossos.status == "otimo"
    deles = _skfolio(mu, cov, retornos, cfg, objective_function=ObjectiveFunction.MAXIMIZE_RATIO,
                     risk_measure=RiskMeasure.STANDARD_DEVIATION, risk_free_rate=cfg.taxa_livre_risco)
    assert (nossos.pesos - deles).abs().max() < TOLERANCIA_PESO, pd.DataFrame({"nossos": nossos.pesos,
                                                                               "skfolio": deles})


def test_teto_de_vol_igual_a_max_standard_deviation(mercado) -> None:
    """The vol cap ||L'w|| <= vol_max, binding here (min vol 7.8%, free Kelly 13.2%), is skfolio's max_standard_deviation."""
    mu, cov, retornos = mercado
    cfg = ConfigOtimizacao(objetivo=Objetivo.KELLY, peso_maximo=0.25, n_ativos_max=None,
                           vol_maxima=0.10, aversao_risco=1.0, grupos=_grupos(0.40))
    nossos = otimizar(mu, cov, cfg)
    assert nossos.status == "otimo"
    assert nossos.volatilidade == pytest.approx(cfg.vol_maxima, abs=1e-6)   # the cap binds
    deles = _skfolio(mu, cov, retornos, cfg, objective_function=ObjectiveFunction.MAXIMIZE_UTILITY,
                     risk_measure=RiskMeasure.VARIANCE, risk_aversion=cfg.aversao_risco / 2.0,
                     max_standard_deviation=cfg.vol_maxima)
    assert (nossos.pesos - deles).abs().max() < TOLERANCIA_PESO, pd.DataFrame({"nossos": nossos.pesos,
                                                                               "skfolio": deles})


def _utilidade_kelly(pesos: pd.Series, mu: pd.Series, cov: pd.DataFrame, aversao: float) -> float:
    return float(pesos @ mu - aversao / 2.0 * pesos @ cov @ pesos)


@pytest.mark.parametrize("k", [3, 4, 5, 6])
def test_cardinalidade_heuristica_nunca_supera_o_otimo_exato(mercado, k: int) -> None:
    """
    Our cardinality is a two-step heuristic (relax, keep the k largest, solve again); skfolio's is an exact
    MIQP (SCIP). The heuristic must respect k and the caps and can never beat the exact optimum. With
    lambda = 20 the free solution holds 7 names, so k binds; k = 2 is infeasible (2 x 50% under 45% caps).
    """
    mu, cov, retornos = mercado
    cfg = ConfigOtimizacao(objetivo=Objetivo.KELLY, peso_maximo=0.50, n_ativos_max=k,
                           vol_maxima=None, aversao_risco=20.0, grupos=_grupos(0.45))
    nossos = otimizar(mu, cov, cfg).pesos
    exatos = _skfolio(mu, cov, retornos, cfg, objective_function=ObjectiveFunction.MAXIMIZE_UTILITY,
                      risk_measure=RiskMeasure.VARIANCE, risk_aversion=cfg.aversao_risco / 2.0,
                      cardinality=k, solver="SCIP")
    assert int((nossos > 0).sum()) <= k
    assert int((exatos > TOLERANCIA_PESO).sum()) <= k
    assert _utilidade_kelly(nossos, mu, cov, cfg.aversao_risco) <= (
        _utilidade_kelly(exatos, mu, cov, cfg.aversao_risco) + 1e-7)



@pytest.fixture
def retornos_diarios() -> pd.DataFrame:
    """500 sessions of 12 assets with a common factor and different means, so Jorion's intensity is inside (0, 1)."""
    gerador = np.random.default_rng(7)
    fator = gerador.normal(0.0004, 0.012, size=(500, 1))
    medias = gerador.uniform(-0.0004, 0.0012, size=12)
    choques = gerador.normal(0.0, 0.015, size=(500, 12))
    valores = medias + fator @ gerador.uniform(0.5, 1.5, size=(1, 12)) + choques
    return pd.DataFrame(valores, index=pd.bdate_range("2023-01-02", periods=500),
                        columns=[f"A{i}" for i in range(12)])


def test_ledoit_wolf_igual_ao_do_skfolio(retornos_diarios) -> None:
    from skfolio.moments import LedoitWolf as LedoitWolfSkfolio

    from core.estimation import covariancia_ledoit_wolf

    nossa = covariancia_ledoit_wolf(retornos_diarios).to_numpy()
    deles = LedoitWolfSkfolio().fit(retornos_diarios).covariance_
    assert np.abs(nossa - deles).max() < 1e-15


def test_jorion_igual_ao_bayes_stein_do_skfolio(retornos_diarios) -> None:
    """Jorion (1986) with the Ledoit-Wolf plug-in is skfolio's Bayes-Stein toward the min-variance mean."""
    from skfolio.moments import LedoitWolf as LedoitWolfSkfolio
    from skfolio.moments import ShrunkMu, ShrunkMuMethods

    from core.estimation import covariancia_ledoit_wolf, shrinkage_jorion

    nossa, intensidade = shrinkage_jorion(retornos_diarios.mean().to_numpy(),
                                          covariancia_ledoit_wolf(retornos_diarios).to_numpy(),
                                          len(retornos_diarios))
    assert 0.05 < intensidade < 0.95   # the shrinkage is neither clipped nor negligible
    deles = ShrunkMu(covariance_estimator=LedoitWolfSkfolio(), vol_weighted_target=True,
                     method=ShrunkMuMethods.BAYES_STEIN).fit(retornos_diarios)
    assert deles.beta_ == pytest.approx(intensidade, rel=1e-10)
    assert np.abs(nossa - deles.mu_).max() < 1e-15



@pytest.fixture
def excesso_diario() -> pd.Series:
    """Fat-tailed, skewed daily excess returns (Student t plus a jump), Sharpe ~0.05 a day."""
    gerador = np.random.default_rng(11)
    valores = 0.0006 + 0.01 * gerador.standard_t(4, size=1000) - 0.03 * (gerador.random(1000) < 0.01)
    return pd.Series(valores, index=pd.bdate_range("2022-01-03", periods=1000))


def test_psr_igual_ao_do_skfolio(excesso_diario) -> None:
    from skfolio.measures import probabilistic_sharpe_ratio

    from core.metrics import momentos_sharpe, sharpe_probabilistico

    for referencia in (0.0, 0.02, 0.05):
        nosso = sharpe_probabilistico(momentos_sharpe(excesso_diario), referencia)
        deles = probabilistic_sharpe_ratio(excesso_diario.to_numpy(), benchmark_sharpe_ratio=referencia)
        assert 0.01 < nosso < 0.999
        assert nosso == pytest.approx(deles, abs=1e-12)


def test_dsr_igual_ao_do_skfolio(excesso_diario) -> None:
    """Our SR0 with the trials' sample variance (ddof = 1, as the scripts pass it) is skfolio's benchmark."""
    from skfolio.measures import deflated_sharpe_ratio, expected_max_sharpe_ratio

    from core.metrics import momentos_sharpe, sharpe_probabilistico, sharpe_referencia_deflacionado

    sharpes_tentativas = np.random.default_rng(3).normal(0.02, 0.015, size=51)   # N = 51 trials
    referencia = sharpe_referencia_deflacionado(float(np.var(sharpes_tentativas, ddof=1)), 51)
    assert referencia == pytest.approx(
        expected_max_sharpe_ratio(51, float(np.std(sharpes_tentativas, ddof=1))), abs=1e-15)
    nosso = sharpe_probabilistico(momentos_sharpe(excesso_diario), referencia)
    deles = deflated_sharpe_ratio(excesso_diario.to_numpy(), sharpes_tentativas)
    assert nosso == pytest.approx(deles, abs=1e-12)
