"""The second batch's arms as engine strategies (docs/superpowers/specs/2026-11-01-lote2-livro-design.md, section 1)."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from core.backtest import CarteiraRecusada
from core.baixo_posto import densa, ledoit_wolf_baixo_posto
from core.estrategias_todas import no_grid_mensal_real


def _treino(linhas: int = 120, colunas: int = 30, escala: float = 1.0) -> pd.DataFrame:
    gerador = np.random.default_rng(0)
    comum = gerador.normal(0.0008, 0.008, linhas)[:, None] * gerador.uniform(0.5, 1.5, colunas)
    dados = (comum + gerador.normal(0.0004, 0.012, (linhas, colunas))) * escala
    return pd.DataFrame(dados, columns=[f"N{i:03d}" for i in range(colunas)], index=pd.bdate_range("2019-01-01", periods=linhas))


def _setores(treino: pd.DataFrame) -> dict[str, str]:
    return {papel: f"S{i % 6}" for i, papel in enumerate(treino.columns)}


def _nota(mes: pd.Timestamp, candidatos: list[str]) -> pd.Series:
    return pd.Series(np.arange(len(candidatos), 0, -1, dtype=float), index=candidatos)


def _bracos():
    from core.estrategias_lote2 import bracos_lote2
    return bracos_lote2(_nota)


NOMES = ("Kelly-exato", "Kelly-exato-sem-teto", "Quintil-mu", "Quintil-mu-sigma", "Quintil-mu-sigma2", "Box-robusto", "Subset-resampling",
         "Subset-bagging", "Downside-1", "Semi-variancia", "Downside-3", "MVSK-tilting", "Utilidade-raiz", "Retorno-minimo")


def test_the_arms_are_the_fourteen_preregistered_rows():
    linhas = Path("docs/tentativas.csv").read_text(encoding="utf-8").splitlines()
    assert tuple(_bracos()) == NOMES
    assert all(any(f'"{nome}:' in linha for linha in linhas) for nome in NOMES)


@pytest.mark.parametrize("nome", NOMES)
def test_each_arm_gives_a_portfolio_inside_the_rules_on_daily_rows(nome):
    treino = _treino(colunas=100 if nome.startswith("Quintil") else 30)     # a small quintile may not fit the sector caps: see below
    setores = _setores(treino)
    pesos = _bracos()[nome](setores, 252)(treino)
    assert pesos.sum() == pytest.approx(1.0, abs=1e-6)
    assert pesos.min() >= 0.0
    assert pesos.max() <= 0.20 + 1e-6
    assert pesos.groupby(pd.Series(setores)).sum().max() <= 0.30 + 1e-6


@pytest.mark.parametrize("nome", NOMES)
def test_each_arm_gives_a_portfolio_on_monthly_rows_with_more_names_than_rows(nome):
    treino = _treino(linhas=36, colunas=60, escala=4.0)
    treino = treino.set_axis(pd.date_range("2019-01-31", periods=36, freq="ME"))
    pesos = no_grid_mensal_real(_bracos()[nome](_setores(treino), 12))(treino)
    assert pesos.sum() == pytest.approx(1.0, abs=1e-6)
    assert pesos.max() <= 0.20 + 1e-6


def test_the_cap_free_kelly_goes_past_the_cap_that_binds_the_other():
    treino = _treino(escala=3.0)
    setores = _setores(treino)
    cov = densa(ledoit_wolf_baixo_posto(treino, 252)).to_numpy()
    com_teto = _bracos()["Kelly-exato"](setores, 252)(treino).to_numpy()
    sem_teto = _bracos()["Kelly-exato-sem-teto"](setores, 252)(treino).to_numpy()
    assert np.sqrt(com_teto @ cov @ com_teto) <= 0.40 + 1e-6
    assert np.sqrt(sem_teto @ cov @ sem_teto) > 0.40 + 1e-3


def test_the_cap_free_kelly_still_keeps_the_sector_cap():
    treino = _treino(escala=3.0)
    setores = _setores(treino)
    pesos = _bracos()["Kelly-exato-sem-teto"](setores, 252)(treino)
    assert pesos.groupby(pd.Series(setores)).sum().max() <= 0.30 + 1e-6


def test_a_quintile_left_with_cash_is_refused_and_not_scaled_past_the_caps():
    """The engine divides by the sum, so a quintile whose sectors cannot hold it all would go over the caps there."""
    treino = _treino(colunas=30)
    setores = {papel: ("A" if i < 15 else "B") for i, papel in enumerate(treino.columns)}
    with pytest.raises(CarteiraRecusada):
        _bracos()["Quintil-mu"](setores, 252)(treino)


def test_tilting_starts_from_the_composite_at_the_engines_scale():
    """The Composite's own volatility check annualizes by 252: on real monthly rows it would see 21 times the variance and refuse."""
    treino = _treino(linhas=36, colunas=60, escala=4.0).set_axis(pd.date_range("2019-01-31", periods=36, freq="ME"))
    from core.composto import estrategia_por_nota

    setores = _setores(treino)
    composto = estrategia_por_nota(_nota, setores)(treino * np.sqrt(12 / 252))
    pesos = no_grid_mensal_real(_bracos()["MVSK-tilting"](setores, 12))(treino).to_numpy()
    base = composto.reindex(treino.columns).fillna(0.0).to_numpy()
    sigma = ledoit_wolf_baixo_posto(treino, 12)
    cov, media = densa(sigma).to_numpy(), treino.mean().to_numpy()
    centrados = treino.to_numpy() - treino.to_numpy().mean(axis=0)

    def momentos(w: np.ndarray) -> np.ndarray:
        return np.array([media @ w, w @ cov @ w, ((centrados @ w) ** 3).mean(), ((centrados @ w) ** 4).mean()])

    assert np.all((momentos(pesos) - momentos(base)) * np.array([1.0, -1.0, 1.0, -1.0]) > 0.0)
    assert float(pesos @ pesos) <= 1.01 * float(base @ base) + 1e-6


def test_subset_arms_use_a_block_of_a_month_on_either_grid():
    from core.estrategias_lote2 import bloco_de_um_mes

    assert bloco_de_um_mes(252) == 21
    assert bloco_de_um_mes(12) == 1


@pytest.mark.parametrize("nome", ["Kelly-exato", "Utilidade-raiz", "Quintil-mu", "Quintil-mu-sigma", "Quintil-mu-sigma2", "Box-robusto",
                                  "Downside-1", "Semi-variancia", "Downside-3", "Retorno-minimo"])
def test_each_arm_is_wired_to_its_own_formulation(nome):
    from core import carteiras_lote2 as cl
    from core.carteiras_livros import media_por_periodo
    from core.config import ConfigOtimizacao, Objetivo, limites_por_setor

    treino = _treino(colunas=100 if nome.startswith("Quintil") else 30)
    setores = _setores(treino)
    sigma = ledoit_wolf_baixo_posto(treino, 252)
    cfg = ConfigOtimizacao(objetivo=Objetivo.KELLY, peso_maximo=0.20, n_ativos_max=10, vol_maxima=0.40, aversao_risco=1.0,
                           grupos=limites_por_setor(setores, 0.30))
    media = media_por_periodo(treino, sigma, 252)
    esperado = {"Kelly-exato": lambda: cl.pesos_utilidade(treino, sigma, cfg, cl.LOG),
                "Utilidade-raiz": lambda: cl.pesos_utilidade(treino, sigma, cfg, cl.RAIZ),
                "Quintil-mu": lambda: cl.pesos_quintil(media, sigma, cfg, "mu"),
                "Quintil-mu-sigma": lambda: cl.pesos_quintil(media, sigma, cfg, "mu_sigma"),
                "Quintil-mu-sigma2": lambda: cl.pesos_quintil(media, sigma, cfg, "mu_sigma2"),
                "Box-robusto": lambda: cl.pesos_box(media * 252, sigma, cfg, len(treino) / 252),
                "Downside-1": lambda: cl.pesos_downside(treino, sigma, cfg, 1),
                "Semi-variancia": lambda: cl.pesos_downside(treino, sigma, cfg, 2),
                "Downside-3": lambda: cl.pesos_downside(treino, sigma, cfg, 3),
                "Retorno-minimo": lambda: cl.pesos_retorno_minimo(media * 252, sigma, cfg)}[nome]()
    assert _bracos()[nome](setores, 252)(treino).to_numpy() == pytest.approx(esperado.reindex(treino.columns).fillna(0.0).to_numpy(), abs=1e-9)
