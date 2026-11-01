"""The new arms as engine strategies (docs/superpowers/specs/2026-11-01-todas-as-estrategias-design.md, section 3)."""

import numpy as np
import pandas as pd
import pytest

from core.backtest import CarteiraRecusada


def _treino(linhas: int = 120, colunas: int = 30) -> pd.DataFrame:
    gerador = np.random.default_rng(0)
    dados = gerador.normal(0.0005, 0.008, linhas)[:, None] * gerador.uniform(0.5, 1.5, colunas) + gerador.normal(0.0002, 0.015, (linhas, colunas))
    return pd.DataFrame(dados, columns=[f"N{i:03d}" for i in range(colunas)], index=pd.bdate_range("2019-01-01", periods=linhas))


def _setores(treino: pd.DataFrame) -> dict[str, str]:
    return {papel: f"S{i % 6}" for i, papel in enumerate(treino.columns)}


NOVOS = ("HRP-projetado", "IVarP", "IVolP", "MDivP", "MDecP", "RPP", "HERC", "Mean-CVaR", "Mean-CDaR", "MVSK", "Robust-MV")


def test_the_registry_holds_the_eleven_pre_registered_arms():
    from core.estrategias_todas import BRACOS_NOVOS

    assert tuple(BRACOS_NOVOS) == NOVOS


@pytest.mark.parametrize("nome", NOVOS)
def test_each_arm_is_fully_invested_within_the_caps(nome):
    from core.estrategias_todas import BRACOS_NOVOS

    treino = _treino()
    setores = _setores(treino)
    pesos = BRACOS_NOVOS[nome](setores, 252)(treino)
    assert pesos.sum() == pytest.approx(1.0, abs=1e-6)
    assert pesos.min() >= -1e-9 and pesos.max() <= 0.20 + 1e-6
    assert pesos.groupby(setores).sum().max() <= 0.30 + 1e-6


@pytest.mark.parametrize(("nome", "maximo"), [("MDivP", 10), ("Mean-CVaR", 10), ("Mean-CDaR", 10), ("MVSK", 10), ("Robust-MV", 10), ("HRP-projetado", 10)])
def test_the_concentrated_arms_hold_at_most_ten_names(nome, maximo):
    from core.estrategias_todas import BRACOS_NOVOS

    treino = _treino()
    assert (BRACOS_NOVOS[nome](_setores(treino), 252)(treino) > 1e-9).sum() <= maximo


def test_a_name_without_a_sector_is_not_eligible():
    from core.estrategias_todas import BRACOS_NOVOS

    treino = _treino()
    setores = _setores(treino)
    del setores["N000"]
    assert "N000" not in BRACOS_NOVOS["IVarP"](setores, 252)(treino).index


def test_a_portfolio_that_breaks_the_rules_is_refused():
    from core.estrategias_todas import BRACOS_NOVOS
    from core.risco import Limites

    treino = _treino()
    # a volatility rule below anything possible: the whole-universe arm cannot meet it, so the engine keeps the current portfolio
    with pytest.raises(CarteiraRecusada):
        BRACOS_NOVOS["IVarP"](_setores(treino), 252, Limites(vol_maxima=0.0001))(treino)


def test_the_robust_arm_does_not_depend_on_the_annualization():
    """S = Sigma/T holds per period: the arm must hand the optimizer T in years, or the penalty shrinks by the root of the periods."""
    from core.estrategias_todas import BRACOS_NOVOS
    from core.risco import Limites

    treino = _treino()
    setores = _setores(treino)
    sem_teto_de_vol = Limites(vol_maxima=50.0)
    def pesos(periodos: int) -> np.ndarray:
        return BRACOS_NOVOS["Robust-MV"](setores, periodos, sem_teto_de_vol)(treino).to_numpy()

    assert pesos(252) == pytest.approx(pesos(12), abs=2e-3)


def test_the_monthly_grid_moves_the_dates_and_keeps_the_returns():
    from core.estrategias_todas import no_grid_mensal_real

    treino = _treino(36, 5).set_axis(pd.date_range("2020-01-01", periods=36, freq="ME"))
    visto = {}

    def estrategia(recebido: pd.DataFrame) -> pd.Series:
        visto["treino"] = recebido
        return pd.Series(1.0 / 5, index=recebido.columns)

    no_grid_mensal_real(estrategia)(treino)
    assert visto["treino"].to_numpy() == pytest.approx(treino.to_numpy())
    assert visto["treino"].index[-1] == pd.Timestamp("2022-12-31") + pd.offsets.MonthEnd(1)
