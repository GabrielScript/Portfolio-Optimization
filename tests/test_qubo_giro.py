"""The turnover arm of the hybrid QPO (docs/superpowers/specs/2026-11-01-qpo-hibrido-design.md, Prop. 4 and section 4)."""

import numpy as np
import pandas as pd
import pytest

from core.qpo import QuboInstance


def _instancia(n: int = 5, k: int = 2) -> QuboInstance:
    gerador = np.random.default_rng(0)
    bruta = gerador.normal(size=(n, n))
    return QuboInstance(pd.Timestamp("2020-01-31"), tuple(f"P{i}" for i in range(n)), (bruta + bruta.T) / 2.0,
                        np.zeros(n), np.eye(n), k)


def test_the_diagonal_term_is_the_turnover_penalty_of_prop_4():
    from core.qubo_giro import com_giro

    base = _instancia()
    tau = 0.3
    nova = com_giro(base, {"P1", "P3"}, tau)
    h = np.array([0.0, 1.0, 0.0, 1.0, 0.0])
    for bits in ([1, 1, 0, 0, 0], [0, 1, 0, 1, 0], [0, 0, 1, 0, 1]):
        x = np.array(bits, dtype=float)
        assert x @ nova.q @ x - x @ base.q @ x == pytest.approx(tau * np.sum((x - h) ** 2) - tau * h.sum())


def test_tau_is_the_derived_value():
    from core.qubo_giro import TAU

    assert TAU == pytest.approx(4 * 0.001 / 15)


def _mercado(nomes: int = 20, pregoes: int = 300) -> tuple[pd.DataFrame, dict[str, str]]:
    gerador = np.random.default_rng(1)
    retornos = gerador.normal(0.0005, 0.01, (pregoes, nomes))
    colunas = [f"AA{i:02d}3" for i in range(nomes)]
    treino = pd.DataFrame(retornos, columns=colunas, index=pd.bdate_range("2019-01-01", periods=pregoes))
    return treino, {papel: f"S{i % 5}" for i, papel in enumerate(colunas)}


def _nota(mes: pd.Timestamp, candidatos: list[str]) -> pd.Series:
    return pd.Series(np.arange(len(candidatos), 0, -1, dtype=float), index=candidatos)


def _resolver_que_guarda(guardadas: list[np.ndarray]):
    def _resolver(instancia: QuboInstance) -> tuple[int, ...]:
        guardadas.append(instancia.q.copy())
        return tuple(range(15))
    return _resolver


def test_the_next_rebalance_is_charged_for_leaving_what_the_last_one_bought():
    from core.qubo_giro import TAU, estrategia_qubo_giro

    treino, setores = _mercado()
    guardadas: list[np.ndarray] = []
    estrategia = estrategia_qubo_giro(_nota, setores, resolver=_resolver_que_guarda(guardadas))
    primeira = estrategia(treino)
    estrategia(treino)
    assert np.allclose(np.diag(guardadas[1] - guardadas[0]), [-2.0 * TAU] * 15 + [0.0] * 5)
    assert len(primeira) == 15 and primeira.sum() == pytest.approx(1.0) and primeira.nunique() == 1


def test_a_refused_portfolio_was_never_held():
    from dataclasses import replace

    from core.backtest import CarteiraRecusada
    from core.qubo_giro import estrategia_qubo_giro
    from core.risco import LIMITES

    treino, setores = _mercado()
    guardadas: list[np.ndarray] = []
    estrategia = estrategia_qubo_giro(_nota, setores, resolver=_resolver_que_guarda(guardadas),
                                      limites=replace(LIMITES, vol_maxima=1e-6))
    for _ in range(2):
        with pytest.raises(CarteiraRecusada):
            estrategia(treino)
    assert np.allclose(guardadas[0], guardadas[1])


def test_the_annealer_of_record_keeps_the_size_the_sector_count_and_logs_the_two_runs():
    from core.qubo_giro import estrategia_qubo_giro

    treino, setores = _mercado(nomes=24)
    log: dict = {}
    pesos = estrategia_qubo_giro(_nota, setores, log=log)(treino)
    assert len(pesos) == 15
    assert pd.Series([setores[papel] for papel in pesos.index]).value_counts().max() <= 4
    assert list(log) == [treino.index[-1]] and isinstance(log[treino.index[-1]].agree, bool)
