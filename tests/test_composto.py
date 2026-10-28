"""The Conservadora's portfolio rule ordered by another score (docs/superpowers/specs/2026-10-28-composto-crescimento-design.md)."""

import numpy as np
import pandas as pd
import pytest

NOMES = ["AAAA3", "BBBB3", "CCCC3", "DDDD3", "EEEE3", "FFFF3", "GGGG3"]


def _treino(nomes: list[str], fim: str = "2021-02-26", pregoes: int = 300, escala: float = 0.01) -> pd.DataFrame:
    gerador = np.random.default_rng(0)
    datas = pd.bdate_range(end=fim, periods=pregoes)
    return pd.DataFrame(gerador.normal(0.0, escala, (pregoes, len(nomes))), index=datas, columns=nomes)


def test_the_strategy_takes_the_top_names_of_the_months_score_among_the_engines_eligible_names_equal_weights():
    from core.composto import estrategia_por_nota
    vistos = []

    def nota(mes, candidatos):
        vistos.append((mes, list(candidatos)))
        return pd.Series(dict(zip(NOMES, [6.0, 1.0, 5.0, 4.0, 3.0, 2.0, 9.0]))).reindex(list(candidatos))

    setores = dict(zip(NOMES, ["S1", "S2", "S3", "S4", "S5", "S6", "S7"]))
    pesos = estrategia_por_nota(nota, setores, nomes=5, por_setor=1)(_treino(NOMES[:6]))     # GGGG3 not eligible
    assert vistos == [(pd.Timestamp("2021-02-01"), NOMES[:6])]
    assert pesos.to_dict() == {nome: pytest.approx(0.2) for nome in ["AAAA3", "CCCC3", "DDDD3", "EEEE3", "FFFF3"]}


def test_a_portfolio_breaking_a_fixed_rule_is_refused_so_the_engine_keeps_the_current_one():
    from core.backtest import CarteiraRecusada
    from core.composto import estrategia_por_nota
    nota = lambda mes, candidatos: pd.Series(range(len(candidatos)), index=list(candidatos), dtype=float)  # noqa: E731
    setores = dict(zip(NOMES, ["S1", "S2", "S3", "S4", "S5", "S6", "S7"]))
    volateis = _treino(NOMES[:5], escala=0.2)                                   # ~300% a year each: vol cap 40%
    with pytest.raises(CarteiraRecusada):
        estrategia_por_nota(nota, setores, nomes=5, por_setor=1)(volateis)
