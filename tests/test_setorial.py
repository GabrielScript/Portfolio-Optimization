"""Sector-relative scores of spec 2026-11-16-composto-setorial."""

import pandas as pd
import pytest

from core.setorial import dentro_do_setor

MES = pd.Timestamp("2020-01-01")
SETORES = {**{f"BAN{i}3": "Bancos" for i in range(6)}, **{f"ELE{i}3": "Eletricas" for i in range(6)},
           "PEQ13": "Pequeno", "PEQ23": "Pequeno", "SEMS3": ""}


def _nota(valores: dict[str, float]):
    return lambda mes, candidatos: pd.Series(valores).reindex(list(candidatos)).dropna()


def test_names_are_ranked_against_their_own_sector():
    valores = {**{f"BAN{i}3": 1.0 + i for i in range(6)}, **{f"ELE{i}3": 10.0 + i for i in range(6)}}
    nota = dentro_do_setor(_nota(valores), SETORES)(MES, list(valores))
    assert nota["BAN53"] == pytest.approx(1.0) and nota["ELE53"] == pytest.approx(1.0)   # best of each sector
    assert nota["BAN03"] == pytest.approx(nota["ELE03"])                                 # worst of each sector


def test_small_sector_and_no_sector_keep_the_global_rank():
    valores = {**{f"BAN{i}3": float(i) for i in range(6)}, "PEQ13": 100.0, "PEQ23": -100.0, "SEMS3": 2.5}
    nota = dentro_do_setor(_nota(valores), SETORES)(MES, list(valores))
    global_ = pd.Series(valores).rank(pct=True)
    for papel in ("PEQ13", "PEQ23", "SEMS3"):
        assert nota[papel] == pytest.approx(global_[papel])
    assert nota["BAN53"] == pytest.approx(1.0)


def test_names_without_a_raw_score_stay_out():
    valores = {f"BAN{i}3": float(i) for i in range(6)}
    nota = dentro_do_setor(_nota(valores), SETORES)(MES, [*valores, "ELE03"])
    assert "ELE03" not in nota.index and len(nota) == 6
