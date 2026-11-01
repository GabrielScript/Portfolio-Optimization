"""The solver bench's summary (docs/superpowers/specs/2026-11-01-qpo-hibrido-design.md, section 6)."""

import pandas as pd
import pytest


def _linhas() -> pd.DataFrame:
    return pd.DataFrame({"n": [12, 12, 12, 12, 16], "eixo": ["datas", "datas", "datas", "sementes", "datas"],
                         "solver": ["VQE", "VQE", "VQE", "VQE", "VQE"], "etapa": ["cru"] * 5,
                         "viavel": [0.2, 0.4, 0.6, 0.9, 0.1], "p_otimo": [0.0, 0.0, 0.3, 0.1, 0.0],
                         "razao_media": [0.5, 0.6, 0.7, 0.8, 0.4], "razao_melhor": [0.9, 1.0, 1.0, 1.0, 0.8]})


def test_each_axis_and_size_has_its_own_median_and_iqr():
    from core.qml_resumo import resumo_da_bancada

    resumo = resumo_da_bancada(_linhas())
    assert len(resumo) == 3
    linha = resumo[(resumo["n"] == 12) & (resumo["eixo"] == "datas")].iloc[0]
    assert linha["viavel"] == pytest.approx(0.4) and linha["razao_media"] == pytest.approx(0.6)
    assert linha["iqr_razao_media"] == pytest.approx(0.1) and linha["casos"] == 3


def test_a_case_with_no_feasible_shot_stays_counted():
    from core.qml_resumo import resumo_da_bancada

    linhas = _linhas().assign(razao_media=[float("nan"), 0.6, 0.7, 0.8, 0.4])
    assert resumo_da_bancada(linhas).iloc[0]["casos"] == 3
