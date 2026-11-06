"""The Composite top-5 added to the saved strategies comparison without the 40-minute full run (DECISOES 290)."""

import pandas as pd
import pytest

from scripts.comparacao_estrategias import COMPOSITE, juntar_curva

DATAS = pd.bdate_range("2010-06-01", periods=5)


def _salvas() -> tuple[pd.DataFrame, pd.DataFrame]:
    curvas = pd.DataFrame({COMPOSITE: [100.0, 101.0, 102.0, 103.0, 104.0], "Conservadora": 100.0, "CDI": 0.0004},
                          index=DATAS)
    giro = pd.DataFrame({"giro_anual": [2.9, 2.9], "nomes_medios": 15.0, "recusas": 0.0, "falhas": 0.0},
                        index=pd.Index([COMPOSITE, "Conservadora"], name="estrategia"))
    return curvas, giro


def test_the_new_curve_goes_right_after_the_composite_on_the_saved_dates_with_its_turnover():
    curvas, giro = _salvas()
    composite = curvas[COMPOSITE].copy()
    composite.loc[pd.Timestamp("2026-11-06")] = 110.0                  # a newer session than the saved run: ignored
    nova = pd.Series([100.0, 99.0, 98.0, 97.0, 96.0, 95.0], index=composite.index)
    giro_novo = pd.Series({"giro_anual": 3.4, "nomes_medios": 5.0, "recusas": 0.0, "falhas": 0.0})
    juntas, giro_junto = juntar_curva(curvas, giro, "Composite top-5", composite, nova, giro_novo)
    assert list(juntas.columns) == [COMPOSITE, "Composite top-5", "Conservadora", "CDI"]
    assert juntas.index.equals(DATAS) and juntas["Composite top-5"].tolist() == [100.0, 99.0, 98.0, 97.0, 96.0]
    assert list(giro_junto.index) == [COMPOSITE, "Composite top-5", "Conservadora"]
    assert giro_junto.loc["Composite top-5", "nomes_medios"] == 5.0


def test_a_composite_that_does_not_match_the_saved_one_is_refused():
    # the inputs moved since the saved run (a revised price, a new name): the two curves would not be like with like
    curvas, giro = _salvas()
    outra = curvas[COMPOSITE] * 1.001
    with pytest.raises(ValueError, match="Composite"):
        juntar_curva(curvas, giro, "Composite top-5", outra, outra, pd.Series({"giro_anual": 1.0}))


def test_running_it_again_replaces_the_curve_instead_of_adding_a_second_one():
    curvas, giro = _salvas()
    composite = curvas[COMPOSITE]
    giro_novo = pd.Series({"giro_anual": 3.4, "nomes_medios": 5.0, "recusas": 0.0, "falhas": 0.0})
    uma, giro_um = juntar_curva(curvas, giro, "Composite top-5", composite, composite * 0.9, giro_novo)
    duas, giro_dois = juntar_curva(uma, giro_um, "Composite top-5", composite, composite * 0.8, giro_novo)
    assert list(duas.columns) == list(uma.columns) and duas["Composite top-5"].iloc[0] == pytest.approx(80.0)
    assert list(giro_dois.index) == list(giro_um.index)


def test_the_comparisons_legend_says_what_the_top5_is():
    from ui.comparacao import LEGENDA
    assert "Composite top-5" in LEGENDA and "uma por setor" in LEGENDA
