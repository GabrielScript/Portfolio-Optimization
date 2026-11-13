import pandas as pd

from ui import simulador


def test_the_conservadora_runs_silently_out_of_the_simulators_chart_and_positions():
    # the user, 2026-11-05 (DECISOES 261): the Conservadora leaves the main line, stays in the other tables
    curvas = pd.DataFrame({"Composto": [1.0], "Conservadora": [1.0], "DeepSeek": [1.0], "BOVA11": [1.0],
                           "Selic": [1.0]})
    assert list(simulador.curvas_do_grafico(curvas).columns) == ["Composto", "DeepSeek", "BOVA11", "Selic"]
    assert simulador.carteiras_com_posicoes(["Composto", "Conservadora", "DeepSeek"]) == ["Composto", "DeepSeek"]


def test_the_top5_is_on_the_chart_and_the_caption_says_what_it_is_and_when_it_starts():
    # the user, 2026-11-06 (DECISOES 290): a ghost strategy followed like the others
    curvas = pd.DataFrame({"Composto": [1.0], "Top-5": [1.0], "BOVA11": [1.0]})
    assert list(simulador.curvas_do_grafico(curvas).columns) == ["Composto", "Top-5", "BOVA11"]
    assert simulador.carteiras_com_posicoes(["Composto", "Top-5"]) == ["Composto", "Top-5"]
    assert "Top-5" in simulador.SOBRE_AS_CARTEIRAS and "06/11/2026" in simulador.SOBRE_AS_CARTEIRAS
    assert "uma por setor" in simulador.SOBRE_AS_CARTEIRAS


def test_the_caption_says_what_the_composite_80_20_is_and_when_it_starts():
    # the user, 2026-11-13: the Composite 80/20 is the official strategy, followed as a ghost account too
    assert "Composto 80/20" in simulador.SOBRE_AS_CARTEIRAS and "13/11/2026" in simulador.SOBRE_AS_CARTEIRAS
    assert "BITH11" in simulador.SOBRE_AS_CARTEIRAS and "Bitcoin" in simulador.SOBRE_AS_CARTEIRAS


def test_the_80_20_is_a_ghost_account_not_the_official_strategy():
    # the user, 2026-11-13 ~16:00: the pure Composite stays official on MT5 and in the app; the 80/20 is a ghost
    assert "oficial" not in simulador.SOBRE_AS_CARTEIRAS
