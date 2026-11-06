import pandas as pd

from ui import simulador


def test_the_conservadora_runs_silently_out_of_the_simulators_chart_and_positions():
    # the user, 2026-11-05 (DECISOES 261): the Conservadora leaves the main line, stays in the other tables
    curvas = pd.DataFrame({"Composto": [1.0], "Conservadora": [1.0], "DeepSeek": [1.0], "BOVA11": [1.0],
                           "Selic": [1.0]})
    assert list(simulador.curvas_do_grafico(curvas).columns) == ["Composto", "DeepSeek", "BOVA11", "Selic"]
    assert simulador.carteiras_com_posicoes(["Composto", "Conservadora", "DeepSeek"]) == ["Composto", "DeepSeek"]
