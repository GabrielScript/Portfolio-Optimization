"""The saved curves of the composite reading (scripts/composto_crescimento_avaliar.py), the bench's input."""

import pandas as pd


def test_the_curves_frame_has_one_row_per_phase_and_session_with_the_arm_the_conservadora_and_bova11():
    from scripts.composto_crescimento_avaliar import ARM, curves_frame
    from scripts.qpo_avaliar import CONSERVADORA
    datas = pd.to_datetime(["2010-02-09", "2010-02-10"])
    fase0 = {ARM: pd.Series([1.0, 1.1], index=datas), CONSERVADORA: pd.Series([1.0, 1.2], index=datas),
             "BOVA11": pd.Series([1.0, 0.9], index=datas), "IVVB11": pd.Series([1.0, 1.0], index=datas)}
    quadro = curves_frame([(0, fase0), (21, fase0)])
    assert list(quadro.columns) == ["fase", "date", "Composto", "Conservadora", "BOVA11"]
    assert quadro["fase"].tolist() == [0, 0, 21, 21]
    assert quadro["Conservadora"].tolist() == [1.0, 1.2, 1.0, 1.2]
