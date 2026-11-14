import pandas as pd
import pytest

from scripts import fermionica_grade_avaliar as grade


def test_the_gate_reads_the_raw_p_not_the_printed_three_decimals(monkeypatch, tmp_path, capsys):
    # P 0.9496 prints as 0.950 but is below 0.95: the arm fails
    monkeypatch.setattr(grade, "DATA", tmp_path)
    for fase in grade.FASES:
        pd.DataFrame({"arm": [grade.COMPOSITE, "Arm"], "P[> Comp]": [None, 0.9496], "DSR": [0.99, 0.99],
                      "CAGR": [0.2, 0.2], "fase": fase}).to_csv(tmp_path / f"fermionica-grade-metricas-fase{fase}-2026-11-11.csv",
                                                   index=False)
    grade.summary()
    assert "Arms passing the gate: none" in capsys.readouterr().out


@pytest.mark.parametrize(("fase", "desvio", "cagr", "ok"), [
    (0, 0.0, 0.227, True),
    (0, 1e-6, 0.227, False),        # phase 0 must reproduce the saved curve (spec: 1e-9), not only its CAGR
    (21, 1e-3, 0.2041, True),       # phases 21 / 42: the refiled-statements fix (DECISOES 349) moved decisions
    (42, 1e-3, 0.2100, False),
])
def test_anchor_2_needs_the_exact_curve_in_phase_0(fase, desvio, cagr, ok):
    assert grade.anchor_2_holds(fase, desvio, cagr) is ok
