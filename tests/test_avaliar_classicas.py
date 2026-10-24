"""The pre-registered decision of the classic value strategies against the Conservadora."""

import pandas as pd

from scripts.avaliar_classicas import decidir


def _ics(media: float, inicio: str = "2014-01-01", fim: str = "2026-09-01") -> pd.Series:
    meses = pd.date_range(inicio, fim, freq="MS")
    return pd.Series([media + 0.1, media - 0.1] * (len(meses) // 2) + [media] * (len(meses) % 2), index=meses)


def test_gate_reads_only_2014_to_2026_and_the_proof_needs_all_three_phases():
    # a strong 2012-2013 must not rescue a flat 2014-2026: those months are outside the gate
    antes = pd.Series(0.5, index=pd.date_range("2012-01-01", "2013-12-01", freq="MS"))
    assert not decidir(pd.concat([antes, _ics(0.0)]), [0.99, 0.99, 0.99]).aprovada
    assert decidir(_ics(0.05), [0.99, 0.96, 0.95]).candidata
    assert not decidir(_ics(0.05), [0.99, 0.94, 0.99]).candidata
