"""Stage A of the NPY pre-registration: the IC gate (docs/superpowers/specs/2026-10-29-npy-conservadora-design.md)."""

import numpy as np
import pandas as pd

from scripts.npy_ic import PERIODO, gate

MESES = pd.date_range("2012-06-01", "2026-09-01", freq="MS")


def _ics(media: float, ruido: float = 0.05, semente: int = 0) -> pd.Series:
    gerador = np.random.default_rng(semente)
    valores = gerador.normal(0.0, ruido, len(MESES))
    return pd.Series(valores - valores.mean() + media, index=MESES)


def test_periodo_vai_de_2012_06_a_2026_09():
    assert PERIODO == (pd.Timestamp("2012-06-01"), pd.Timestamp("2026-09-01"))


def test_passa_com_ic_forte_e_npy_acima_do_dy():
    assert gate(_ics(0.05), _ics(0.04, semente=1)) == (True, True)


def test_npy_abaixo_do_dy_reprova_mesmo_com_ic_forte():
    assert gate(_ics(0.05), _ics(0.07, semente=1)) == (True, False)


def test_ic_fraco_reprova_mesmo_acima_do_dy():
    assert gate(_ics(0.01), _ics(0.0, semente=1)) == (False, True)


def test_le_so_os_meses_do_periodo():
    fora = pd.Series(1.0, index=pd.date_range("2008-01-01", "2012-05-01", freq="MS"))
    npy, dy = _ics(0.05), _ics(0.06, semente=1)
    assert gate(pd.concat([fora, npy]), pd.concat([-fora, dy])) == gate(npy, dy) == (True, False)
