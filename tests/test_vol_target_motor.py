"""The volatility-target replay's pure parts: the share read at each close and the pre-registered gate."""

import numpy as np
import pandas as pd
import pytest

from scripts.vol_target_motor_avaliar import Leitura, aprovado, parte_no_fechamento


def test_share_at_a_close_uses_the_window_ending_at_that_close():
    datas = pd.bdate_range("2024-01-01", periods=30)
    valores = pd.Series(100.0 * np.cumprod(np.r_[1.0, 1.0 + np.tile([0.02, -0.02], 29)[:29]]), index=datas)
    parte = parte_no_fechamento(valores, alvo=0.20, janela=21)
    retornos = valores.pct_change()
    vol = retornos.iloc[datas.get_loc(datas[25]) - 20:datas.get_loc(datas[25]) + 1].std() * np.sqrt(252)
    assert parte.loc[datas[25]] == pytest.approx(min(1.0, 0.20 / vol))
    assert (parte.iloc[:21] == 1.0).all()            # no full window yet: fully invested
    assert parte.index.equals(datas)


def _leitura(mdd_arm: float, cagr_arm: float) -> Leitura:
    return Leitura(cagr_arm=cagr_arm, mdd_arm=mdd_arm, cagr_plain=0.18, mdd_plain=-0.40)


def test_gate_needs_ten_points_less_drawdown_and_at_most_one_point_less_cagr_everywhere():
    boa = _leitura(-0.30, 0.17)
    assert aprovado({(10_000.0, f): boa for f in (0, 21, 42)} | {(100_000.0, f): boa for f in (0, 21, 42)})
    assert not aprovado({(10_000.0, 0): _leitura(-0.31, 0.18)} | {(c, f): boa for c in (10_000.0, 100_000.0)
                                                                  for f in (0, 21, 42) if (c, f) != (10_000.0, 0)})
    assert not aprovado({(100_000.0, 42): _leitura(-0.20, 0.1699)} | {(c, f): boa for c in (10_000.0, 100_000.0)
                                                                      for f in (0, 21, 42) if (c, f) != (100_000.0, 42)})


def test_gate_ignores_one_million_and_refuses_a_missing_cell():
    boa, ruim = _leitura(-0.30, 0.17), _leitura(-0.40, 0.10)
    celulas = {(c, f): boa for c in (10_000.0, 100_000.0) for f in (0, 21, 42)}
    assert aprovado(celulas | {(1_000_000.0, 0): ruim})
    with pytest.raises(ValueError, match="missing"):
        aprovado({k: v for k, v in celulas.items() if k != (100_000.0, 21)})
