"""The app's liquidity snapshot (``scripts/liquidez_do_app.py``): the last closed month of the lake's monthly metrics,
in the final proof's measures (half the quoted spread, daily vol, median daily traded value)."""

from datetime import date

import pandas as pd
import pytest


def test_the_snapshot_keeps_the_last_month_before_the_month_of_the_last_session_in_the_proofs_units():
    from scripts.liquidez_do_app import instantaneo
    metricas = pd.DataFrame({"month": [date(2026, 8, 1), date(2026, 9, 1), date(2026, 10, 1)],
                             "security": ["PETR4"] * 3, "quoted_spread": [0.004, 0.002, 0.010],
                             "vol_daily": [0.02, 0.018, 0.05], "median_dtv_brl": [1e9, 2e9, 3e9]})
    foto = instantaneo(metricas, ate=date(2026, 10, 22))
    assert foto.to_dict("records") == [{"security": "PETR4", "month": date(2026, 9, 1), "meio_spread": 0.001,
                                        "vol_diaria": 0.018, "volume_mediano": 2e9}]
