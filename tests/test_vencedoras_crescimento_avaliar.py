"""The reported lines of the extreme-winners reading (scripts/vencedoras_crescimento_avaliar.py)."""

import pandas as pd


def test_the_picks_profile_counts_stocks_per_rebalance_empty_rebalances_and_the_mean_cash_share():
    from scripts.vencedoras_crescimento_avaliar import picks_profile
    pesos_alvo = pd.DataFrame({"AAAA3": [0.2, 0.0, 0.1], "BBBB3": [0.2, 0.0, 0.0], "CDI": [0.6, 1.0, 0.9]},
                              index=pd.to_datetime(["2015-01-02", "2015-04-02", "2015-07-02"]))
    assert picks_profile(pesos_alvo) == ("stocks per rebalance: min 0, median 1, max 2; rebalances with no pick 1 of 3; "
                                         "mean CDI share 83%")
