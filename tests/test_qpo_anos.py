"""Year-by-year QPO table against BOVA11, IVVB11 and the crypto ETFs (scripts/qpo_anos.py): descriptive, no gate."""

import numpy as np
import pandas as pd
import pytest

from scripts.qpo_anos import gold_closes, wins


def _gold(tmp_path, linhas):
    arquivo = tmp_path / "equity_adjusted.parquet"
    tabela = pd.DataFrame(linhas, columns=["date", "ticker", "tr_close", "return_break"])
    tabela["date"] = pd.to_datetime(tabela["date"]).dt.date
    tabela.to_parquet(arquivo)
    return arquivo


def test_gold_closes_are_the_total_return_closes_by_date_cut_at_the_last_session(tmp_path):
    arquivo = _gold(tmp_path, [("2021-08-05", "BITH11", 10.0, 0), ("2021-08-06", "BITH11", 11.0, 0),
                               ("2021-08-06", "HASH11", 50.0, 0), ("2021-08-09", "HASH11", 55.0, 0),
                               ("2021-08-09", "BITH11", 12.0, 0), ("2021-08-06", "PETR4", 30.0, 1)])
    precos = gold_closes(arquivo, ("HASH11", "BITH11"), pd.Timestamp("2021-08-06"))
    assert list(precos.columns) == ["HASH11", "BITH11"]
    assert list(precos.index) == list(pd.to_datetime(["2021-08-05", "2021-08-06"]))
    assert precos["BITH11"].tolist() == [10.0, 11.0]
    assert np.isnan(precos.loc["2021-08-05", "HASH11"]) and precos.loc["2021-08-06", "HASH11"] == 50.0


def test_gold_closes_stop_on_a_return_break_or_a_missing_ticker(tmp_path):
    # a break row is a print the lake does not trust: a buy-and-hold through it would carry the bad jump
    arquivo = _gold(tmp_path, [("2021-08-05", "BITH11", 10.0, 0), ("2021-08-06", "BITH11", 40.0, 1)])
    with pytest.raises(ValueError, match="return_break"):
        gold_closes(arquivo, ("BITH11",), pd.Timestamp("2021-08-06"))
    with pytest.raises(ValueError, match="QBTC11"):
        gold_closes(arquivo, ("QBTC11",), pd.Timestamp("2021-08-06"))


def test_wins_count_the_years_the_arm_beats_each_reference_where_both_have_a_return():
    tabela = pd.DataFrame({"QPO": [0.10, 0.05, -0.02, 0.30], "BOVA11": [0.08, 0.07, -0.05, 0.10],
                           "HASH11": [np.nan, np.nan, 0.50, -0.20]}, index=[2021, 2022, 2023, 2024])
    assert wins(tabela, "QPO", ("BOVA11", "HASH11")) == {"BOVA11": (3, 4), "HASH11": (1, 2)}
