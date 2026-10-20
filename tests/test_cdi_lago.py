"""Daily CDI from the lake's copy, topped up from the BCB only for the days the lake does not have yet (no network)."""

from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from core import bcb
from core.cdi_lago import daily_cdi

TODAY = date(2026, 10, 20)  # a Tuesday: the last CDI published is Monday's


def lake_file(tmp_path: Path, days: list[str], pct: float = 0.05) -> Path:
    path = tmp_path / "cdi_daily.parquet"
    pd.DataFrame({"date": pd.to_datetime(days).date, "cdi_pct": pct}).to_parquet(path)
    return path


def no_network(*args):
    raise AssertionError("the BCB must not be asked")


def test_a_current_lake_file_is_the_whole_answer_in_decimal_per_day(tmp_path):
    # one year back plus the 30 days of margin of ``bcb.baixar_cdi_diario``: 2025-01-02 is outside
    path = lake_file(tmp_path, ["2025-01-02", "2026-10-15", "2026-10-16", "2026-10-19"])
    cdi = daily_cdi(1, TODAY, path, fetch_tail=no_network, download_all=no_network)
    assert (cdi.name, cdi.to_dict()) == ("cdi_diario", {pd.Timestamp(d): pytest.approx(0.0005)
                                                         for d in ("2026-10-15", "2026-10-16", "2026-10-19")})


def test_days_after_the_lake_are_asked_to_the_bcb_and_appended(tmp_path):
    path = lake_file(tmp_path, ["2025-01-02", "2026-10-16"])
    asked = []

    def tail(start: date, end: date) -> pd.Series:
        asked.append((start, end))
        return pd.Series([0.0005078], index=pd.to_datetime(["2026-10-19"]))

    cdi = daily_cdi(1, TODAY, path, fetch_tail=tail, download_all=no_network)
    assert (asked, cdi.index[-2:].tolist(), cdi.iloc[-1]) == ([(date(2026, 10, 17), TODAY)],
                                                            [pd.Timestamp("2026-10-16"), pd.Timestamp("2026-10-19")],
                                                            0.0005078)


def test_a_bcb_failure_on_the_tail_keeps_the_lake_series(tmp_path):
    # ``serie_rf_diaria`` carries the last CDI forward over the missing days
    path = lake_file(tmp_path, ["2025-01-02", "2026-10-16"])

    def bcb_down(start: date, end: date) -> pd.Series:
        raise bcb.ErroBcb("BCB SGS 12, janela 2020-2029: Expecting value: line 1 column 1 (char 0)")

    cdi = daily_cdi(1, TODAY, path, fetch_tail=bcb_down, download_all=no_network)
    assert cdi.index[-1] == pd.Timestamp("2026-10-16")


def live(years: int, today: date) -> pd.Series:
    return pd.Series([0.0005], index=pd.to_datetime(["2026-10-19"]), name="cdi_diario")


def test_without_the_lake_file_the_whole_window_comes_from_the_bcb_as_before(tmp_path):
    # Streamlit Cloud: data/ is not versioned, so the app there keeps the live download
    cdi = daily_cdi(1, TODAY, tmp_path / "cdi_daily.parquet", fetch_tail=no_network, download_all=live)
    assert cdi.to_dict() == {pd.Timestamp("2026-10-19"): 0.0005}


def test_a_lake_file_that_starts_after_the_window_is_not_used(tmp_path):
    # a truncated file must not shorten the risk-free series in silence
    path = lake_file(tmp_path, ["2026-10-15", "2026-10-16", "2026-10-19"])
    cdi = daily_cdi(1, TODAY, path, fetch_tail=no_network, download_all=live)
    assert cdi.to_dict() == {pd.Timestamp("2026-10-19"): 0.0005}


@pytest.mark.parametrize("content", [b"", b"PAR1 cut off mid-copy"])
def test_an_unreadable_lake_file_falls_back_to_the_bcb(tmp_path, content):
    # the nightly publish rewrites the file in place: a page load during the copy sees it empty or torn
    path = tmp_path / "cdi_daily.parquet"
    path.write_bytes(content)
    cdi = daily_cdi(1, TODAY, path, fetch_tail=no_network, download_all=live)
    assert cdi.to_dict() == {pd.Timestamp("2026-10-19"): 0.0005}


def test_a_lake_file_with_other_columns_falls_back_to_the_bcb(tmp_path):
    path = tmp_path / "cdi_daily.parquet"
    pd.DataFrame({"date": pd.to_datetime(["2026-10-19"]).date, "rate": 0.05}).to_parquet(path)
    cdi = daily_cdi(1, TODAY, path, fetch_tail=no_network, download_all=live)
    assert cdi.to_dict() == {pd.Timestamp("2026-10-19"): 0.0005}
