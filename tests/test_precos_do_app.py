"""The deployed app's prices: the committed lake file plus the small file of the sessions after it
(``core.precos_do_app``), so the screen is one session old at most, not one rebalance old."""

import pandas as pd
import pytest

from core.precos_do_app import emendar


def _quadro(datas: list[str], **colunas: list[float]) -> pd.DataFrame:
    return pd.DataFrame(colunas, index=pd.to_datetime(datas))


def test_the_sessions_after_the_base_are_appended():
    base = _quadro(["2026-10-16", "2026-10-19"], PETR4=[57.0, 58.0])
    recentes = _quadro(["2026-10-19", "2026-10-20", "2026-10-21"], PETR4=[58.0, 58.21, 57.75])
    emendado = emendar(base, recentes)
    assert list(emendado.index) == list(pd.to_datetime(["2026-10-16", "2026-10-19", "2026-10-20", "2026-10-21"]))
    assert emendado["PETR4"].tolist() == pytest.approx([57.0, 58.0, 58.21, 57.75])


def test_a_dividend_between_the_two_files_rescales_the_base_at_the_overlap_session():
    # the lake paid R$ 1 on 20/10: its new file rescales 19/10 from 58.00 to 57.00 (and every earlier session alike)
    base = _quadro(["2026-10-16", "2026-10-19"], PETR4=[57.0, 58.0])
    recentes = _quadro(["2026-10-19", "2026-10-20"], PETR4=[57.0, 57.5])
    emendado = emendar(base, recentes)
    assert emendado["PETR4"].tolist() == pytest.approx([57.0 * 57.0 / 58.0, 57.0, 57.5])


def _gravar(caminho, datas: list[str], **colunas: list[float]) -> None:
    quadro = pd.DataFrame({"date": pd.to_datetime(datas).date, **colunas}).astype({nome: "float32" for nome in colunas})
    quadro.to_parquet(caminho, index=False)


def test_the_loader_reads_both_lake_files_in_the_app_format(tmp_path):
    from datetime import date
    from core.precos_do_app import ler_fechamentos
    _gravar(tmp_path / "base.parquet", ["2026-10-16", "2026-10-19"], PETR4=[57.0, 58.0], VALE3=[66.0, 66.36])
    _gravar(tmp_path / "recentes.parquet", ["2026-10-19", "2026-10-20"], PETR4=[58.0, 58.21], VALE3=[66.36, 67.0])
    precos = ler_fechamentos(["PETR4.SA", "VALE3.SA"], date(2026, 10, 17), date(2026, 10, 23),
                             tmp_path / "base.parquet", tmp_path / "recentes.parquet")
    assert list(precos.columns) == ["PETR4.SA", "VALE3.SA"]
    assert list(precos.index) == list(pd.to_datetime(["2026-10-19", "2026-10-20"]))
    assert precos.loc["2026-10-20", "VALE3.SA"] == pytest.approx(67.0)
    assert precos.dtypes.eq("float64").all()


def test_without_the_small_file_the_loader_reads_the_base_alone(tmp_path):
    from datetime import date
    from core.precos_do_app import ler_fechamentos
    _gravar(tmp_path / "base.parquet", ["2026-10-16", "2026-10-19"], PETR4=[57.0, 58.0])
    precos = ler_fechamentos(["PETR4.SA"], date(2026, 10, 1), date(2026, 10, 23), tmp_path / "base.parquet",
                             tmp_path / "ausente.parquet")
    assert precos["PETR4.SA"].tolist() == pytest.approx([57.0, 58.0])


def test_the_cache_stamp_changes_when_only_the_small_file_changes(tmp_path):
    import os
    from core.precos_do_app import versao_dos_dados
    _gravar(tmp_path / "base.parquet", ["2026-10-19"], PETR4=[58.0])
    _gravar(tmp_path / "recentes.parquet", ["2026-10-19"], PETR4=[58.0])
    antes = versao_dos_dados(tmp_path / "base.parquet", tmp_path / "recentes.parquet")
    os.utime(tmp_path / "recentes.parquet", ns=(1, 1))
    assert versao_dos_dados(tmp_path / "base.parquet", tmp_path / "recentes.parquet") != antes


def test_a_stock_without_a_trade_on_the_base_s_last_session_is_rescaled_on_the_last_common_one():
    # review finding 3: FIQE3 did not trade on 19/10 (NaN there) and paid R$ 0.10 on 20/10; the small file starts a
    # few sessions earlier, so both files have 16/10 and the base is rescaled there
    base = _quadro(["2026-10-15", "2026-10-16", "2026-10-19"], FIQE3=[6.0, 6.2, float("nan")])
    recentes = _quadro(["2026-10-15", "2026-10-16", "2026-10-19", "2026-10-20"], FIQE3=[5.9, 6.1, float("nan"), 6.0])
    emendado = emendar(base, recentes)
    assert emendado.loc["2026-10-15", "FIQE3"] == pytest.approx(6.0 * 6.1 / 6.2)
    assert emendado.loc["2026-10-20", "FIQE3"] == pytest.approx(6.0)
