"""The live rerun of the Conservadora and the composite (``scripts/curvas_ao_vivo.py``): the hashed snapshot of the
lake's exports it reads and the one table of daily curves the Alepha screen shows."""

import numpy as np
import pandas as pd
import pytest


def test_the_snapshot_copies_the_files_and_writes_hashes_the_frozen_loaders_accept(tmp_path):
    from scripts.avaliar_robustez import conferir_hashes
    from scripts.curvas_ao_vivo import congelar
    origem, destino = tmp_path / "lago", tmp_path / "foto"
    origem.mkdir()
    (origem / "a.parquet").write_bytes(b"um")
    (origem / "b.json").write_bytes(b"dois")
    hashes = congelar(origem, destino, ("a.parquet", "b.json"))
    assert (destino / "a.parquet").read_bytes() == b"um"
    assert conferir_hashes(destino, ["a.parquet", "b.json"]) == hashes


def test_the_snapshot_stops_on_a_missing_export_before_writing_anything(tmp_path):
    from scripts.curvas_ao_vivo import congelar
    (tmp_path / "lago").mkdir()
    with pytest.raises(SystemExit, match="a.parquet"):
        congelar(tmp_path / "lago", tmp_path / "foto", ("a.parquet",))
    assert not (tmp_path / "foto").exists()


def test_the_table_adds_the_crypto_etfs_and_the_cdi_of_each_session_to_attempt_51s_curves():
    from scripts.curvas_ao_vivo import quadro_ao_vivo
    datas = pd.bdate_range("2026-10-26", periods=3)
    curvas = pd.DataFrame({"fase": [0, 0, 0, 1, 1], "date": [*datas, *datas[1:]],
                           "Composto": [1.0, 0.9, 1.0, 1.0, 1.1], "Conservadora": [1.0, 1.1, 1.2, 1.0, 1.1],
                           "BOVA11": [1.0, 1.0, 1.05, 1.0, 1.05]})
    extras = pd.DataFrame({"BITH11": [np.nan, 50.0, 51.0], "QBTC11": [10.0, 10.5, 11.0],
                           "CDI": [0.0005, 0.0005, 0.0006]}, index=datas)
    quadro = quadro_ao_vivo(curvas, extras)
    assert list(quadro.columns) == ["fase", "date", "Conservadora", "Composto", "BOVA11", "BITH11", "QBTC11", "CDI"]
    assert list(quadro["fase"]) == [0, 0, 0, 1, 1]
    linha = quadro[(quadro["fase"] == 1) & (quadro["date"] == datas[2])].iloc[0]
    assert (linha["Composto"], linha["QBTC11"], linha["CDI"]) == (1.1, 11.0, 0.0006)
    assert np.isnan(quadro.iloc[0]["BITH11"])                        # before its listing
