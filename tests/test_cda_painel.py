"""The CDA best-ideas panel builder reads only the pre-registered lake export."""

import pytest

from scripts.cda_painel import ESPERADOS, conferir_export


def test_the_expected_hashes_are_the_pre_registered_ones():
    assert ESPERADOS == {
        "fund_stock_holdings.parquet": "cd997a5d06f1b41b4988ef0da06518c1a032768380bd2ebee350af5e78b14feb",
        "fund_nav.parquet": "e12d47008b06a5a7eca999c0e00c3554bda862dfaceeacbb6f8893a4896121ee"}


def test_a_file_with_another_hash_is_refused(tmp_path):
    (tmp_path / "fund_nav.parquet").write_bytes(b"not the export")
    with pytest.raises(SystemExit, match="fund_nav.parquet"):
        conferir_export(tmp_path, {"fund_nav.parquet": "e12d47008b06a5a7eca999c0e00c3554bda862dfaceeacbb6f8893a4896121ee"})


def test_the_matching_file_returns_its_hash(tmp_path):
    (tmp_path / "x.parquet").write_bytes(b"abc")
    esperado = "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    assert conferir_export(tmp_path, {"x.parquet": esperado}) == {"x.parquet": esperado}
