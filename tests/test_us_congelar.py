"""The freeze of the US exports: an exact copy with a checksum per file, made once."""

import hashlib
import json

import pytest


def _origem(tmp_path):
    origem = tmp_path / "origem"
    (origem / "sub").mkdir(parents=True)
    (origem / "a.parquet").write_bytes(b"aaa")
    (origem / "sub" / "b.parquet").write_bytes(b"bb")
    return origem


def test_it_copies_every_file_and_records_each_checksum(tmp_path):
    from scripts.us_congelar import MANIFESTO, congelar

    destino = tmp_path / "destino"
    congelar(_origem(tmp_path), destino)
    assert (destino / "sub" / "b.parquet").read_bytes() == b"bb"
    manifesto = json.loads((destino / MANIFESTO).read_text(encoding="utf-8"))
    assert manifesto["arquivos"]["a.parquet"] == hashlib.sha256(b"aaa").hexdigest()
    assert manifesto["arquivos"]["sub/b.parquet"] == hashlib.sha256(b"bb").hexdigest()


def test_a_frozen_folder_is_never_written_again(tmp_path):
    from scripts.us_congelar import congelar

    origem, destino = _origem(tmp_path), tmp_path / "destino"
    congelar(origem, destino)
    (origem / "a.parquet").write_bytes(b"mudou")
    with pytest.raises(FileExistsError):
        congelar(origem, destino)
    assert (destino / "a.parquet").read_bytes() == b"aaa"


def test_it_verifies_a_frozen_folder_and_names_the_file_that_changed(tmp_path):
    from scripts.us_congelar import conferir, congelar

    destino = tmp_path / "destino"
    congelar(_origem(tmp_path), destino)
    assert conferir(destino) == []
    (destino / "a.parquet").write_bytes(b"outro")
    (destino / "sub" / "b.parquet").unlink()
    assert conferir(destino) == ["a.parquet", "sub/b.parquet"]


def test_a_copy_cut_short_leaves_no_folder_that_looks_frozen(tmp_path, monkeypatch):
    from scripts import us_congelar

    origem, destino = _origem(tmp_path), tmp_path / "destino"

    def corta(*_):
        raise OSError("disk full")

    monkeypatch.setattr(us_congelar.shutil, "copy2", corta)
    with pytest.raises(OSError):
        us_congelar.congelar(origem, destino)
    assert not destino.exists()
