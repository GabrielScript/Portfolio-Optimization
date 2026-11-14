"""Export of the QPO QUBOs (scripts/qpo_montar.py): one npz per decision, hash-listed for the Kaggle round trip."""

import hashlib

import numpy as np
import pandas as pd
import pytest

from core.qpo import QuboInstance, build_qubo, solve_exact
from scripts.qpo_montar import export_phase, write_sha256sums


def _instancia(data: str) -> QuboInstance:
    rng = np.random.default_rng(3)
    a = rng.normal(size=(7, 7)) * 0.1
    cov = a @ a.T
    alpha = rng.normal(size=7) * 0.02
    return QuboInstance(pd.Timestamp(data), tuple(f"P{i}AB3" for i in range(7)), build_qubo(alpha, cov, 3, 1.0),
                        alpha, cov, 3)


def test_export_writes_one_npz_per_decision_with_the_qubo_and_the_exact_solution(tmp_path):
    instancia = _instancia("2012-05-14")
    exata = solve_exact(instancia.q, 3)
    arquivos = export_phase({instancia.date: (instancia, exata.picks)}, fase=21, pasta=tmp_path)
    assert [arquivo.relative_to(tmp_path).as_posix() for arquivo in arquivos] == ["fase21/2012-05-14.npz"]
    lido = np.load(arquivos[0])
    assert lido["q"] == pytest.approx(instancia.q)
    assert (lido["tickers"].tolist(), int(lido["k"]), tuple(lido["exact_picks"].tolist())) == (
        list(instancia.tickers), 3, exata.picks)
    assert (float(lido["exact_energy"]), float(lido["exact_worst"])) == (pytest.approx(exata.energy),
                                                                         pytest.approx(exata.worst))


def test_sha256sums_lists_every_npz_with_its_hash(tmp_path):
    arquivos = export_phase({pd.Timestamp(d): (_instancia(d), (0, 1, 2)) for d in ("2011-01-10", "2011-04-11")},
                            fase=0, pasta=tmp_path)
    write_sha256sums(tmp_path)
    linhas = (tmp_path / "SHA256SUMS").read_text(encoding="utf-8").splitlines()
    esperadas = [f"{hashlib.sha256(a.read_bytes()).hexdigest()}  {a.relative_to(tmp_path).as_posix()}"
                 for a in sorted(arquivos)]
    assert linhas == esperadas
