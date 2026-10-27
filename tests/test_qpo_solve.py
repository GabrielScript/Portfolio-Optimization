"""Kaggle solve job of the QPO (qpo_kaggle/solve.py). Runs where PennyLane and torch are installed (.venv-qpo, Kaggle)."""

import hashlib
import itertools
import json

import numpy as np
import pytest

pytest.importorskip("pennylane")
pytest.importorskip("torch")

from qpo_kaggle.solve import run  # noqa: E402


def _salvar(pasta, fase, data, seed, n=7, k=3):
    rng = np.random.default_rng(seed)
    a = rng.normal(size=(n, n))
    q = (a + a.T) / 2.0
    energias = {c: float(np.isin(np.arange(n), c).astype(float) @ q @ np.isin(np.arange(n), c).astype(float))
                for c in itertools.combinations(range(n), k)}
    melhor = min(energias, key=energias.get)
    destino = pasta / f"fase{fase}"
    destino.mkdir(parents=True, exist_ok=True)
    arquivo = destino / f"{data}.npz"
    np.savez(arquivo, q=q, tickers=np.array([f"P{i}" for i in range(n)]), alpha=np.zeros(n), cov=np.eye(n), k=k,
             exact_picks=np.array(melhor), exact_energy=energias[melhor], exact_worst=max(energias.values()))
    return arquivo


def test_run_writes_one_json_per_qubo_and_p_with_hash_ratio_and_warm_start(tmp_path):
    entrada, saida = tmp_path / "in", tmp_path / "out"
    primeiro = _salvar(entrada, 0, "2010-02-09", seed=1)
    _salvar(entrada, 0, "2010-05-12", seed=2)
    run(entrada, saida, ps=(1,), phases=(0,), device_name="lightning.qubit", require_gpu=False, steps=20, shots=500)
    a = json.loads((saida / "fase0" / "2010-02-09_p1.json").read_text(encoding="utf-8"))
    b = json.loads((saida / "fase0" / "2010-05-12_p1.json").read_text(encoding="utf-8"))
    assert a["sha256"] == hashlib.sha256(primeiro.read_bytes()).hexdigest()
    assert (a["match"], a["ratio"], a["picks"] == a["exact_picks"]) == (True, 1.0, True)
    assert a["init"] is None and b["init"] == a["angles"]          # warm start from the previous rebalance
    assert {"device", "pennylane", "torch", "seed", "seconds", "tickers"} <= a.keys()


def test_run_skips_qubos_already_solved(tmp_path):
    entrada, saida = tmp_path / "in", tmp_path / "out"
    _salvar(entrada, 0, "2010-02-09", seed=1)
    run(entrada, saida, ps=(1,), phases=(0,), device_name="lightning.qubit", require_gpu=False, steps=5, shots=100)
    arquivo = saida / "fase0" / "2010-02-09_p1.json"
    arquivo.write_text(json.dumps({**json.loads(arquivo.read_text(encoding="utf-8")), "marca": 1}), encoding="utf-8")
    run(entrada, saida, ps=(1,), phases=(0,), device_name="lightning.qubit", require_gpu=False, steps=5, shots=100)
    assert json.loads(arquivo.read_text(encoding="utf-8"))["marca"] == 1
