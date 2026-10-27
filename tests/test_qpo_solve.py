"""QAOA solve job of the QPO (qpo_kaggle/solve.py). Runs where PennyLane and torch are installed (.venv-qpo, Kaggle)."""

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
    run(entrada, saida, ps=(1,), phases=(0,), steps=20, shots=500)
    a = json.loads((saida / "fase0" / "2010-02-09_p1.json").read_text(encoding="utf-8"))
    b = json.loads((saida / "fase0" / "2010-05-12_p1.json").read_text(encoding="utf-8"))
    assert a["sha256"] == hashlib.sha256(primeiro.read_bytes()).hexdigest()
    assert (a["match"], a["ratio"], a["picks"] == a["exact_picks"]) == (True, 1.0, True)
    assert a["init"] is None and b["init"] == a["angles"]          # warm start from the previous rebalance
    assert {"backend", "device", "torch", "seed", "seconds", "tickers"} <= a.keys()
    assert (a["backend"], a["device"]) == ("subspace", "torch-subspace:cpu")


def test_run_skips_qubos_already_solved(tmp_path):
    entrada, saida = tmp_path / "in", tmp_path / "out"
    _salvar(entrada, 0, "2010-02-09", seed=1)
    run(entrada, saida, ps=(1,), phases=(0,), steps=5, shots=100)
    arquivo = saida / "fase0" / "2010-02-09_p1.json"
    arquivo.write_text(json.dumps({**json.loads(arquivo.read_text(encoding="utf-8")), "marca": 1}), encoding="utf-8")
    run(entrada, saida, ps=(1,), phases=(0,), steps=5, shots=100)
    assert json.loads(arquivo.read_text(encoding="utf-8"))["marca"] == 1


def _sem_relogio(pasta):
    return {a.relative_to(pasta).as_posix(): {c: v for c, v in json.loads(a.read_text(encoding="utf-8")).items()
                                              if c != "seconds"} for a in sorted(pasta.rglob("*.json"))}


def test_parallel_chains_write_the_same_results_as_one_worker(tmp_path):
    # chains (phase, p) are independent; inside a chain the warm start keeps the date order
    entrada = tmp_path / "in"
    for fase, data, semente in ((0, "2010-02-09", 1), (0, "2010-05-12", 2), (21, "2010-03-12", 3)):
        _salvar(entrada, fase, data, seed=semente)
    run(entrada, tmp_path / "um", ps=(1, 2), phases=(0, 21), steps=10, shots=200, workers=1)
    run(entrada, tmp_path / "dois", ps=(1, 2), phases=(0, 21), steps=10, shots=200, workers=2)
    assert _sem_relogio(tmp_path / "um") == _sem_relogio(tmp_path / "dois")
    assert len(_sem_relogio(tmp_path / "um")) == 6


def test_pennylane_reference_backend_gives_the_same_angles(tmp_path):
    entrada = tmp_path / "in"
    _salvar(entrada, 0, "2010-02-09", seed=1)
    run(entrada, tmp_path / "rapido", ps=(1,), phases=(0,), steps=15, shots=200)
    run(entrada, tmp_path / "referencia", ps=(1,), phases=(0,), steps=15, shots=200, backend="pennylane-cpu")
    rapido = json.loads((tmp_path / "rapido" / "fase0" / "2010-02-09_p1.json").read_text(encoding="utf-8"))
    referencia = json.loads((tmp_path / "referencia" / "fase0" / "2010-02-09_p1.json").read_text(encoding="utf-8"))
    assert np.asarray(rapido["angles"]) == pytest.approx(np.asarray(referencia["angles"]), abs=1e-8)
    assert referencia["device"] == "lightning.qubit"


@pytest.mark.parametrize(("campo", "outro"), [("steps", 6), ("shots", 101), ("backend", "pennylane-cpu"),
                                              ("sha256", "0" * 64)])
def test_resume_refuses_a_json_from_another_run_or_qubo(tmp_path, campo, outro):
    # the stored angles warm-start the next rebalance: mixing settings or a changed QUBO would chain silently
    entrada, saida = tmp_path / "in", tmp_path / "out"
    _salvar(entrada, 0, "2010-02-09", seed=1)
    _salvar(entrada, 0, "2010-05-12", seed=2)
    run(entrada, saida, ps=(1,), phases=(0,), steps=5, shots=100)
    arquivo = saida / "fase0" / "2010-02-09_p1.json"
    (saida / "fase0" / "2010-05-12_p1.json").unlink()
    arquivo.write_text(json.dumps({**json.loads(arquivo.read_text(encoding="utf-8")), campo: outro}),
                       encoding="utf-8")
    with pytest.raises(ValueError, match=campo):
        run(entrada, saida, ps=(1,), phases=(0,), steps=5, shots=100)
