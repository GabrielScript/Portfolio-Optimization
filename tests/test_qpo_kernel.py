"""Kaggle script kernels of the QPO (qpo_kaggle/build_kernel.py) and the solve job's input lookup."""

import ast
import json
import subprocess
import sys

import numpy as np
import pytest

from qpo_kaggle.build_kernel import MODULES, kernel_source, main


def test_kernel_carries_every_module_bakes_the_arguments_and_parses():
    fonte = kernel_source("solve", ["auto", "/kaggle/working", "1,2,3", "0"])
    ast.parse(fonte)
    assert "sys.argv = ['solve.py', 'auto', '/kaggle/working', '1,2,3', '0']" in fonte
    assert all(repr(modulo) in fonte for modulo in MODULES)


def test_solve_kernel_metadata_mounts_the_qubo_dataset(tmp_path):
    main(["solve", str(tmp_path), "auto", "/kaggle/working", "1,2,3", "0"])
    metadata = json.loads((tmp_path / "kernel-metadata.json").read_text(encoding="utf-8"))
    assert (metadata["id"], metadata["dataset_sources"]) == ("gabrielestrelalopes/qpo-solve-0",
                                                            ["gabrielestrelalopes/qpo-qubos"])
    assert (metadata["enable_gpu"], metadata["machine_shape"], metadata["is_private"]) == ("true", "NvidiaTeslaT4",
                                                                                          "true")


def test_builder_refuses_a_windows_path_baked_by_git_bash(tmp_path):
    # Git Bash rewrites /kaggle/working to C:/Program Files/Git/kaggle/working unless MSYS_NO_PATHCONV=1
    with pytest.raises(SystemExit, match="MSYS_NO_PATHCONV"):
        main(["solve", str(tmp_path), "auto", "C:/Program Files/Git/kaggle/working", "1,2,3", "0"])


def test_find_input_locates_the_folder_with_sha256sums(tmp_path):
    pytest.importorskip("torch")
    from qpo_kaggle.solve import find_input
    alvo = tmp_path / "datasets" / "gabrielestrelalopes" / "qpo-qubos"
    alvo.mkdir(parents=True)
    (alvo / "SHA256SUMS").write_text("", encoding="utf-8")
    assert find_input(tmp_path) == alvo


def test_solve_kernel_runs_end_to_end_outside_the_repository(tmp_path):
    # the kernel file alone (no qpo_kaggle on the path) solves a QUBO folder and writes the JSON
    pytest.importorskip("torch")
    rng = np.random.default_rng(1)
    a = rng.normal(size=(6, 6))
    q = (a + a.T) / 2.0
    (tmp_path / "in" / "fase0").mkdir(parents=True)
    np.savez(tmp_path / "in" / "fase0" / "2010-02-09.npz", q=q, tickers=np.array([f"P{i}" for i in range(6)]),
             alpha=np.zeros(6), cov=np.eye(6), k=3, exact_picks=np.array([0, 1, 2]), exact_energy=0.0,
             exact_worst=1.0)
    kernel = tmp_path / "kernel.py"
    kernel.write_text(kernel_source("solve", [str(tmp_path / "in"), str(tmp_path / "out"), "1", "0", "subspace"]),
                      encoding="utf-8")
    subprocess.run([sys.executable, "-I", str(kernel)], check=True, cwd=tmp_path, capture_output=True)
    resultado = json.loads((tmp_path / "out" / "fase0" / "2010-02-09_p1.json").read_text(encoding="utf-8"))
    assert (resultado["backend"], len(resultado["picks"])) == ("subspace", 3)
