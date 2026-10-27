"""Single-file Kaggle kernels of the QPO (qpo_kaggle/build_kernel.py) and the solve job's input lookup."""

import ast
import json

import pytest

from qpo_kaggle.build_kernel import kernel_source, main


def test_kernel_bakes_the_job_arguments_and_parses():
    fonte = kernel_source("solve.py", ["auto", "/kaggle/working", "1,2,3", "0"])
    ast.parse(fonte)
    assert "sys.argv = ['solve.py', 'auto', '/kaggle/working', '1,2,3', '0']" in fonte
    assert "# KERNEL: inlined" not in fonte and "from __future__" not in fonte


def test_solve_kernel_metadata_mounts_the_qubo_dataset(tmp_path):
    main(["solve", str(tmp_path), "auto", "/kaggle/working", "1,2,3", "0"])
    metadata = json.loads((tmp_path / "kernel-metadata.json").read_text(encoding="utf-8"))
    assert metadata["dataset_sources"] == ["gabrielestrelalopes/qpo-qubos"]
    assert (metadata["enable_gpu"], metadata["machine_shape"], metadata["is_private"]) == ("true", "NvidiaTeslaT4",
                                                                                          "true")


def test_find_input_locates_the_folder_with_sha256sums(tmp_path):
    pytest.importorskip("pennylane")
    from qpo_kaggle.solve import find_input
    alvo = tmp_path / "datasets" / "gabrielestrelalopes" / "qpo-qubos"
    alvo.mkdir(parents=True)
    (alvo / "SHA256SUMS").write_text("", encoding="utf-8")
    assert find_input(tmp_path) == alvo


def test_builder_refuses_a_windows_path_baked_by_git_bash(tmp_path):
    # Git Bash rewrites /kaggle/working to C:/Program Files/Git/kaggle/working unless MSYS_NO_PATHCONV=1
    with pytest.raises(SystemExit, match="MSYS_NO_PATHCONV"):
        main(["solve", str(tmp_path), "auto", "C:/Program Files/Git/kaggle/working", "1,2,3", "0"])
