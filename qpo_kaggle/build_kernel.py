"""
build_kernel.py - writes a single-file Kaggle kernel (folder + kernel-metadata.json): a pip bootstrap, then
``qpo_qaoa.py`` inlined, then the job script with its ``# KERNEL: inlined`` import removed.

    python qpo_kaggle/build_kernel.py spike OUT_DIR
    kaggle kernels push -p OUT_DIR --accelerator NvidiaTeslaT4
"""

import json
import sys
from pathlib import Path

PASTA = Path(__file__).resolve().parent
USUARIO = "gabrielestrelalopes"
BOOTSTRAP = '''import subprocess, sys
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "pennylane==0.45.1", "pennylane-lightning==0.45.0",
                "pennylane-lightning-gpu==0.45.0"], check=True)
'''
JOBS = {"spike": ("spike.py", "qpo-timing-spike", "QPO timing spike")}


def kernel_source(job_file: str) -> str:
    solver = (PASTA / "qpo_qaoa.py").read_text(encoding="utf-8").replace("from __future__ import annotations\n", "")
    job = "\n".join(line for line in (PASTA / job_file).read_text(encoding="utf-8").splitlines()
                    if "# KERNEL: inlined" not in line)
    return f"{BOOTSTRAP}\n# ---- qpo_qaoa.py ----\n{solver}\n# ---- {job_file} ----\n{job}\n"


def main(argv: list[str]) -> int:
    job, destino = argv[0], Path(argv[1])
    arquivo, slug, titulo = JOBS[job]
    destino.mkdir(parents=True, exist_ok=True)
    (destino / arquivo).write_text(kernel_source(arquivo), encoding="utf-8")
    metadata = {"id": f"{USUARIO}/{slug}", "title": titulo, "code_file": arquivo, "language": "python",
                "kernel_type": "script", "is_private": "true", "enable_gpu": "true", "enable_tpu": "false",
                "enable_internet": "true", "machine_shape": "NvidiaTeslaT4", "dataset_sources": [],
                "competition_sources": [], "kernel_sources": [], "model_sources": []}
    (destino / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(destino / arquivo)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
