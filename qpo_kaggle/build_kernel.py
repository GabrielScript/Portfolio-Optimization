"""
build_kernel.py - writes a Kaggle script kernel (folder + kernel-metadata.json) for one QPO job.

Kaggle runs a single file with no arguments, so the kernel carries the ``qpo_kaggle`` modules as text, writes them
to a temporary package, sets ``sys.argv`` and runs the job with ``runpy``: the modules run unchanged, as locally.
The pip bootstrap runs only on Kaggle (``/kaggle`` exists), so the same kernel also runs in a local test.

    MSYS_NO_PATHCONV=1 python qpo_kaggle/build_kernel.py spike OUT_DIR
    MSYS_NO_PATHCONV=1 python qpo_kaggle/build_kernel.py solve OUT_DIR auto /kaggle/working 1,2,3 0 pennylane-gpu
    kaggle kernels push -p OUT_DIR --accelerator NvidiaTeslaT4
"""

import json
import re
import sys
from pathlib import Path

PASTA = Path(__file__).resolve().parent
USUARIO = "gabrielestrelalopes"
QUBOS = f"{USUARIO}/qpo-qubos"          # private dataset with docs/qpo/qubos (scripts/qpo_montar.py)
JOBS = {"spike": "spike", "solve": "solve"}
MODULES = ("__init__.py", "qubo_math.py", "qaoa_subspace.py", "qpo_qaoa.py", "solve.py", "spike.py")
PACOTES = ("pennylane==0.45.1", "pennylane-lightning==0.45.0", "pennylane-lightning-gpu==0.45.0")
MOLDE = '''import os, runpy, subprocess, sys, tempfile
from pathlib import Path

if Path("/kaggle").exists():
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", *{pacotes!r}], check=True)
FONTES = {fontes!r}
raiz = Path(tempfile.mkdtemp())
(raiz / "qpo_kaggle").mkdir()
for nome, texto in FONTES.items():
    (raiz / "qpo_kaggle" / nome).write_text(texto, encoding="utf-8")
sys.path.insert(0, str(raiz))
sys.argv = {argv!r}
runpy.run_module("qpo_kaggle.{modulo}", run_name="__main__")
'''


def kernel_source(job: str, args: list[str] | None = None) -> str:
    fontes = {nome: (PASTA / nome).read_text(encoding="utf-8") for nome in MODULES}
    return MOLDE.format(pacotes=PACOTES, fontes=fontes, argv=[f"{JOBS[job]}.py", *(args or [])], modulo=JOBS[job])


def main(argv: list[str]) -> int:
    job, destino, args = argv[0], Path(argv[1]), argv[2:]
    if any(re.match(r"^[A-Za-z]:[/\\]", arg) for arg in args):
        raise SystemExit(f"Windows path in the kernel arguments {args}: run with MSYS_NO_PATHCONV=1 (Git Bash "
                         "rewrites /kaggle/... to C:/Program Files/Git/kaggle/...)")
    if job == "spike":
        titulo, fontes = "QPO timing spike", []
    else:                                   # one kernel per phase set: "QPO solve 0", "QPO solve 21 42"
        titulo, fontes = f"QPO solve {args[3].replace(',', ' ')}", [QUBOS]
    arquivo = f"{JOBS[job]}.py"
    destino.mkdir(parents=True, exist_ok=True)
    (destino / arquivo).write_text(kernel_source(job, args), encoding="utf-8")
    metadata = {"id": f"{USUARIO}/{titulo.lower().replace(' ', '-')}", "title": titulo, "code_file": arquivo,
                "language": "python", "kernel_type": "script", "is_private": "true", "enable_gpu": "true",
                "enable_tpu": "false", "enable_internet": "true", "machine_shape": "NvidiaTeslaT4",
                "dataset_sources": fontes, "competition_sources": [], "kernel_sources": [], "model_sources": []}
    (destino / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(destino / arquivo)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
