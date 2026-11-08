"""
mars_kernel.py - writes one Kaggle script kernel per MARS arm and seed (spec section 6): the kernel carries
``mars.py`` as text and runs it with the package from the private dataset ``mars-pacote``.

    python qpo_kaggle/mars_kernel.py OUT_DIR MARS-50 42 [--gpu]
    kaggle kernels push -p OUT_DIR/<kernel>
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

PASTA = Path(__file__).resolve().parent
USUARIO = "gabrielestrelalopes"
CONJUNTO = f"{USUARIO}/mars-pacote"
MOLDE = '''import runpy, sys, tempfile
from pathlib import Path

FONTE = {fonte!r}
raiz = Path(tempfile.mkdtemp())
(raiz / "mars.py").write_text(FONTE, encoding="utf-8")
pacote = next(Path("/kaggle/input").rglob("mars_pacote.npz"))
sys.argv = ["mars.py", "--pacote", str(pacote), "--braco", {braco!r}, "--semente", {semente!r},
            "--saida", "/kaggle/working", "--threads", "4"]
runpy.run_path(str(raiz / "mars.py"), run_name="__main__")
'''


def main(argv: list[str]) -> int:
    destino, braco, semente = Path(argv[0]), argv[1], argv[2]
    gpu = "--gpu" in argv
    titulo = f"MARS {braco.replace('+', ' ')} s{semente}"
    slug = titulo.lower().replace(" ", "-")
    pasta = destino / slug
    pasta.mkdir(parents=True, exist_ok=True)
    fonte = (PASTA / "mars.py").read_text(encoding="utf-8")
    (pasta / "mars_job.py").write_text(MOLDE.format(fonte=fonte, braco=braco, semente=semente), encoding="utf-8")
    metadata = {"id": f"{USUARIO}/{slug}", "title": titulo, "code_file": "mars_job.py", "language": "python",
                "kernel_type": "script", "is_private": "true", "enable_gpu": "true" if gpu else "false",
                "enable_tpu": "false", "enable_internet": "false", "dataset_sources": [CONJUNTO],
                "competition_sources": [], "kernel_sources": [], "model_sources": []}
    if gpu:
        metadata["machine_shape"] = "NvidiaTeslaT4"
    (pasta / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(pasta)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
