"""
mars_kernel.py - writes one Kaggle script kernel per MARS arm: the kernel carries ``mars.py`` as text, trains the
three seeds as three processes (the 17 years in sequence), chains the test years and leaves one npz per seed in
``/kaggle/working``. The package comes from the private dataset ``mars-pacote``.

    python qpo_kaggle/mars_kernel.py OUT_DIR MARS-50R
    kaggle kernels push -p OUT_DIR/<kernel>
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

PASTA = Path(__file__).resolve().parent
USUARIO = "gabrielestrelalopes"
CONJUNTO = f"{USUARIO}/mars-pacote"
SEMENTES = (42, 0, 1)
MOLDE = '''import subprocess, sys, tempfile
from pathlib import Path

FONTE = {fonte!r}
BRACO, SEMENTES = {braco!r}, {sementes!r}
raiz = Path(tempfile.mkdtemp())
(raiz / "mars.py").write_text(FONTE, encoding="utf-8")
pacote = str(next(Path("/kaggle/input").rglob("mars_pacote.npz")))
saida = Path("/kaggle/working")
comum = ["--pacote", pacote, "--braco", BRACO, "--estados", str(saida / "estados")]
treinos = {{}}
for semente in SEMENTES:
    registro = open(saida / f"treino_s{{semente}}.log", "w")
    treinos[semente] = subprocess.Popen([sys.executable, str(raiz / "mars.py"), "treinar", *comum, "--semente",
                                         str(semente), "--threads", "1"], stdout=registro, stderr=subprocess.STDOUT)
falhas = [semente for semente, processo in treinos.items() if processo.wait() != 0]
if falhas:
    for semente in falhas:
        print((saida / f"treino_s{{semente}}.log").read_text()[-3000:])
    raise SystemExit(f"seeds failed: {{falhas}}")
for semente in SEMENTES:
    subprocess.run([sys.executable, str(raiz / "mars.py"), "encadear", *comum, "--semente", str(semente),
                    "--saida", str(saida)], check=True)
print("done")
'''


def main(argv: list[str]) -> int:
    destino, braco = Path(argv[0]), argv[1]
    titulo = f"MARS {braco.replace('+', ' ')}"
    slug = titulo.lower().replace(" ", "-")
    pasta = destino / slug
    pasta.mkdir(parents=True, exist_ok=True)
    fonte = (PASTA / "mars.py").read_text(encoding="utf-8")
    (pasta / "mars_job.py").write_text(MOLDE.format(fonte=fonte, braco=braco, sementes=SEMENTES), encoding="utf-8")
    metadata = {"id": f"{USUARIO}/{slug}", "title": titulo, "code_file": "mars_job.py", "language": "python",
                "kernel_type": "script", "is_private": "true", "enable_gpu": "true", "enable_tpu": "false",
                "enable_internet": "false", "dataset_sources": [CONJUNTO], "competition_sources": [],
                "kernel_sources": [], "model_sources": [], "machine_shape": "NvidiaTeslaT4"}
    (pasta / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(pasta)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
