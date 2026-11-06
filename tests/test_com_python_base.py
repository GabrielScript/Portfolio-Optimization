"""The base-Python launcher hands the venv's packages to the child processes too (DECISOES 286): the upkeep runs
carteira_deepseek.py, the readings and the live curves with ``sys.executable``, the base interpreter."""
import os
import subprocess
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent


def test_children_of_a_launched_script_import_the_venvs_packages(tmp_path):
    script = tmp_path / "filho.py"
    script.write_text("import os, subprocess, sys\n"
                      "r = subprocess.run([sys.executable, '-c', 'import os; print(os.environ.get(\"PYTHONPATH\", \"\"))'],"
                      " capture_output=True, text=True)\n"
                      "print(r.stdout.strip())\n", encoding="utf-8")
    feita = subprocess.run([sys.executable, str(RAIZ / "scripts" / "com_python_base.py"), str(script)],
                           capture_output=True, text=True, cwd=RAIZ, env={k: v for k, v in os.environ.items()
                                                                          if k != "PYTHONPATH"})
    assert feita.returncode == 0, feita.stderr
    caminhos = feita.stdout.strip().split(os.pathsep)
    assert str(RAIZ / ".venv" / "Lib" / "site-packages") in caminhos and str(RAIZ) in caminhos
