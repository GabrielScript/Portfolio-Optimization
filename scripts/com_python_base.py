"""Runs an app script with the base Python 3.12 and the app venv's packages, the workaround of the lake's
``%LOCALAPPDATA%/b3lake/run_base.py`` (2026-11-03): since about 2026-11-02 the venv's python.exe/pythonw.exe are
denied file writes outside the project folders, while the base interpreter is not.

Usage (Windows task): <Python312>/pythonw.exe -X utf8 scripts/com_python_base.py scripts/rodar_paper.py sentinela
"""

import os
import runpy
import site
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
site.addsitedir(str(RAIZ / ".venv" / "Lib" / "site-packages"))
os.chdir(RAIZ)
sys.path.insert(0, str(RAIZ))
script = sys.argv[1]
sys.argv = [script, *sys.argv[2:]]
runpy.run_path(str(RAIZ / script), run_name="__main__")
