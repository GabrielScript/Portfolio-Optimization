"""
python_do_lago.py - The command that runs a ``b3lake.cli`` subcommand from the paper routines (DECISOES 285).

Since about 2026-11-02 an executable inside a project's ``.venv\Scripts`` cannot write outside the project folders
(path-based Windows block, DECISOES 282). The lake's own tasks run ``%LOCALAPPDATA%\b3lake\run_base.py`` with uv's
base Python 3.14 and the lake venv's packages; this module uses that runner when both files exist, else the lake
venv's ``python.exe`` (the old command, kept for a machine without the runner).
"""

from __future__ import annotations

import os
from pathlib import Path

PYTHON_DO_UV = Path("uv") / "python" / "cpython-3.14-windows-x86_64-none" / "python.exe"
RUNNER = Path("b3lake") / "run_base.py"


def comando_do_lago(projeto: Path, *argumentos: str, local: Path | None = None, roaming: Path | None = None
                    ) -> list[str]:
    """``[python, ...] b3lake.cli <argumentos>`` for ``subprocess.run`` with ``cwd=projeto``."""
    local = Path(os.environ.get("LOCALAPPDATA", Path.home())) if local is None else local
    roaming = Path(os.environ.get("APPDATA", Path.home())) if roaming is None else roaming
    runner, python = local / RUNNER, roaming / PYTHON_DO_UV
    if runner.exists() and python.exists():
        return [str(python), str(runner), "b3lake.cli", *argumentos]
    return [str(projeto / ".venv" / "Scripts" / "python.exe"), "-m", "b3lake.cli", *argumentos]
