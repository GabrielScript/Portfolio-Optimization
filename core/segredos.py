"""
segredos.py - Reads the local ``.env`` of ``ai-hedge-fund`` (one level above this project) for the scripts that run on
the user's machine: API keys and the Gmail app password. Values are never printed or logged.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

ARQUIVO_ENV = Path(__file__).resolve().parents[2] / ".env"
_LINHA = re.compile(r"^\s*(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*)$")


def _valor(bruto: str) -> str:
    """The value without quotes; an unquoted value loses a trailing `` # comment``."""
    bruto = bruto.strip()
    if bruto[:1] in ("'", '"'):
        fim = bruto.find(bruto[0], 1)
        return bruto[1:fim] if fim > 0 else bruto[1:]
    return re.split(r"\s+#", bruto, maxsplit=1)[0].strip()


def ler_env(arquivo: Path = ARQUIVO_ENV) -> dict[str, str]:
    """KEY=value pairs of ``arquivo`` (the last one wins, like python-dotenv); empty when the file is missing."""
    if not arquivo.exists():
        return {}
    pares = {}
    for linha in arquivo.read_text(encoding="utf-8-sig").splitlines():
        if (achado := _LINHA.match(linha)) and not linha.lstrip().startswith("#"):
            pares[achado.group(1)] = _valor(achado.group(2))
    return pares


def segredo(nome: str, arquivo: Path = ARQUIVO_ENV) -> str | None:
    """``nome``'s value in ``arquivo``, else the environment variable; None when neither has it. The .env comes first:
    the user keeps the keys there, and a stale Windows variable once shadowed a valid key (2026-11-03)."""
    return ler_env(arquivo).get(nome) or os.environ.get(nome) or None
