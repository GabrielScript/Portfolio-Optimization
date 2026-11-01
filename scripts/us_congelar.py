"""
us_congelar.py - Freezes the US exports for the readings: an exact copy with a checksum per file, made once. The lake's exports change
with each gold rebuild; a reading on a frozen folder can be repeated and its inputs proved.

    python -X utf8 scripts/us_congelar.py D:/b3-datalake/exports/us_stocks D:/b3-datalake/exports/qml/us/exports
"""

from __future__ import annotations

import hashlib
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

MANIFESTO = "manifesto.json"
PEDACO = 1 << 20


def _sha256(caminho: Path) -> str:
    resumo = hashlib.sha256()
    with caminho.open("rb") as arquivo:
        while pedaco := arquivo.read(PEDACO):
            resumo.update(pedaco)
    return resumo.hexdigest()


def congelar(origem: Path, destino: Path) -> dict[str, str]:
    """Copies to a temporary folder, hashes the copies and only then gives it the final name, so a folder under that name is whole."""
    if destino.exists():
        raise FileExistsError(f"{destino} is already frozen")
    parcial = destino.with_name(destino.name + ".parcial")
    if parcial.exists():
        shutil.rmtree(parcial)
    try:
        arquivos = {}
        for caminho in sorted(item for item in origem.rglob("*") if item.is_file()):
            relativo = caminho.relative_to(origem)
            copia = parcial / relativo
            copia.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(caminho, copia)
            arquivos[relativo.as_posix()] = _sha256(copia)
        (parcial / MANIFESTO).write_text(json.dumps({"origem": str(origem), "em": datetime.now().isoformat(timespec="seconds"), "arquivos": arquivos}, indent=1), encoding="utf-8")
        parcial.replace(destino)
    except OSError:
        shutil.rmtree(parcial, ignore_errors=True)
        raise
    return arquivos


def conferir(destino: Path) -> list[str]:
    """The manifest's files that are missing or changed; empty if the folder is as frozen."""
    arquivos = json.loads((destino / MANIFESTO).read_text(encoding="utf-8"))["arquivos"]
    return [nome for nome, soma in arquivos.items() if not (destino / nome).is_file() or _sha256(destino / nome) != soma]


if __name__ == "__main__":
    congelados = congelar(Path(sys.argv[1]), Path(sys.argv[2]))
    print(f"{len(congelados)} files frozen at {sys.argv[2]}; changed: {conferir(Path(sys.argv[2]))}")
