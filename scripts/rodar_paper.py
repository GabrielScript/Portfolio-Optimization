"""
Lançador das rotinas do paper trading (as tarefas "paper trading" e "paper v2" do Windows chamam este arquivo, não o
``paper_trading.py`` nem o ``paper_v2.py``): importa a rotina dentro de um ``try`` e roda o ``main()`` dela.

Uso: .venv/Scripts/pythonw.exe -X utf8 scripts/rodar_paper.py            (paper-v1, ``paper_trading.py``)
     .venv/Scripts/pythonw.exe -X utf8 scripts/rodar_paper.py paper_v2   (paper-v2 da Conservadora)

Só biblioteca padrão de propósito. Se a importação da rotina falhar (venv quebrado, pacote faltando, erro de sintaxe),
o ``main()`` dela nem começa e não anota nada; aqui a falha vira um log com o traceback e uma linha "falhou" no
runs.csv, no formato do ``anotar`` da rotina, que o vigia das 12h lê e avisa. Os imports da rotina não podem ir para
dentro do ``main()`` dela: a guarda do paper-v1 compara os imports do projeto no topo do script.
"""

from __future__ import annotations

import csv
import importlib
import os
import sys
import traceback
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
_LOCAL = Path(os.environ.get("LOCALAPPDATA", Path.home()))


@dataclass(frozen=True)
class Rotina:
    """A pasta do runs.csv e o formato do ``anotar`` de cada rotina (a linha "falhou" tem de caber no dela)."""
    pasta: Path
    cabecalho: list[str]
    linha: Callable[[datetime, str], list[str]]      # (início, nota) -> linha "falhou"


ROTINAS = {
    "paper_trading": Rotina(_LOCAL / "paper-trading", ["inicio", "fim", "fechamento", "desfecho", "nota"],
                            lambda inicio, nota: [inicio.isoformat(timespec="seconds"),
                                                  datetime.now().isoformat(timespec="seconds"), "", "falhou", nota]),
    "paper_v2": Rotina(_LOCAL / "paper-v2", ["quando", "fechamento", "desfecho", "nota", "codigo", "guarda"],
                       lambda inicio, nota: [inicio.isoformat(timespec="seconds"), "", "falhou", nota, "", ""]),
}


def anotar_falha(rotina: Rotina, inicio: datetime, onde: str, erro: Exception) -> Path:
    """Log com o traceback e a linha "falhou" do runs.csv da rotina; devolve o caminho do log."""
    pasta = rotina.pasta
    log = pasta / "logs" / f"{inicio:%Y%m%d-%H%M%S}-lancador.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    log.write_text(f"{onde}\n" + "".join(traceback.format_exception(erro)), encoding="utf-8")
    runs = pasta / "runs.csv"
    try:
        novo = not runs.exists()
        with runs.open("a", encoding="utf-8", newline="") as saida:
            escritor = csv.writer(saida, lineterminator="\n")
            if novo:
                escritor.writerow(rotina.cabecalho)
            escritor.writerow(rotina.linha(inicio, f"{onde}: {type(erro).__name__}: {erro} (log {log.name})"[:300]))
    except OSError as travado:      # aberto no Excel: sem a linha, o vigia avisa o dia sem rodada
        with log.open("a", encoding="utf-8") as saida:
            saida.write(f"\nruns.csv sem a linha da falha: {travado}\n")
    return log


def main() -> int:
    inicio = datetime.now()
    nome = sys.argv[1] if len(sys.argv) > 1 else "paper_trading"
    rotina = ROTINAS[nome]
    sys.path.insert(0, str(RAIZ))
    try:
        modulo = importlib.import_module(f"scripts.{nome}")
    except Exception as erro:
        anotar_falha(rotina, inicio, "importação da rotina", erro)
        return 1
    try:
        return modulo.main()
    except Exception as erro:       # o main() da rotina monta o log antes do try dele
        anotar_falha(rotina, inicio, "main() da rotina", erro)
        return 1


if __name__ == "__main__":
    sys.exit(main())
