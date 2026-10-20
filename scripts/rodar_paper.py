"""
Lançador da rotina do paper trading (a tarefa "paper trading" do Windows chama este arquivo, não o
``paper_trading.py``): importa a rotina dentro de um ``try`` e roda o ``main()`` dela.

Uso: .venv/Scripts/pythonw.exe -X utf8 scripts/rodar_paper.py

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
from datetime import datetime
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
PASTA_ESTADO = Path(os.environ.get("LOCALAPPDATA", Path.home())) / "paper-trading"     # a mesma da rotina
CABECALHO_DO_RUNS = ["inicio", "fim", "fechamento", "desfecho", "nota"]


def anotar_falha(pasta: Path, inicio: datetime, onde: str, erro: Exception) -> Path:
    """Log com o traceback e a linha "falhou" do runs.csv; devolve o caminho do log."""
    log = pasta / "logs" / f"{inicio:%Y%m%d-%H%M%S}-lancador.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    log.write_text(f"{onde}\n" + "".join(traceback.format_exception(erro)), encoding="utf-8")
    runs = pasta / "runs.csv"
    novo = not runs.exists()
    with runs.open("a", encoding="utf-8", newline="") as saida:
        escritor = csv.writer(saida, lineterminator="\n")
        if novo:
            escritor.writerow(CABECALHO_DO_RUNS)
        escritor.writerow([inicio.isoformat(timespec="seconds"), datetime.now().isoformat(timespec="seconds"), "",
                           "falhou", f"{onde}: {type(erro).__name__}: {erro} (log {log.name})"[:300]])
    return log


def main() -> int:
    inicio = datetime.now()
    sys.path.insert(0, str(RAIZ))
    try:
        rotina = importlib.import_module("scripts.paper_trading")
    except Exception as erro:
        anotar_falha(PASTA_ESTADO, inicio, "importação da rotina", erro)
        return 1
    try:
        return rotina.main()
    except Exception as erro:       # o main() da rotina monta o log antes do try dele
        anotar_falha(PASTA_ESTADO, inicio, "main() da rotina", erro)
        return 1


if __name__ == "__main__":
    sys.exit(main())
