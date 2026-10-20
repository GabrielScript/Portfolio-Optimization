"""
proveniencia.py - De onde saiu cada carteira do paper (``paper/proveniencia.csv``): uma linha por registro, no mesmo
commit dele, com o código que decidiu, o hash do arquivo de preços lido, a Selic usada (valor e fonte) e as versões
do Python e das bibliotecas do cálculo. Com isso, o 1º registro de cada carteira se refaz depois: o golden do
``test_satelite_do_dia_congelado_no_pre_registro`` faz isso só para o de 16/10/2026.
"""

from __future__ import annotations

import csv
import sys
from datetime import date, datetime
from importlib import metadata
from pathlib import Path

# as que mudam o número do satélite (o otimizador, os solvers e as contas); o resto do lock não entra
BIBLIOTECAS = ("pandas", "numpy", "scipy", "scikit-learn", "cvxpy", "clarabel", "pyarrow")
COLUNAS = ["fechamento", "codigo", "precos_sha256", "selic", "python", *BIBLIOTECAS, "gravado_em"]


def _versao(biblioteca: str) -> str:
    try:
        return metadata.version(biblioteca)
    except metadata.PackageNotFoundError:
        return "ausente"


def registrar_proveniencia(arquivo: Path, fechamento: date, codigo: str, precos_sha256: str, selic: str,
                           agora: datetime) -> None:
    """Acrescenta a linha do registro de ``fechamento``; ``selic``: a nota da Selic da rodada (vazia no stop, que
    não otimiza)."""
    novo = not arquivo.exists()
    arquivo.parent.mkdir(parents=True, exist_ok=True)
    with arquivo.open("a", encoding="utf-8", newline="") as saida:
        escritor = csv.writer(saida, lineterminator="\n")
        if novo:
            escritor.writerow(COLUNAS)
        escritor.writerow([fechamento.isoformat(), codigo, precos_sha256, selic,
                           ".".join(map(str, sys.version_info[:3])), *map(_versao, BIBLIOTECAS),
                           agora.isoformat(timespec="seconds")])
