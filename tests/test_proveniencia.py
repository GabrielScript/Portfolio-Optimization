"""Proveniência de cada registro do paper: código, dados, Selic e versões das bibliotecas que decidiram."""

import csv
import sys
from datetime import date, datetime

import pandas as pd

from core.proveniencia import registrar_proveniencia


def test_registro_grava_codigo_hash_selic_e_versoes_numa_linha(tmp_path):
    arquivo = tmp_path / "proveniencia.csv"
    registrar_proveniencia(arquivo, date(2026, 10, 16), "0ccfeeb", "ab" * 32, "Selic 13.75% (fallback 2026-10)",
                           datetime(2026, 10, 18, 20, 53))
    with arquivo.open(encoding="utf-8", newline="") as entrada:
        linhas = list(csv.DictReader(entrada))
    assert len(linhas) == 1
    assert {chave: linhas[0][chave] for chave in ("fechamento", "codigo", "precos_sha256", "selic", "pandas")} == {
        "fechamento": "2026-10-16", "codigo": "0ccfeeb", "precos_sha256": "ab" * 32,
        "selic": "Selic 13.75% (fallback 2026-10)", "pandas": pd.__version__}
    assert linhas[0]["python"] == ".".join(map(str, sys.version_info[:3]))
