"""Registro do paper trading: só acrescenta linhas (o git prova que nada foi reescrito)."""

from datetime import date

import pandas as pd
import pytest

from core.registro import registrar_carteira

PESOS = pd.Series({"PETR4.SA": 0.6, "VALE3.SA": 0.4})
PRECOS = pd.Series({"PETR4.SA": 38.5, "VALE3.SA": 61.25})


def test_registro_so_acrescenta_linhas(tmp_path):
    arquivo = tmp_path / "carteiras.csv"
    registrar_carteira(date(2026, 10, 19), PESOS, PRECOS, arquivo)
    antes = arquivo.read_bytes()
    registrar_carteira(date(2026, 10, 20), PESOS * 0.5, PRECOS, arquivo)
    depois = arquivo.read_bytes()
    assert depois.startswith(antes)
    assert depois[len(antes):].decode("utf-8").splitlines() == ["2026-10-20,PETR4.SA,0.3,38.5",
                                                                "2026-10-20,VALE3.SA,0.2,61.25"]


@pytest.mark.parametrize("data", [date(2026, 10, 20), date(2026, 10, 16)], ids=["repetida", "anterior"])
def test_registro_recusa_data_repetida_ou_anterior(tmp_path, data):
    arquivo = tmp_path / "carteiras.csv"
    registrar_carteira(date(2026, 10, 20), PESOS, PRECOS, arquivo)
    antes = arquivo.read_bytes()
    with pytest.raises(ValueError, match="última registrada é de 20/10/2026"):
        registrar_carteira(data, PESOS, PRECOS, arquivo)
    assert arquivo.read_bytes() == antes
