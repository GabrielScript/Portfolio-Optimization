from datetime import datetime

import numpy as np
import pandas as pd

from core.sentinela.itens import MACRO, Item
from core.sentinela.pacote import MAX_ITENS, contexto_da_acao, montar_pacote
from core.sentinela.vistos import ler_vistos, marcar, novos_por_acao


def _item(i: int, ticker: str = "PETR4", fonte: str = "cvm", data: str = "2026-11-01") -> Item:
    return Item(f"{fonte}-{i}", fonte, data, ticker, f"texto {i}")


def test_new_items_per_name_are_its_own_and_the_macro_ones_it_has_not_seen(tmp_path):
    itens = [_item(1), _item(2, "VALE3"), _item(3, MACRO, "bcb")]
    arquivo = tmp_path / "vistos.csv"
    assert ler_vistos(arquivo) == set()
    novos = novos_por_acao(itens, ["PETR4", "VALE3", "ITUB4"], set())
    assert [i.id for i in novos["PETR4"]] == ["cvm-1", "bcb-3"]
    assert [i.id for i in novos["ITUB4"]] == ["bcb-3"]
    marcar(arquivo, [("PETR4", "cvm-1"), ("PETR4", "bcb-3")], datetime(2026, 11, 6, 9))
    vistos = ler_vistos(arquivo)
    assert vistos == {("PETR4", "cvm-1"), ("PETR4", "bcb-3")}
    novos = novos_por_acao(itens, ["PETR4", "VALE3"], vistos)
    assert "PETR4" not in novos and [i.id for i in novos["VALE3"]] == ["cvm-2", "bcb-3"]


def test_marking_appends_and_keeps_old_rows(tmp_path):
    arquivo = tmp_path / "vistos.csv"
    marcar(arquivo, [("PETR4", "a")], datetime(2026, 11, 6, 9))
    marcar(arquivo, [("VALE3", "b")], datetime(2026, 11, 6, 10))
    assert ler_vistos(arquivo) == {("PETR4", "a"), ("VALE3", "b")}
    assert arquivo.read_bytes().count(b"\r") == 0


def test_pack_keeps_at_most_8_items_company_first_newest_first_and_the_rest_wait():
    empresa = [_item(i, data=f"2026-10-{10 + i:02d}") for i in range(7)]
    macro = [_item(10 + i, MACRO, "bcb", f"2026-10-{20 + i:02d}") for i in range(3)]
    pacote = montar_pacote("PETR4", "Petróleo", 0.0667, macro + empresa, ["PETR4: último fechamento R$ 30,00"])
    assert len(pacote.itens) == MAX_ITENS
    assert [i.id for i in pacote.itens[:7]] == [f"cvm-{i}" for i in range(6, -1, -1)]
    assert pacote.itens[7].id == "bcb-12"
    assert [i.id for i in pacote.pendentes] == ["bcb-11", "bcb-10"]


def test_pack_text_has_the_header_the_context_and_one_line_per_item_and_evidences_include_context():
    pacote = montar_pacote("PETR4", "Petróleo", 0.0667, [_item(1)], ["linha de contexto"])
    texto = pacote.texto()
    assert "PETR4" in texto and "Petróleo" in texto and "6,67%" in texto
    assert "cvm-1 | cvm | 2026-11-01 | texto 1" in texto and "linha de contexto" in texto
    evidencias = pacote.evidencias()
    assert set(evidencias) == {"cvm-1", "ctx-PETR4-1"} and evidencias["ctx-PETR4-1"].texto == "linha de contexto"


def test_context_is_last_close_21_session_return_and_the_last_record():
    dias = pd.bdate_range("2026-10-01", periods=30)
    fechamentos = pd.DataFrame({"PETR4": np.linspace(20, 30, 30)}, index=dias)
    linhas = contexto_da_acao("PETR4", fechamentos, pd.Timestamp("2026-10-27"), 0.0667)
    assert len(linhas) == 3
    assert linhas[0] == f"PETR4: último fechamento R$ 30,00 em {dias[-1]:%d/%m/%Y}"
    retorno = 30 / fechamentos["PETR4"].iloc[-22] - 1
    assert linhas[1] == f"PETR4: retorno de {100 * retorno:.2f}% nos últimos 21 pregões".replace(".", ",")
    assert linhas[2] == "PETR4: na Composite desde o registro de 27/10/2026, peso alvo 6,67%"
