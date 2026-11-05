"""core.dolar_bcb: the day's last PTAX bulletin from the BCB's Olinda service, without the network."""

from datetime import date

import pandas as pd
import pytest

from core import dolar_bcb

BOLETINS_04_11 = {"value": [
    {"cotacaoCompra": 5.4663, "cotacaoVenda": 5.4669, "dataHoraCotacao": "2026-11-04 10:07:11.299421",
     "tipoBoletim": "Abertura"},
    {"cotacaoCompra": 5.4749, "cotacaoVenda": 5.4755, "dataHoraCotacao": "2026-11-04 11:09:10.007807",
     "tipoBoletim": "Intermediário"},
    {"cotacaoCompra": 5.4772, "cotacaoVenda": 5.4778, "dataHoraCotacao": "2026-11-04 13:05:19.232051",
     "tipoBoletim": "Fechamento PTAX"}]}


def test_the_last_bulletin_of_the_day_is_the_quote():
    dolar = dolar_bcb.dolar_do_dia(date(2026, 11, 4), lambda dia: BOLETINS_04_11)
    assert dolar == dolar_bcb.DolarDoDia(5.4778, pd.Timestamp("2026-11-04 13:05:19.232051"), "Fechamento PTAX")


def test_before_the_first_bulletin_the_last_day_with_one_stands_in():
    # 05/11 before 10h (and weekends, holidays): no bulletin yet, so the day before's close
    pedidos = []

    def baixar(dia: date) -> dict:
        pedidos.append(dia)
        return BOLETINS_04_11 if dia == date(2026, 11, 4) else {"value": []}

    dolar = dolar_bcb.dolar_do_dia(date(2026, 11, 5), baixar)
    assert dolar.valor == pytest.approx(5.4778) and dolar.boletim == "Fechamento PTAX"
    assert pedidos == [date(2026, 11, 5), date(2026, 11, 4)]


def test_no_bulletin_in_the_whole_window_is_none():
    assert dolar_bcb.dolar_do_dia(date(2026, 11, 5), lambda dia: {"value": []}) is None


def test_the_request_names_the_day_as_the_service_wants_it():
    assert "@dataCotacao='11-05-2026'" in dolar_bcb.url_do_dia(date(2026, 11, 5))
    assert "/olinda/servico/PTAX/" in dolar_bcb.url_do_dia(date(2026, 11, 5))
