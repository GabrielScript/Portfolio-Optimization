"""Calendário B3 e limpeza de preços, só com ações (sem rede)."""

from datetime import date

import numpy as np
import pandas as pd

from core import bcb, market_data

DIAS = pd.bdate_range("2026-01-05", periods=40)


def _precos(**colunas) -> pd.DataFrame:
    return pd.DataFrame({t: v for t, v in colunas.items()}, index=DIAS)


def test_calendario_descarta_datas_em_que_menos_da_metade_das_acoes_tem_preco():
    a, b, c = np.full(40, 10.0), np.full(40, 20.0), np.full(40, 30.0)
    a[3] = b[3] = c[3] = np.nan          # feriado: ninguém negocia
    b[7] = c[7] = np.nan                 # só 1 de 3 com preço
    grade = market_data.calendario_b3(_precos(**{"A.SA": a, "B.SA": b, "C.SA": c}))
    assert DIAS[3] not in grade and DIAS[7] not in grade
    assert len(grade) == 38


def test_lacuna_curta_e_preenchida_e_a_longa_fica_nan():
    """ffill de até 5 pregões: um ativo parado 7 dias tem os 2 últimos sem preço."""
    longa = np.full(40, 10.0)
    longa[10:17] = np.nan
    curta = np.full(40, 10.0)
    curta[20:23] = np.nan
    dados = market_data.limpar_precos(_precos(**{"L.SA": longa, "C.SA": curta, "X.SA": np.full(40, 5.0)}),
                                      cobertura_minima=0.0)
    assert dados.precos["C.SA"].notna().all()
    assert dados.precos["L.SA"].iloc[10:15].notna().all()
    assert dados.precos["L.SA"].iloc[15:17].isna().all()


def test_ativo_sem_dados_e_ativo_com_cobertura_baixa_sao_descartados_e_reportados():
    pouca = np.full(40, np.nan)
    pouca[:20] = 10.0                    # 50% de cobertura
    dados = market_data.limpar_precos(_precos(**{
        "OK.SA": np.full(40, 10.0), "VAZIO.SA": np.full(40, np.nan), "POUCA.SA": pouca,
        "OUTRA.SA": np.full(40, 7.0)}))
    assert list(dados.precos.columns) == ["OK.SA", "OUTRA.SA"]
    assert dados.descartados_sem_dados == ["VAZIO.SA"]
    assert dados.descartados_cobertura == ["POUCA.SA"]


def test_carregar_precos_de_acoes_baixa_so_os_tickers_pedidos_e_nao_consulta_o_bcb(monkeypatch):
    pedidos: list[list[str]] = []

    def baixar(tickers, inicio, fim):
        pedidos.append(list(tickers))
        dias = pd.bdate_range(inicio, fim)
        return pd.DataFrame({t: 10.0 for t in tickers}, index=dias)

    def proibido(*a, **k):
        raise AssertionError("não deveria consultar o BCB")

    monkeypatch.setattr(market_data, "baixar_fechamentos", baixar)
    monkeypatch.setattr(bcb.requests, "get", proibido)

    dados = market_data.carregar_precos(["AAA.SA", "BBB.SA"], 2, hoje=date(2026, 10, 9))

    assert pedidos == [["AAA.SA", "BBB.SA"]]
    assert list(dados.precos.columns) == ["AAA.SA", "BBB.SA"]
    assert dados.precos.notna().all().all()
