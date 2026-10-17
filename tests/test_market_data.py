"""Calendário B3 e limpeza de preços, só com ações (sem rede)."""

from datetime import date

import numpy as np
import pandas as pd
import pytest

from core import bcb, lago, market_data

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


def test_carregar_precos_le_do_lago_so_os_tickers_pedidos_sem_rede(monkeypatch):
    pedidos: list[list[str]] = []

    def ler_do_lago(tickers, inicio, fim):
        pedidos.append(list(tickers))
        dias = pd.bdate_range(inicio, fim)
        return pd.DataFrame({t: 10.0 for t in tickers}, index=dias)

    def proibido(*a, **k):
        raise AssertionError("não deveria ir à rede")

    monkeypatch.setattr(lago, "ler_fechamentos", ler_do_lago)
    monkeypatch.setattr(market_data, "baixar_fechamentos", proibido)
    monkeypatch.setattr(bcb.requests, "get", proibido)

    dados = market_data.carregar_precos(["AAA.SA", "BBB.SA"], 2, hoje=date(2026, 10, 9))

    assert pedidos == [["AAA.SA", "BBB.SA"]]
    assert list(dados.precos.columns) == ["AAA.SA", "BBB.SA"]
    assert dados.precos.notna().all().all()


def test_sem_o_arquivo_do_lago_o_erro_diz_qual_arquivo_falta(monkeypatch, tmp_path):
    # nada de cair no Yahoo em silêncio: o app mostra o erro
    ausente = tmp_path / "precos_diarios.parquet"
    monkeypatch.setattr(lago, "ARQUIVO_PRECOS", ausente)
    with pytest.raises(market_data.ErroDadosMercado, match="precos_diarios.parquet"):
        market_data.carregar_precos(["PETR4.SA"], 2, hoje=date(2026, 10, 16))


# --- cotações do dia ---------------------------------------------------------------------------

DIARIOS = pd.DataFrame({"A.SA": [9.0, 9.5], "B.SA": [19.0, 19.8], "C.SA": [5.0, np.nan]},
                       index=pd.to_datetime(["2026-10-12", "2026-10-13"]))


def _barras_1m(**fechamentos) -> pd.DataFrame:
    """Saída do yfinance com interval='1m': colunas (campo, ticker) e índice com fuso de São Paulo."""
    indice = pd.date_range("2026-10-14 12:54", periods=3, freq="min", tz="America/Sao_Paulo")
    colunas = pd.MultiIndex.from_product([["Close", "Volume"], list(fechamentos)])
    dados = {("Close", t): v for t, v in fechamentos.items()}
    dados |= {("Volume", t): [100.0] * 3 for t in fechamentos}
    return pd.DataFrame(dados, index=indice)[colunas]


def test_cotacao_e_o_ultimo_negocio_valido_do_dia(monkeypatch):
    chamadas = []

    def download(tickers, **kwargs):
        chamadas.append(kwargs)
        return _barras_1m(**{"A.SA": [10.0, 10.5, 10.6], "B.SA": [20.0, 20.2, np.nan]})

    monkeypatch.setattr(market_data.yf, "download", download)

    cotacoes = market_data.cotacoes_atuais(["A.SA", "B.SA"], DIARIOS)

    assert chamadas[0]["period"] == "1d" and chamadas[0]["interval"] == "1m"
    assert cotacoes.loc["A.SA", "preco"] == 10.6
    assert cotacoes.loc["A.SA", "horario"] == pd.Timestamp("2026-10-14 12:56")
    assert cotacoes.loc["B.SA", "preco"] == 20.2
    assert cotacoes.loc["B.SA", "horario"] == pd.Timestamp("2026-10-14 12:55")
    assert set(cotacoes["fonte"]) == {market_data.FONTE_INTRADIARIA}


def test_ticker_sem_negocio_no_dia_usa_o_ultimo_fechamento(monkeypatch):
    monkeypatch.setattr(market_data.yf, "download",
                        lambda tickers, **kw: _barras_1m(**{"A.SA": [10.0, 10.5, 10.6]}))

    cotacoes = market_data.cotacoes_atuais(["A.SA", "B.SA", "C.SA"], DIARIOS)

    assert cotacoes.loc["A.SA", "fonte"] == market_data.FONTE_INTRADIARIA
    assert cotacoes.loc["B.SA", "preco"] == 19.8
    assert cotacoes.loc["B.SA", "horario"] == pd.Timestamp("2026-10-13")
    assert cotacoes.loc["B.SA", "fonte"] == market_data.FONTE_FECHAMENTO
    assert cotacoes.loc["C.SA", "preco"] == 5.0             # último fechamento VÁLIDO
    assert cotacoes.loc["C.SA", "horario"] == pd.Timestamp("2026-10-12")


def test_falha_do_yahoo_usa_o_fechamento_de_todos(monkeypatch):
    def falha(tickers, **kwargs):
        raise ConnectionError("sem rede")

    monkeypatch.setattr(market_data.yf, "download", falha)

    cotacoes = market_data.cotacoes_atuais(["A.SA", "B.SA"], DIARIOS)

    assert list(cotacoes["preco"]) == [9.5, 19.8]
    assert set(cotacoes["fonte"]) == {market_data.FONTE_FECHAMENTO}


def test_barra_com_preco_zero_e_ignorada_e_vale_o_ultimo_negocio_positivo(monkeypatch):
    monkeypatch.setattr(market_data.yf, "download",
                        lambda tickers, **kw: _barras_1m(**{"A.SA": [10.0, 10.5, 0.0]}))

    cotacoes = market_data.cotacoes_atuais(["A.SA"], DIARIOS)

    assert cotacoes.loc["A.SA", "preco"] == 10.5
    assert cotacoes.loc["A.SA", "horario"] == pd.Timestamp("2026-10-14 12:55")


def test_ticker_so_com_barras_de_preco_zero_usa_o_ultimo_fechamento(monkeypatch):
    monkeypatch.setattr(market_data.yf, "download",
                        lambda tickers, **kw: _barras_1m(**{"B.SA": [0.0, 0.0, 0.0]}))

    cotacoes = market_data.cotacoes_atuais(["B.SA"], DIARIOS)

    assert cotacoes.loc["B.SA", "preco"] == 19.8
    assert cotacoes.loc["B.SA", "fonte"] == market_data.FONTE_FECHAMENTO


def test_ticker_sem_nenhum_preco_fica_nan(monkeypatch):
    monkeypatch.setattr(market_data.yf, "download", lambda tickers, **kw: pd.DataFrame())

    cotacoes = market_data.cotacoes_atuais(["A.SA", "Z.SA"], DIARIOS)

    assert cotacoes.loc["A.SA", "preco"] == 9.5
    assert np.isnan(cotacoes.loc["Z.SA", "preco"])
