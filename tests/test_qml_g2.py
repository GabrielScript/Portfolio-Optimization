"""Tests for the G2 pieces: the daily prediction path and the Composite+BDR selector."""

from __future__ import annotations

import pandas as pd
import pytest

from core.qml import QMLConfig
from scripts.qml_g2_avaliar import probabilidades_de_todas_as_sessoes, seletor_composite_mais_bdr


class _FakePredictor:
    """Predicts a constant per ticker and remembers every date it was asked for."""

    def __init__(self, tickers: list[str]) -> None:
        self.config = QMLConfig()
        self.asked: list[pd.Timestamp] = []
        self._tickers = tickers

    def predict(self, date: pd.Timestamp) -> pd.Series:
        self.asked.append(date)
        return pd.Series(0.5, index=self._tickers)


def _acoes(datas: pd.DatetimeIndex, tickers: list[str]) -> pd.DataFrame:
    return pd.DataFrame(1.0, index=datas, columns=tickers)


def test_daily_path_predicts_every_session_from_the_first_decision(tmp_path) -> None:
    datas = pd.bdate_range("2020-01-01", periods=10)
    acoes = _acoes(datas, ["AAAA3", "BBBB3"])
    predictor = _FakePredictor(["AAAA3", "BBBB3"])
    quadro = probabilidades_de_todas_as_sessoes(predictor, acoes, tmp_path / "c.parquet", janela=3)
    assert list(quadro.index) == list(datas[3:])
    assert predictor.asked == list(datas[3:])
    assert list(quadro.columns) == ["AAAA3", "BBBB3"]


def test_daily_path_uses_the_cache_on_a_second_call(tmp_path) -> None:
    datas = pd.bdate_range("2020-01-01", periods=6)
    acoes = _acoes(datas, ["AAAA3"])
    cache = tmp_path / "c.parquet"
    primeiro = _FakePredictor(["AAAA3"])
    probabilidades_de_todas_as_sessoes(primeiro, acoes, cache, janela=2)
    assert len(primeiro.asked) == 4
    segundo = _FakePredictor(["AAAA3"])
    quadro = probabilidades_de_todas_as_sessoes(segundo, acoes, cache, janela=2)
    assert segundo.asked == []                       # nothing to predict: the cache covers every date
    assert list(quadro.index) == list(datas[2:])


def test_daily_path_never_reads_a_session_after_the_decision(tmp_path) -> None:
    """The predictor is only ever asked for a date that is already in the frame (causal)."""
    datas = pd.bdate_range("2020-01-01", periods=8)
    acoes = _acoes(datas, ["AAAA3"])
    predictor = _FakePredictor(["AAAA3"])
    probabilidades_de_todas_as_sessoes(predictor, acoes, tmp_path / "c.parquet", janela=4)
    assert max(predictor.asked) <= datas[-1]
    assert predictor.asked == sorted(predictor.asked)


def test_bdr_selector_adds_the_eligible_bdrs_and_keeps_them_out_of_the_composite_call() -> None:
    meses = pd.date_range("2024-01-01", periods=1, freq="MS")
    colunas = ["AAAA3", "BBBB3", "CCCC3", "DDDD3", "AAPL34", "MSFT34"]
    treino = pd.DataFrame(0.0, index=pd.bdate_range("2023-01-02", periods=300), columns=colunas)
    setores = {"AAAA3": "A", "BBBB3": "B", "CCCC3": "C", "DDDD3": "D",
               "AAPL34": "EUA", "MSFT34": "EUA"}
    chamadas: list[list[str]] = []

    def nota(mes, papeis):
        chamadas.append(list(papeis))
        return pd.Series({papel: float(i) for i, papel in enumerate(reversed(list(papeis)))},
                         index=list(papeis))

    seletor = seletor_composite_mais_bdr(nota, setores, bdrs=["AAPL34", "MSFT34"], nomes=2, por_setor=1)
    nomes = seletor(meses[0], treino)
    assert "AAPL34" in nomes and "MSFT34" in nomes
    assert set(chamadas[0]) == {"AAAA3", "BBBB3", "CCCC3", "DDDD3"}   # the Composite never sees a BDR
    assert len([n for n in nomes if n not in ("AAPL34", "MSFT34")]) == 2


def test_bdr_selector_skips_a_bdr_with_no_data_in_the_window() -> None:
    treino = pd.DataFrame(0.0, index=pd.bdate_range("2023-01-02", periods=10), columns=["AAAA3"])
    setores = {"AAAA3": "A", "AAPL34": "EUA"}

    def nota(mes, papeis):
        return pd.Series(1.0, index=list(papeis))

    seletor = seletor_composite_mais_bdr(nota, setores, bdrs=["AAPL34"], nomes=1, por_setor=1)
    assert seletor(pd.Timestamp("2024-01-01"), treino) == ["AAAA3"]
