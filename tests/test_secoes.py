"""PSR e DSR da tabela do walk-forward (ui/secoes.py)."""

import numpy as np
import pandas as pd
import pytest

from core.backtest import ResultadoBacktest
from core.metrics import momentos_sharpe, sharpe_probabilistico, sharpe_referencia_deflacionado
from ui.secoes import _psr_e_dsr

DATAS = pd.bdate_range("2022-01-03", periods=300)
RF_ZERO = pd.Series(0.0, index=DATAS)


def _backtest(nome: str, valores: np.ndarray) -> ResultadoBacktest:
    return ResultadoBacktest(nome, "walk-forward OOS", pd.Series(valores, index=DATAS),
                             pd.DataFrame(), pd.Series(dtype=float), 0.0)


def _curva(semente: int, media: float) -> np.ndarray:
    retornos = np.random.default_rng(semente).normal(media, 0.01, len(DATAS) - 1)
    return 100 * np.concatenate([[1.0], np.cumprod(1 + retornos)])


def test_estrategia_com_sharpe_nan_nao_anula_o_dsr_das_outras():
    """Curva constante tem Sharpe NaN: o DSR das demais deflaciona só pelas 2 com Sharpe finito."""
    a, b = _backtest("A", _curva(1, 0.001)), _backtest("B", _curva(2, 0.0005))
    resultados = {"A": a, "B": b, "Parada": _backtest("Parada", np.full(len(DATAS), 100.0))}

    saida = _psr_e_dsr(resultados, None, RF_ZERO)

    momentos = {n: momentos_sharpe(bt.valores.pct_change().dropna()) for n, bt in [("A", a), ("B", b)]}
    variancia = float(np.var([m.sharpe for m in momentos.values()], ddof=1))
    referencia = sharpe_referencia_deflacionado(variancia, 2)
    assert saida["A"][1] == pytest.approx(sharpe_probabilistico(momentos["A"], referencia))
    assert saida["B"][1] == pytest.approx(sharpe_probabilistico(momentos["B"], referencia))
    assert np.isnan(saida["Parada"][0]) and np.isnan(saida["Parada"][1])


def test_benchmark_tem_psr_mas_nao_tem_dsr():
    a, b = _backtest("A", _curva(1, 0.001)), _backtest("B", _curva(2, 0.0005))
    saida = _psr_e_dsr({"A": a}, b, RF_ZERO)
    assert np.isfinite(saida["B"][0]) and np.isnan(saida["B"][1])
