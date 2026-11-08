"""
mars_indicadores.py - the four technical indicators of the MARS state (docs/superpowers/specs/2026-11-08-mars-design.md,
section 4), on one name's daily series: MACD (12, 26) over the price, RSI, CCI and ADX with Wilder smoothing.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _wilder(serie: pd.Series, janela: int) -> pd.Series:
    return serie.ewm(alpha=1.0 / janela, adjust=False).mean()


def macd_relativo(fechamento: pd.Series) -> pd.Series:
    """(EMA 12 - EMA 26) / price, so that names of any price level share one scale."""
    rapida = fechamento.ewm(span=12, adjust=False).mean()
    lenta = fechamento.ewm(span=26, adjust=False).mean()
    return (rapida - lenta) / fechamento


def rsi(fechamento: pd.Series, janela: int) -> pd.Series:
    """Wilder's RSI in [0, 100]; 100 when there was no fall in the window, 50 when nothing moved."""
    variacao = fechamento.diff()
    alta = _wilder(variacao.clip(lower=0.0), janela)
    baixa = _wilder((-variacao).clip(lower=0.0), janela)
    total = alta + baixa
    return (100.0 * alta / total.where(total > 0)).fillna(50.0)


def cci(maxima: pd.Series, minima: pd.Series, fechamento: pd.Series, janela: int) -> pd.Series:
    """(typical price - its mean) / (0.015 x mean absolute deviation) over the window; 0 when nothing moved."""
    tipico = (maxima + minima + fechamento) / 3.0
    media = tipico.rolling(janela, min_periods=1).mean()
    desvio = tipico.rolling(janela, min_periods=1).apply(lambda janela_: np.mean(np.abs(janela_ - janela_.mean())),
                                                          raw=True)
    return ((tipico - media) / (0.015 * desvio.where(desvio > 0))).fillna(0.0)


def adx(maxima: pd.Series, minima: pd.Series, fechamento: pd.Series, janela: int) -> pd.Series:
    """Wilder's ADX in [0, 100]: the smoothed spread between the directional indices."""
    subida = maxima.diff()
    descida = -minima.diff()
    mais = subida.where((subida > descida) & (subida > 0), 0.0)
    menos = descida.where((descida > subida) & (descida > 0), 0.0)
    anterior = fechamento.shift()
    amplitude = pd.concat([maxima - minima, (maxima - anterior).abs(), (minima - anterior).abs()], axis=1).max(axis=1)
    media_amplitude = _wilder(amplitude, janela)
    di_mais = 100.0 * _wilder(mais, janela) / media_amplitude.where(media_amplitude > 0)
    di_menos = 100.0 * _wilder(menos, janela) / media_amplitude.where(media_amplitude > 0)
    soma = di_mais + di_menos
    dx = (100.0 * (di_mais - di_menos).abs() / soma.where(soma > 0)).fillna(0.0)
    return _wilder(dx.where(subida.notna()), janela)      # the first session has no move: it does not seed the mean
