"""
bancada.py - The five-grade scenario Monte Carlo bench (DECISOES 165, design in
``docs/superpowers/specs/2026-10-28-bancada-cenarios-design.md``): month-end grades from BOVA11's trend and volatility
and the CDI's change, a Markov chain of grades, and paths made of whole historical months drawn by grade, the same
month for every sleeve. Descriptive: no gate, no trial; history is resampled, not forecast.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

DIAS_UTEIS_ANO = 252
MESES_DA_TENDENCIA = 12
PREGOES_DA_VOL = 63
MESES_DO_CDI = 6


def cdi_anual(cdi_pct: pd.Series) -> pd.Series:
    """The CDI in percent per day as an annual rate."""
    return (1.0 + cdi_pct / 100.0) ** DIAS_UTEIS_ANO - 1.0


def componentes(bova: pd.Series, cdi_pct: pd.Series, fins: pd.DatetimeIndex) -> pd.DataFrame:
    """At each date of ``fins`` (the last value on or before it): BOVA11's total return over the last 12 months
    (``trend``), the annualized standard deviation of its last 63 daily log returns (``vol``), and the annual CDI
    minus the annual CDI 6 months earlier (``dcdi``)."""
    log_retornos = np.log(bova).diff().dropna()
    anual = cdi_anual(cdi_pct)
    linhas = {fim: {"trend": bova.asof(fim) / bova.asof(fim - pd.DateOffset(months=MESES_DA_TENDENCIA)) - 1.0,
                    "vol": log_retornos.loc[:fim].iloc[-PREGOES_DA_VOL:].std() * np.sqrt(DIAS_UTEIS_ANO),
                    "dcdi": anual.asof(fim) - anual.asof(fim - pd.DateOffset(months=MESES_DO_CDI))}
              for fim in fins}
    return pd.DataFrame.from_dict(linhas, orient="index")


SINAIS = {"trend": 1.0, "vol": -1.0, "dcdi": -1.0}     # favorable: rising market, low vol, falling rates
GRAUS = 5


def notas(comp: pd.DataFrame, janela: pd.DataFrame | None = None) -> pd.Series:
    """z(trend) - z(vol) - z(dcdi), each z with the mean and standard deviation of ``janela`` (default ``comp``)."""
    base = comp if janela is None else janela
    return sum(sinal * (comp[nome] - base[nome].mean()) / base[nome].std() for nome, sinal in SINAIS.items())


def graus(comp: pd.DataFrame) -> pd.Series:
    """The quintile of ``notas`` over ``comp`` itself: 1 = pessimistic ... 5 = optimistic (in-sample labeling)."""
    return pd.qcut(notas(comp), GRAUS, labels=range(1, GRAUS + 1)).astype(int)


def cortes(janela: pd.DataFrame) -> np.ndarray:
    """The quintile cut points of the window's scores (``graus``'s bins, right-closed)."""
    return notas(janela).quantile(np.arange(1, GRAUS) / GRAUS).to_numpy()


def grau_de(comp: pd.DataFrame, janela: pd.DataFrame) -> pd.Series:
    """The grade of each row of ``comp`` with the window's z and quintile cut points (``graus``'s rule)."""
    return pd.Series(np.searchsorted(cortes(janela), notas(comp, janela).to_numpy(), side="left") + 1,
                     index=comp.index)


@dataclass(frozen=True)
class Resumo:
    """Each complete calendar month (row) of each sleeve (column), in log terms from the month's starting value (the
    last value before it): ``fim`` the month's log return, ``minimo``/``maximo`` the lowest/highest running level
    (0 included), ``dd`` the largest fall from a running peak inside the month."""
    meses: pd.DatetimeIndex
    sleeves: tuple[str, ...]
    fim: np.ndarray
    minimo: np.ndarray
    maximo: np.ndarray
    dd: np.ndarray


def resumir_meses(valores: pd.DataFrame) -> Resumo:
    """``Resumo`` of the daily ``valores`` (sessions x sleeves) over the months after the first value's month (the
    first value is their starting point); the caller cuts ``valores`` at a month end so the last month is complete."""
    log_diario = np.log(valores).diff().iloc[1:]
    mes_de = log_diario.index.to_period("M").to_timestamp()
    meses = pd.DatetimeIndex(mes_de.unique())
    meses = meses[meses > valores.index[0].to_period("M").to_timestamp()]
    partes = {campo: [] for campo in ("fim", "minimo", "maximo", "dd")}
    for mes in meses:
        nivel = np.vstack([np.zeros((1, valores.shape[1])), np.cumsum(log_diario[mes_de == mes].to_numpy(), axis=0)])
        partes["fim"].append(nivel[-1])
        partes["minimo"].append(nivel.min(axis=0))
        partes["maximo"].append(nivel.max(axis=0))
        partes["dd"].append((np.maximum.accumulate(nivel, axis=0) - nivel).max(axis=0))
    return Resumo(meses, tuple(valores.columns), *(np.array(partes[campo]) for campo in partes))


def transicoes(graus_inicio: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Counts (GRAUS x GRAUS) of the grade a month started in -> the grade the next month started in, and the
    row-normalized matrix; a grade with no next month stays in itself."""
    contagem = np.zeros((GRAUS, GRAUS), dtype=int)
    np.add.at(contagem, (graus_inicio[:-1] - 1, graus_inicio[1:] - 1), 1)
    linhas = contagem.sum(axis=1, keepdims=True)
    matriz = np.where(linhas > 0, contagem / np.maximum(linhas, 1), np.eye(GRAUS))
    return contagem, matriz


def sortear(graus_inicio: np.ndarray, matriz: np.ndarray, grau0: int, n_caminhos: int, meses: int,
            gerador: np.random.Generator) -> np.ndarray:
    """Paths x months of historical month indexes: month 1 uniform among the months that started in ``grau0``; then
    each path's next grade from ``matriz`` and a month uniform among that grade's months."""
    reservas = {grau: np.flatnonzero(graus_inicio == grau) for grau in range(1, GRAUS + 1)}
    acumulada = matriz.cumsum(axis=1)
    grau = np.full(n_caminhos, grau0)
    sorteio = np.empty((n_caminhos, meses), dtype=int)
    for coluna in range(meses):
        if coluna:
            grau = np.minimum((gerador.random(n_caminhos)[:, None] > acumulada[grau - 1]).sum(axis=1) + 1, GRAUS)
        for valor in np.unique(grau):
            quem = grau == valor
            sorteio[quem, coluna] = gerador.choice(reservas[valor], size=int(quem.sum()))
    return sorteio


def estatisticas(terminal: np.ndarray, mdd: np.ndarray, sleeves: tuple[str, ...],
                 pares: list[tuple[str, str]]) -> dict[str, float]:
    """Over the paths: P[a > b] (terminal) for each pair, the 5/50/95% terminal return quantiles and the drawdown's
    median and worst 5% (its 95% quantile) for each sleeve."""
    coluna = {nome: posicao for posicao, nome in enumerate(sleeves)}
    linha = {f"P[{a} > {b}]": float(np.mean(terminal[:, coluna[a]] > terminal[:, coluna[b]])) for a, b in pares}
    for nome, posicao in coluna.items():
        for q in (5, 50, 95):
            linha[f"{nome} ret p{q}"] = float(np.quantile(terminal[:, posicao], q / 100))
        linha[f"{nome} MDD p50"] = float(np.quantile(mdd[:, posicao], 0.5))
        linha[f"{nome} MDD worst 5%"] = float(np.quantile(mdd[:, posicao], 0.95))
    return linha


def caminhos(resumo: Resumo, sorteio: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Terminal return and maximum drawdown (paths x sleeves) of paths made of the months ``sorteio`` (paths x
    months, indexes into ``resumo.meses``), exact: with the gap g to the running peak entering a month, the month's
    worst fall is max(g - minimo, dd) and the gap leaving it is max(g, maximo) - fim."""
    lacuna = np.zeros((sorteio.shape[0], len(resumo.sleeves)))
    pior, total = np.zeros_like(lacuna), np.zeros_like(lacuna)
    for coluna in range(sorteio.shape[1]):
        mes = sorteio[:, coluna]
        pior = np.maximum(pior, np.maximum(lacuna - resumo.minimo[mes], resumo.dd[mes]))
        lacuna = np.maximum(lacuna, resumo.maximo[mes]) - resumo.fim[mes]
        total += resumo.fim[mes]
    return np.expm1(total), -np.expm1(-pior)
