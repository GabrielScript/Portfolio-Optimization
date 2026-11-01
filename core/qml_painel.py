"""
qml_painel.py - The monthly panel of the hybrid forecast (docs/superpowers/specs/2026-11-01-qpo-hibrido-design.md): six
angles built from the Composite's inputs and the purged 3-month target. Every feature of month t reads only months before t.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from core.sinais import _meses_antes
from core.valor import NotaDoMes
from core.views import escore_normal

MESES_DA_VOL = 36
MINIMO_DA_VOL = 12
MESES_DO_ALVO = 3
JANELA_DE_TREINO = 120
MAX_LINHAS = 20_000
ATRIBUTOS = ("momentum", "baixa_vol", "dividendos", "formula_magica", "trending_value", "crescimento")


def angulos(nota: pd.Series) -> pd.Series:
    """theta = pi (u - 1/2), u the percentile among the names with a value; a missing value is theta = 0."""
    presentes = nota.dropna()
    percentil = (presentes.rank(method="average") - 0.5) / len(presentes)
    return (np.pi * (percentil - 0.5)).reindex(nota.index).fillna(0.0)


def baixa_volatilidade(retornos: pd.DataFrame, mes: pd.Timestamp) -> pd.Series:
    """Minus the sample deviation of up to 36 monthly returns before ``mes``, with at least 12."""
    janela = _meses_antes(retornos, mes, MESES_DA_VOL)
    return -janela.std(ddof=1).where(janela.notna().sum() >= MINIMO_DA_VOL)


def alvo_3m(retornos: pd.DataFrame, mes: pd.Timestamp, candidatos: Sequence[str]) -> pd.Series:
    """The normal score across ``candidatos`` of the total return of ``mes`` and the next two, over the months a name has:
    one that stops keeps its loss. All NaN until the last of the three months is in the panel."""
    janela = pd.date_range(mes, periods=MESES_DO_ALVO, freq="MS")
    papeis = [papel for papel in candidatos if papel in retornos.columns]
    if janela[-1] > retornos.index[-1]:
        return pd.Series(np.nan, index=papeis, dtype=float)
    retorno = (1.0 + retornos.reindex(janela)[papeis]).prod(min_count=1).sub(1.0)
    return escore_normal(retorno.dropna()).reindex(papeis)


def linhas_de_treino(meses_das_linhas: pd.Series, mes: pd.Timestamp) -> np.ndarray:
    """Positions of the rows that train the forecast of ``mes``: labels already closed (s <= t - 3), the last 120 months, at
    most 20,000 rows drawn by a generator seeded with the month, so every model sees the same rows."""
    meses = pd.DatetimeIndex(meses_das_linhas)
    ultimo = mes - pd.DateOffset(months=MESES_DO_ALVO)
    primeiro = ultimo - pd.DateOffset(months=JANELA_DE_TREINO - 1)
    linhas = np.flatnonzero((meses >= primeiro) & (meses <= ultimo))
    if len(linhas) <= MAX_LINHAS:
        return linhas
    gerador = np.random.default_rng([mes.year, mes.month])
    return np.sort(gerador.choice(linhas, size=MAX_LINHAS, replace=False))


def sem_vazamento(meses_usados: Iterable[pd.Timestamp], mes: pd.Timestamp) -> bool:
    """Every label used had closed by the month before ``mes``."""
    return max(meses_usados) + pd.DateOffset(months=MESES_DO_ALVO - 1) < mes


def painel_de_atributos(notas: Mapping[str, NotaDoMes], retornos: pd.DataFrame, universo: pd.DataFrame,
                        meses: Iterable[pd.Timestamp]) -> pd.DataFrame:
    """Long table, one row per (month, name of that month's universe): the angles of ``ATRIBUTOS`` and the target ``y``."""
    desconhecidos = set(notas) ^ set(ATRIBUTOS)
    if desconhecidos:
        raise ValueError(f"features differ from {ATRIBUTOS}: {sorted(desconhecidos)}")
    por_mes = universo.assign(month=pd.to_datetime(universo["month"])).groupby("month")["security"].apply(list)
    partes = []
    for mes in meses:
        candidatos = [papel for papel in por_mes.get(mes, []) if papel in retornos.columns]
        if not candidatos:
            continue
        tabela = pd.DataFrame({nome: angulos(notas[nome](mes, candidatos).reindex(candidatos)) for nome in ATRIBUTOS})
        partes.append(tabela.assign(y=alvo_3m(retornos, mes, candidatos), month=mes).rename_axis("security").reset_index())
    return pd.concat(partes, ignore_index=True)
