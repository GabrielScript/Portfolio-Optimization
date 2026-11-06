"""B3 sessions for the Sentinela's schedule: weekdays minus national holidays, Carnival Monday and Tuesday, Good
Friday, Corpus Christi, and 24 and 31 December (no session). Only an estimate for the hourly polling and the
planned execution day: the account always executes on the first session that is actually in the price files
(``core.sentinela.conta``), so a wrong day here costs one idle run, never a wrong order."""

from __future__ import annotations

from datetime import date, timedelta
from functools import lru_cache

import pandas as pd

FIXOS = ((1, 1), (4, 21), (5, 1), (9, 7), (10, 12), (11, 2), (11, 15), (11, 20), (12, 24), (12, 25), (12, 31))


def pascoa(ano: int) -> date:
    """Easter Sunday (anonymous Gregorian algorithm)."""
    a, b, c = ano % 19, ano // 100, ano % 100
    d, e = divmod(b, 4)
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i, k = divmod(c, 4)
    l = (32 + 2 * e + 2 * i - h - k) % 7          # noqa: E741 - the algorithm's own name
    m = (a + 11 * h + 22 * l) // 451
    mes, dia = divmod(h + l - 7 * m + 114, 31)
    return date(ano, mes, dia + 1)


@lru_cache(maxsize=None)
def feriados(ano: int) -> frozenset[date]:
    domingo = pascoa(ano)
    moveis = {domingo - timedelta(days=48), domingo - timedelta(days=47), domingo - timedelta(days=2),
              domingo + timedelta(days=60)}
    return frozenset({date(ano, mes, dia) for mes, dia in FIXOS} | moveis)


def pregao(dia: date) -> bool:
    return dia.weekday() < 5 and dia not in feriados(dia.year)


def pregoes(inicio: date, fim: date) -> pd.DatetimeIndex:
    dias = pd.bdate_range(inicio, fim)
    return pd.DatetimeIndex([dia for dia in dias if pregao(dia.date())])
