from datetime import date

import pandas as pd

from core.sentinela.calendario import pascoa, pregao, pregoes


def test_easter_dates():
    assert pascoa(2026) == date(2026, 4, 5) and pascoa(2027) == date(2027, 3, 28)


def test_b3_holidays_are_not_sessions():
    for feriado in (date(2026, 2, 16), date(2026, 2, 17), date(2026, 4, 3), date(2026, 6, 4), date(2026, 11, 20),
                    date(2026, 12, 24), date(2026, 12, 25), date(2026, 12, 31), date(2027, 1, 1), date(2027, 2, 8)):
        assert not pregao(feriado), feriado
    assert pregao(date(2026, 11, 6)) and pregao(date(2026, 12, 19 - 1)) and not pregao(date(2026, 11, 7))


def test_sessions_between_two_dates():
    dias = pregoes(date(2026, 11, 18), date(2026, 11, 24))
    assert list(dias) == list(pd.to_datetime(["2026-11-18", "2026-11-19", "2026-11-23", "2026-11-24"]))
