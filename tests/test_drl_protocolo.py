from __future__ import annotations

import pandas as pd

from drl import protocolo


def _sessoes(ini: str = "2005-01-03", fim: str = "2026-11-09") -> pd.DatetimeIndex:
    return pd.bdate_range(ini, fim)


def test_each_window_trains_five_years_validates_one_and_tests_one():
    s = _sessoes()
    janelas = protocolo.janelas(s, primeiro_teste=2012, ultimo_teste=2012)
    assert len(janelas) == 1
    j = janelas[0]
    assert set(s[j.treino].year) == {2006, 2007, 2008, 2009, 2010}
    assert set(s[j.validacao].year) == {2011}
    assert set(s[j.teste].year) == {2012}
    assert j.ano_teste == 2012


def test_the_windows_are_in_order_and_each_window_is_disjoint_inside_itself():
    janelas = protocolo.janelas(_sessoes(), primeiro_teste=2012, ultimo_teste=2026)
    anos = [j.ano_teste for j in janelas]
    assert anos == sorted(set(anos))
    for j in janelas:                    # a window reuses an earlier window's test year as validation, never its own
        assert j.treino.stop <= j.validacao.start < j.validacao.stop <= j.teste.start
    for anterior, seguinte in zip(janelas, janelas[1:]):
        assert anterior.treino.start <= seguinte.treino.start      # the training set only grows


def test_a_year_too_thin_to_hold_a_window_is_skipped_not_shortened():
    s = pd.bdate_range("2024-06-03", "2025-06-30")          # barely one year of sessions
    assert protocolo.janelas(s, primeiro_teste=2025, ultimo_teste=2025) == []


def test_the_test_year_is_never_inside_the_training_or_validation():
    s = _sessoes()
    for j in protocolo.janelas(s, primeiro_teste=2012, ultimo_teste=2026):
        assert s[j.treino].max() < s[j.validacao].min()
        assert s[j.validacao].max() < s[j.teste].min()


def test_the_eras_cover_the_whole_tape_without_overlapping():
    s = _sessoes()
    eras = protocolo.eras(s)
    assert list(eras) == ["2005-2012", "2013-2019", "2020-2026"]
    posicoes = [(f.start, f.stop) for f in eras.values()]
    assert posicoes[0][0] == 0
    assert posicoes[-1][1] == len(s)
    for (_, fim), (inicio, _) in zip(posicoes, posicoes[1:]):
        assert fim == inicio
