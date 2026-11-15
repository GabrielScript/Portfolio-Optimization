"""The Composite with a holding band (spec 2026-11-15-composto-faixa): held names stay while in the score's top 20."""

from __future__ import annotations

import pandas as pd

from core.composto_faixa import carteira_com_faixa
from core.sinais import carteira_da_nota

# 30 names, one sector each except where a test sets it; score falls with the number
NOMES = [f"P{i:02d}3" for i in range(1, 31)]
NOTA = pd.Series(range(30, 0, -1), index=NOMES, dtype=float)
SETORES = {n: f"S{i}" for i, n in enumerate(NOMES)}


def test_without_a_previous_book_it_is_the_composites():
    assert list(carteira_com_faixa(NOTA, SETORES, set()).index) == list(carteira_da_nota(NOTA, SETORES, 15, 4).index)


def test_a_held_name_inside_the_band_stays_and_the_last_newcomer_waits():
    book = carteira_com_faixa(NOTA, SETORES, {"P18" + "3"})
    assert "P183" in book.index and "P153" not in book.index
    assert len(book) == 15 and (book == 1 / 15).all()


def test_a_held_name_outside_the_band_leaves():
    book = carteira_com_faixa(NOTA, SETORES, {"P213"})
    assert "P213" not in book.index and list(book.index) == list(carteira_da_nota(NOTA, SETORES, 15, 4).index)


def test_kept_names_count_in_the_sector_cap():
    setores = dict(SETORES)
    for n in ("P013", "P023", "P033", "P173"):
        setores[n] = "Bancos"
    book = carteira_com_faixa(NOTA, setores, {"P173"})
    assert "P173" in book.index
    assert sum(setores[n] == "Bancos" for n in book.index) == 4


def test_the_band_counts_only_names_the_book_could_take():
    # 3 names without a sector above P183 push it to 21st in the raw score but 18th among the names the book can take
    setores = {n: s for n, s in SETORES.items() if n not in ("P013", "P023", "P033")}
    nota = NOTA.copy()
    assert "P213" in carteira_com_faixa(nota, setores, {"P213"}).index
