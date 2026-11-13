"""The Composite 80/20 (spec 2026-11-13-composto-80-20-design): 80% the Composite's record, 20% a Bitcoin ETF."""

from __future__ import annotations

import pandas as pd
import pytest

from core import composto_80_20 as mix


def _composto(*datas: str) -> pd.DataFrame:
    linhas = [(data, f"P{i:02d}3", 1 / 15, 10.0) for data in datas for i in range(15)]
    return pd.DataFrame(linhas, columns=["data", "ticker", "peso", "preco"]).assign(data=lambda t: pd.to_datetime(t["data"]))


def test_the_mix_scales_the_composite_to_80_and_adds_the_etf_at_20() -> None:
    pesos = mix.pesos_mistos(pd.Series(1 / 15, index=[f"P{i:02d}3" for i in range(15)]))
    assert pesos[mix.ETF] == pytest.approx(0.20)
    assert pesos.drop(mix.ETF).tolist() == pytest.approx([0.8 / 15] * 15)
    assert pesos.sum() == pytest.approx(1.0)


def test_the_first_record_is_the_registration_day_on_the_record_in_force() -> None:
    novas = mix.registros_devidos(_composto("2026-10-27"), mix.vazio(), pd.Timestamp("2026-11-13"))
    assert set(novas["data"]) == {mix.INICIO} and set(novas["base"]) == {pd.Timestamp("2026-10-27")}
    assert len(novas) == 16 and novas["peso"].sum() == pytest.approx(1.0)


def test_nothing_is_due_before_the_registration_day() -> None:
    assert mix.registros_devidos(_composto("2026-10-27"), mix.vazio(), pd.Timestamp("2026-11-12")).empty


def test_each_month_start_and_each_new_composite_record_is_a_record_and_none_is_repeated() -> None:
    composto = _composto("2026-10-27", "2027-01-27")
    feito = mix.registros_devidos(composto, mix.vazio(), pd.Timestamp("2026-11-13"))
    novas = mix.registros_devidos(composto, feito, pd.Timestamp("2027-02-03"))
    datas = sorted(novas["data"].unique())
    assert datas == [pd.Timestamp(d) for d in ("2026-12-01", "2027-01-01", "2027-01-27", "2027-02-01")]
    bases = novas.groupby("data")["base"].first()
    assert bases[pd.Timestamp("2027-01-01")] == pd.Timestamp("2026-10-27")
    assert bases[pd.Timestamp("2027-01-27")] == bases[pd.Timestamp("2027-02-01")] == pd.Timestamp("2027-01-27")


def test_the_books_record_key_changes_with_the_month_and_with_the_composite_record() -> None:
    chave, pesos = mix.livro_do_dia(_composto("2026-10-27"), pd.Timestamp("2026-11-13"))
    assert chave == "2026-10-27|2026-11" and pesos[mix.ETF] == pytest.approx(0.20)
    assert mix.livro_do_dia(_composto("2026-10-27"), pd.Timestamp("2026-12-01"))[0] == "2026-10-27|2026-12"
    assert mix.livro_do_dia(_composto("2026-10-27", "2027-01-27"), pd.Timestamp("2027-01-28"))[0] == "2027-01-27|2027-01"


def test_a_composite_record_that_already_holds_the_etf_is_refused() -> None:
    with pytest.raises(ValueError, match=mix.ETF):
        mix.pesos_mistos(pd.Series({"P003": 0.5, mix.ETF: 0.5}))
