"""Copy the top-return equity funds (docs/superpowers/specs/2026-10-29-top-fundos-copia-design.md)."""

import pandas as pd
import pytest

from core.top_fundos import nota_media, retorno_12m, retornos_mensais, top_fundos

T = pd.Timestamp


def _cotas(linhas: list[tuple]) -> pd.DataFrame:
    """(cnpj, subclass, month, last report day, quota, nav) rows, as the lake's fund_monthly_quota export."""
    quadro = pd.DataFrame(linhas, columns=["cnpj", "subclass", "month", "date", "quota", "nav"])
    return quadro.assign(month=pd.to_datetime(quadro["month"]), date=pd.to_datetime(quadro["date"]))


def _r(retornos: pd.DataFrame, cnpj: str, mes: str) -> float:
    linha = retornos[(retornos["cnpj"] == cnpj) & (retornos["month"] == T(mes))]
    return float(linha["r"].iloc[0]) if len(linha) else float("nan")


def test_a_return_divides_quotas_of_one_series_and_prefers_the_class_row():
    cotas = _cotas([("A", None, "2024-01-01", "2024-01-31", 100.0, 9.0),
                    ("A", None, "2024-02-01", "2024-02-29", 110.0, 9.0),
                    ("A", "S1", "2024-01-01", "2024-01-31", 50.0, 99.0),
                    ("A", "S1", "2024-02-01", "2024-02-29", 80.0, 99.0)])
    assert _r(retornos_mensais(cotas), "A", "2024-02-01") == pytest.approx(0.10)


def test_a_class_with_only_subclasses_takes_the_largest_nav_at_the_previous_month_among_those_quoted_both_months():
    cotas = _cotas([("B", "S1", "2024-01-01", "2024-01-31", 10.0, 500.0),
                    ("B", "S1", "2024-02-01", "2024-02-29", 11.0, 100.0),
                    ("B", "S2", "2024-01-01", "2024-01-31", 20.0, 300.0),
                    ("B", "S2", "2024-02-01", "2024-02-29", 30.0, 900.0),
                    ("B", "S3", "2024-02-01", "2024-02-29", 1.0, 5000.0)])
    assert _r(retornos_mensais(cotas), "B", "2024-02-01") == pytest.approx(0.10)    # S1: largest nav in January


def test_a_month_end_quota_older_than_7_days_makes_that_month_and_the_next_missing():
    cotas = _cotas([("C", None, "2024-01-01", "2024-01-31", 100.0, 1.0),
                    ("C", None, "2024-02-01", "2024-02-20", 101.0, 1.0),
                    ("C", None, "2024-03-01", "2024-03-25", 102.0, 1.0),
                    ("C", None, "2024-04-01", "2024-04-30", 103.0, 1.0)])
    retornos = retornos_mensais(cotas)
    assert pd.isna(_r(retornos, "C", "2024-02-01")) and pd.isna(_r(retornos, "C", "2024-03-01"))
    assert _r(retornos, "C", "2024-04-01") == pytest.approx(103 / 102 - 1)       # 25/03 is within 7 days of 31/03


def test_a_return_below_minus_60_or_above_150_percent_is_a_data_error():
    cotas = _cotas([("D", None, "2024-01-01", "2024-01-31", 100.0, 1.0),
                    ("D", None, "2024-02-01", "2024-02-29", 39.0, 1.0),
                    ("D", None, "2024-03-01", "2024-03-31", 98.0, 1.0)])
    retornos = retornos_mensais(cotas)
    assert pd.isna(_r(retornos, "D", "2024-02-01")) and pd.isna(_r(retornos, "D", "2024-03-01"))


def _doze(cnpj: str, inicio: str, r: float, meses: int = 14) -> pd.DataFrame:
    return pd.DataFrame({"cnpj": cnpj, "month": pd.date_range(inicio, periods=meses, freq="MS"), "r": r, "nav": 1.0})


def test_the_trailing_return_compounds_m_minus_13_to_m_minus_2_and_needs_all_12_months():
    retornos = pd.concat([_doze("A", "2023-01-01", 0.01), _doze("B", "2023-01-01", 0.01)])
    retornos.loc[(retornos["cnpj"] == "A") & (retornos["month"] == T("2024-02-01")), "r"] = 0.50   # m-1: skipped
    retornos = retornos[~((retornos["cnpj"] == "B") & (retornos["month"] == T("2023-05-01")))]
    acumulado = retorno_12m(retornos, T("2024-03-01"))       # window 2023-02 .. 2024-01
    assert acumulado["A"] == pytest.approx(1.01 ** 12 - 1)
    assert "B" not in acumulado.index


def test_top_funds_rank_by_trailing_return_then_larger_nav_then_cnpj():
    acumulado = pd.Series({"A": 0.3, "B": 0.5, "C": 0.3, "D": 0.3, "E": 0.1})
    nav = pd.Series({"A": 10.0, "B": 1.0, "C": 50.0, "D": 10.0, "E": 99.0})
    assert top_fundos(acumulado, nav, 4) == ["B", "C", "A", "D"]


def test_the_score_is_the_mean_weight_over_the_top_funds_with_zero_where_not_held():
    nota = nota_media([{"PETR": 0.6, "VALE": 0.4}, {"PETR": 0.2, "ITUB": 0.8}], ["PETR", "VALE", "ITUB", "BBDC"])
    assert nota.to_dict() == pytest.approx({"PETR": 0.4, "VALE": 0.2, "ITUB": 0.4, "BBDC": 0.0})
