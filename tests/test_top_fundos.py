"""Copy the top-return equity funds (docs/superpowers/specs/2026-10-29-top-fundos-copia-design.md)."""

import pandas as pd
import pytest

from core.top_fundos import (Carteiras, meses_cheios, nota_media, painel_top_fundos, retorno_12m, retornos_mensais,
                             top_fundos)

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


MES = T("2024-03-01")                  # signal month: the ranking window is 2023-02 .. 2024-01
CARTEIRA, CARTEIRA_ANTERIOR = T("2023-09-30"), T("2023-08-31")
UNIVERSO = pd.DataFrame({"month": MES, "security": ["PETR4", "VALE3", "ITUB4", "BBDC4"]})


def _carteiras(fundos: dict[str, tuple]) -> Carteiras:
    """cnpj -> (qualifies, issuer weights, portfolio month-end, first month it is public), as attempt 52's tables;
    benchmark weights at both portfolio months."""
    linhas = [(cnpj, t, qualifica, tuple(pesos), pesos, 0.6, len(pesos))
              for cnpj, (qualifica, pesos, t, _) in fundos.items()]
    colunas = ["cnpj", "month", "qualifica", "ideias", "pesos", "active_share", "nomes"]
    ideias = pd.DataFrame(linhas, columns=colunas).set_index(["cnpj", "month"])
    publico = pd.Series({(cnpj, t): publica for cnpj, (_, _, t, publica) in fundos.items()})
    publico.index.names = ["cnpj", "month"]
    bench = pd.DataFrame([(CARTEIRA, "PETR", 0.5), (CARTEIRA, "VALE", 0.3), (CARTEIRA, "ITUB", 0.2),
                          (CARTEIRA_ANTERIOR, "PETR", 0.4), (CARTEIRA_ANTERIOR, "VALE", 0.4),
                          (CARTEIRA_ANTERIOR, "ITUB", 0.2)], columns=["month", "issuer", "w_b"])
    return Carteiras(ideias=ideias, publico=publico, bench=bench, corte=T("2026-04-30"))


def _linhas(painel: pd.DataFrame, coluna: str) -> dict[str, float]:
    return painel[painel["month"] == MES].set_index("issuer")[coluna].to_dict()


PUBLICA = T("2023-11-01")
FUNDOS = {"A": (True, {"PETR": 0.6, "VALE": 0.4}, CARTEIRA, PUBLICA),
          "B": (True, {"PETR": 0.2, "ITUB": 0.8}, CARTEIRA_ANTERIOR, PUBLICA),
          "C": (True, {"VALE": 1.0}, CARTEIRA, PUBLICA),
          "D": (True, {"BBDC": 1.0}, CARTEIRA, PUBLICA),          # 11 months of returns: not ranked
          "E": (False, {"BBDC": 1.0}, CARTEIRA, PUBLICA),         # best return, but does not qualify
          "F": (True, {"BBDC": 1.0}, CARTEIRA, T("2024-04-01"))}  # portfolio not public by March 2024
RETORNOS = pd.concat([_doze("A", "2023-01-01", 0.03), _doze("B", "2023-01-01", 0.02), _doze("C", "2023-01-01", 0.01),
                      _doze("D", "2023-03-01", 0.04, meses=12), _doze("E", "2023-01-01", 0.05),
                      _doze("F", "2023-01-01", 0.04)])


def test_the_score_averages_the_top_n_ranked_qualifying_funds_and_the_baseline_averages_every_ranked_one():
    painel = painel_top_fundos(_carteiras(FUNDOS), RETORNOS, UNIVERSO, [MES], n=2)
    assert _linhas(painel, "score") == pytest.approx({"PETR": 0.4, "VALE": 0.2, "ITUB": 0.4, "BBDC": 0.0})
    assert _linhas(painel, "score_all") == pytest.approx({"PETR": 0.8 / 3, "VALE": 1.4 / 3, "ITUB": 0.8 / 3,
                                                          "BBDC": 0.0})
    assert _linhas(painel, "qualifying") == {"PETR": 4, "VALE": 4, "ITUB": 4, "BBDC": 4}
    assert _linhas(painel, "ranked") == {"PETR": 3, "VALE": 3, "ITUB": 3, "BBDC": 3}


def test_a_tie_in_trailing_return_goes_to_the_larger_nav_at_m_minus_2_not_at_m_minus_1_nor_to_the_cnpj():
    fundos = {"A2": (True, {"VALE": 1.0}, CARTEIRA, PUBLICA), "B2": (True, {"PETR": 1.0}, CARTEIRA, PUBLICA)}
    retornos = pd.concat([_doze("A2", "2023-01-01", 0.02), _doze("B2", "2023-01-01", 0.02)])
    navs = {("A2", "2024-01-01"): 50.0, ("A2", "2024-02-01"): 500.0,
            ("B2", "2024-01-01"): 100.0, ("B2", "2024-02-01"): 1.0}
    for (cnpj, mes), nav in navs.items():
        retornos.loc[(retornos["cnpj"] == cnpj) & (retornos["month"] == T(mes)), "nav"] = nav
    painel = painel_top_fundos(_carteiras(fundos), retornos, UNIVERSO, [MES], n=1)
    assert _linhas(painel, "score") == pytest.approx({"PETR": 1.0, "VALE": 0.0, "ITUB": 0.0, "BBDC": 0.0})


def test_the_reported_active_weight_takes_each_top_funds_benchmark_at_its_own_portfolio_month():
    painel = painel_top_fundos(_carteiras(FUNDOS), RETORNOS, UNIVERSO, [MES], n=2)
    # A at 2023-09 (benchmark PETR .5, VALE .3, ITUB .2), B at 2023-08 (PETR .4, VALE .4, ITUB .2)
    assert _linhas(painel, "active") == pytest.approx({"PETR": -0.05, "VALE": -0.15, "ITUB": 0.2, "BBDC": 0.0})


def test_coverage_counts_the_qualifying_funds_with_any_return_in_the_window_apart_from_the_ranked_ones():
    fundos = FUNDOS | {"G": (True, {"PETR": 1.0}, CARTEIRA, PUBLICA)}        # no quota at all (unmatched cnpj)
    painel = painel_top_fundos(_carteiras(fundos), RETORNOS, UNIVERSO, [MES], n=2)
    linha = painel[painel["month"] == MES].iloc[0]
    assert (linha["qualifying"], linha["with_quotas"], linha["ranked"]) == (5, 4, 3)     # D: 11 of 12 months


def _contagens(linhas: list[tuple]) -> pd.DataFrame:
    """(month, qualifying, with_quotas, ranked) per month, two issuer rows each, as the panel repeats them."""
    meses = pd.DataFrame(linhas, columns=["month", "qualifying", "with_quotas", "ranked"])
    painel = meses.loc[meses.index.repeat(2)].assign(issuer=["PETR", "VALE"] * len(meses), score=0.0)
    return painel.assign(month=pd.to_datetime(painel["month"])).reset_index(drop=True)


CONTAGENS = [("2015-01-01", 40, 36, 30), ("2015-02-01", 20, 10, 5), ("2016-01-01", 40, 20, 14),
             ("2016-02-01", 30, 30, 15)]


def test_a_month_is_read_only_with_at_least_30_qualifying_and_15_ranked_funds():
    assert meses_cheios(_contagens(CONTAGENS)) == [T("2015-01-01"), T("2016-02-01")]
