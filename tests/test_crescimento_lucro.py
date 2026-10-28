"""The profit-growth signal of the pre-registered IC test (docs/superpowers/specs/2026-10-27-crescimento-lucro-design.md)."""

import numpy as np
import pandas as pd
import pytest


def _linha(emissor: str, conta: str, fim: str, valor: float, recebido: str) -> dict:
    return {"issuer": emissor, "account": conta, "dt_refer": pd.Timestamp(fim), "value": valor,
            "available_from": pd.Timestamp(recebido)}


def _fundamentos() -> pd.DataFrame:
    return pd.DataFrame([
        _linha("AAAA", "net_income", "2020-09-30", 9.0, "2020-11-10"),
        _linha("AAAA", "total_assets", "2020-09-30", 90.0, "2020-11-10"),
        _linha("AAAA", "net_income", "2020-12-31", 10.0, "2021-03-20"),
        _linha("AAAA", "total_assets", "2020-12-31", 100.0, "2021-03-20"),
        _linha("AAAA", "net_income", "2021-09-30", 12.0, "2021-11-10"),
        _linha("AAAA", "net_income", "2021-12-31", 15.0, "2022-03-20"),
        _linha("BBBB", "net_income", "2021-12-31", 4.0, "2022-03-01"),          # no filing a year before
        _linha("CCCC", "net_income", "2020-12-31", 1.0, "2021-03-01"),
        _linha("CCCC", "total_assets", "2020-12-31", 0.0, "2021-03-01"),       # assets <= 0 a year before
        _linha("CCCC", "net_income", "2021-12-31", 2.0, "2022-03-01"),
    ])


def test_the_change_is_against_the_same_quarter_a_year_before_over_its_assets_with_filings_known_at_the_month():
    from core.crescimento_lucro import variacao_anual
    abril = variacao_anual(_fundamentos(), "net_income", pd.Timestamp("2022-04-01"))
    assert abril.to_dict() == {"AAAA": pytest.approx((15.0 - 10.0) / 100.0)}
    fevereiro = variacao_anual(_fundamentos(), "net_income", pd.Timestamp("2022-02-01"))   # 2021-12 not yet received
    assert fevereiro.to_dict() == {"AAAA": pytest.approx((12.0 - 9.0) / 90.0)}
    assert variacao_anual(_fundamentos(), "net_income", pd.Timestamp("2021-04-01")).empty  # 2019-12 never filed


def test_the_score_is_the_mean_normal_score_of_both_changes_with_the_ones_available_shared_by_an_issuers_classes():
    from core.crescimento_lucro import nota_de_crescimento
    from core.views import escore_normal
    linhas = []
    for emissor, (lucro, bruto) in {"AAAA": ((10, 30), (50, 40)), "BBBB": ((10, 12), (50, 90)),
                                    "CCCC": ((10, 5), (50, 55)), "DDDD": ((10, 20), None)}.items():
        linhas += [_linha(emissor, "total_assets", "2020-12-31", 100.0, "2021-03-01"),
                   _linha(emissor, "net_income", "2020-12-31", lucro[0], "2021-03-01"),
                   _linha(emissor, "net_income", "2021-12-31", lucro[1], "2022-03-01")]
        if bruto:
            linhas += [_linha(emissor, "gross_profit", "2020-12-31", bruto[0], "2021-03-01"),
                       _linha(emissor, "gross_profit", "2021-12-31", bruto[1], "2022-03-01")]
    candidatos = ["AAAA3", "AAAA4", "BBBB3", "CCCC3", "DDDD3", "EEEE3"]
    nota = nota_de_crescimento(pd.DataFrame(linhas))(pd.Timestamp("2022-04-01"), candidatos)
    lucro = escore_normal(pd.Series([0.2, 0.2, 0.02, -0.05, 0.1], index=candidatos[:5]))
    bruto = escore_normal(pd.Series([-0.1, -0.1, 0.4, 0.05], index=candidatos[:4]))
    esperado = pd.concat([lucro, bruto], axis=1).mean(axis=1)
    assert list(nota.index) == candidatos[:5] and nota["AAAA3"] == nota["AAAA4"]
    assert nota.to_numpy() == pytest.approx(esperado.reindex(candidatos[:5]).to_numpy())
    assert nota["DDDD3"] == pytest.approx(lucro["DDDD3"]) and not np.isnan(nota["DDDD3"])
