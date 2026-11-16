"""The broad panel's characteristics (spec 2026-11-16-painel-amplo, section 3)."""

import numpy as np
import pandas as pd
import pytest

from core.painel_amplo import NOVAS, Dados, angulos_do_mes, caracteristicas
from core.valor import Balancos

MES = pd.Timestamp("2020-06-01")
MESES = pd.date_range("2016-01-01", "2020-08-01", freq="MS")
PAPEIS = ["AAAA3", "BBBB3", "CCCC3"]


def _fundamento(issuer, conta, dt_refer, valor, recebido):
    return {"code_cvm": 1, "account": conta, "dt_refer": pd.Timestamp(dt_refer), "value": valor,
            "available_from": pd.Timestamp(recebido), "issuer": issuer, "sector": "Industria"}


def _dados(fundamentos_extra=(), fechamentos=None, retornos=None) -> Dados:
    linhas = []
    for issuer, ativos in (("AAAA", 100.0), ("BBBB", 200.0), ("CCCC", 300.0)):
        linhas += [_fundamento(issuer, "total_assets", "2019-03-31", ativos * 0.5, "2019-05-10"),
                   _fundamento(issuer, "total_assets", "2020-03-31", ativos, "2020-05-10"),
                   _fundamento(issuer, "net_income", "2020-03-31", 10.0, "2020-05-10"),
                   _fundamento(issuer, "equity", "2020-03-31", 50.0, "2020-05-10"),
                   _fundamento(issuer, "revenue", "2020-03-31", 80.0, "2020-05-10")]
    linhas += list(fundamentos_extra)
    caps = pd.DataFrame([{"issuer": i, "month": m, "market_cap": 1_000.0, "shares_today_units": 100.0}
                         for i in ("AAAA", "BBBB", "CCCC") for m in MESES])
    gerador = np.random.default_rng(0)
    if retornos is None:
        retornos = pd.DataFrame(gerador.normal(0.01, 0.05, (len(MESES), 3)), index=MESES, columns=PAPEIS)
    if fechamentos is None:
        dias = pd.bdate_range("2020-01-01", "2020-07-31")
        fechamentos = pd.DataFrame(100 * np.cumprod(1 + gerador.normal(0, 0.01, (len(dias), 3)), axis=0),
                                   index=dias, columns=PAPEIS)
    return Dados(retornos, fechamentos, Balancos(pd.DataFrame(linhas), caps, {i: "Industria" for i in ("AAAA", "BBBB", "CCCC")}))


def test_a_document_received_in_the_decision_month_is_not_read():
    tardio = [_fundamento("AAAA", "net_income", "2020-03-31", 999.0, "2020-06-15")]   # arrives inside June
    base = caracteristicas(_dados(), MES, PAPEIS)
    com_tardio = caracteristicas(_dados(tardio), MES, PAPEIS)
    pd.testing.assert_frame_equal(base, com_tardio)
    assert base.loc["AAAA3", "roa"] == pytest.approx(10.0 / 100.0)


def test_asset_growth_compares_with_the_snapshot_a_year_before():
    tabela = caracteristicas(_dados(), MES, PAPEIS)
    assert tabela.loc["AAAA3", "crescimento_ativos"] == pytest.approx(-(100.0 / 50.0 - 1.0))


def test_daily_features_ignore_sessions_from_the_decision_month_on():
    dados = _dados()
    mexido = dados.fechamentos.copy()
    mexido.loc[mexido.index >= MES] *= 5.0
    outra = Dados(dados.retornos, mexido, dados.balancos)
    pd.testing.assert_frame_equal(caracteristicas(dados, MES, PAPEIS)[["max_diario", "vol_diaria"]],
                                  caracteristicas(outra, MES, PAPEIS)[["max_diario", "vol_diaria"]])


def test_momentum_6_1_skips_the_last_month_and_reversal_reads_it():
    retornos = pd.DataFrame(0.0, index=MESES, columns=PAPEIS)
    retornos.loc[MES - pd.DateOffset(months=1), "AAAA3"] = 0.50          # m-1
    retornos.loc[MES - pd.DateOffset(months=3), "BBBB3"] = 0.20          # inside m-6 .. m-2
    tabela = caracteristicas(_dados(retornos=retornos), MES, PAPEIS)
    assert tabela.loc["AAAA3", "momentum_6_1"] == pytest.approx(0.0)
    assert tabela.loc["AAAA3", "reversao_curta"] == pytest.approx(-0.50)
    assert tabela.loc["BBBB3", "momentum_6_1"] == pytest.approx(0.20)


def test_angles_put_missing_at_zero_in_the_spec_order():
    angulos = angulos_do_mes(_dados(), MES, [*PAPEIS, "SEMD3"])
    assert list(angulos.columns) == list(NOVAS)
    assert (angulos.loc["SEMD3"] == 0.0).all()
    assert angulos.abs().max().max() < np.pi / 2
