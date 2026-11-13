"""The app table's columns computed at a past month with only what was known then (``core.indicadores_no_tempo``,
spec ``docs/superpowers/specs/2026-11-13-indicadores-risco-design.md``)."""

import numpy as np
import pandas as pd
import pytest

MESES = pd.date_range("2015-01-01", periods=80, freq="MS")
MES = MESES[-1]                                   # the signal of 2021-08 sees months up to 2021-07


def test_the_return_of_n_months_is_the_product_of_the_complete_months_before_the_month():
    from core.indicadores_no_tempo import retorno_de_meses
    retornos = pd.DataFrame({"A": 0.01, "B": 0.01, "C": 0.01}, index=MESES)
    retornos.loc[MES, "A"] = 9.0                                  # the signal's own month stays out
    retornos.loc[MES - pd.DateOffset(months=37), "A"] = 9.0       # and so does the 37th month before
    retornos.loc[MES - pd.DateOffset(months=10), "B"] = np.nan    # a missing month is not a zero return
    de_3_anos = retorno_de_meses(retornos, MES, 36)
    assert de_3_anos["A"] == pytest.approx(1.01 ** 36 - 1.0)
    assert np.isnan(de_3_anos["B"])
    assert retorno_de_meses(retornos, MES, 72)["C"] == pytest.approx(1.01 ** 72 - 1.0)


def _precos_diarios(sessoes: int = 900, fim: str = "2021-09-30") -> tuple[pd.DataFrame, pd.Series]:
    """Daily closes ending after ``MES`` (2021-08-01): the market, a name with twice its moves and a quiet name; and
    a flat daily rate."""
    datas = pd.bdate_range(end=fim, periods=sessoes)
    gerador = np.random.default_rng(7)
    mercado = gerador.normal(0.0004, 0.01, sessoes)
    retornos = pd.DataFrame({"BOVA11": mercado, "ALTO3": 2.0 * mercado,
                             "CALM3": 0.5 * mercado + gerador.normal(0.0003, 0.004, sessoes)}, index=datas)
    return 10.0 * (1.0 + retornos).cumprod(), pd.Series(0.0003, index=datas)


def test_the_daily_measures_use_only_the_sessions_before_the_month_and_the_tables_formulas():
    from core.indicadores_no_tempo import medidas_diarias
    from core.metrics import cvar_historico, sharpe
    precos, juro = _precos_diarios()
    medidas = medidas_diarias(precos, juro, "BOVA11", MES, ["ALTO3", "CALM3", "FORA3"])
    depois = precos.copy()
    depois.loc[depois.index >= MES, "ALTO3"] *= 3.0                # what happens from the month on is unknown
    pd.testing.assert_frame_equal(medidas, medidas_diarias(depois, juro, "BOVA11", MES, ["ALTO3", "CALM3", "FORA3"]))
    assert list(medidas.index) == ["ALTO3", "CALM3"]              # a candidate without prices has no row
    janela = precos.loc[precos.index < MES, "CALM3"].iloc[-756:]
    diarios = janela.pct_change().dropna()
    assert medidas.loc["CALM3", "SHARPE"] == pytest.approx(sharpe(diarios - 0.0003))
    assert medidas.loc["CALM3", "CVAR"] == pytest.approx(-cvar_historico(diarios))     # a smaller loss ranks higher
    assert medidas.loc["ALTO3", "BETA"] == pytest.approx(-2.0)                         # a lower beta ranks higher
    assert medidas.loc["CALM3", "BETA"] > medidas.loc["ALTO3", "BETA"]
    assert set(medidas.columns) == {"SHARPE", "SORTINO", "CVAR", "ALPHA", "T-ALPHA", "BETA"}


def test_a_name_without_closes_on_nine_tenths_of_the_window_has_no_measure_for_it():
    from core.indicadores_no_tempo import medidas_diarias
    precos, juro = _precos_diarios()
    antes = precos.index[precos.index < MES]
    precos.loc[antes[-756:-400], "CALM3"] = np.nan        # listed 400 sessions ago: one full year, not three
    medidas = medidas_diarias(precos, juro, "BOVA11", MES, ["ALTO3", "CALM3"])
    assert medidas.loc["CALM3", ["SHARPE", "SORTINO", "CVAR", "ALPHA", "T-ALPHA"]].isna().all()
    assert np.isfinite(medidas.loc["CALM3", "BETA"])
    assert medidas.loc["ALTO3"].notna().all()


def _linha(issuer: str, account: str, value: float, dt_refer: str, available_from: str) -> dict:
    return {"issuer": issuer, "account": account, "value": value, "dt_refer": pd.Timestamp(dt_refer),
            "available_from": pd.Timestamp(available_from)}


def test_the_roe_of_a_month_comes_from_the_last_statement_received_before_it():
    from core.indicadores_no_tempo import roe_no_mes
    fundamentos = pd.DataFrame([
        _linha("PETR", "net_income_parent", 10.0, "2021-03-31", "2021-05-15"),
        _linha("PETR", "equity", 50.0, "2021-03-31", "2021-05-15"),
        _linha("PETR", "net_income_parent", 40.0, "2021-06-30", "2021-08-10"),     # received after the month starts
        _linha("PETR", "equity", 50.0, "2021-06-30", "2021-08-10"),
        _linha("MGLU", "net_income_parent", 5.0, "2021-03-31", "2021-05-15"),
        _linha("MGLU", "equity", -20.0, "2021-03-31", "2021-05-15"),                # negative equity: no ROE
    ])
    roe = roe_no_mes(fundamentos, MES, ["PETR3", "PETR4", "MGLU3", "VALE3"])
    assert roe.to_dict() == pytest.approx({"PETR3": 0.2, "PETR4": 0.2})            # an issuer's classes share it


def _fontes():
    """Six seasoned names and NOVO3, listed 400 sessions and 40 months ago."""
    from core.indicadores_no_tempo import Fontes
    datas = pd.bdate_range(end="2021-09-30", periods=900)
    gerador = np.random.default_rng(11)
    mercado = gerador.normal(0.0004, 0.01, len(datas))
    nomes = ["AAAA3", "BBBB3", "CCCC3", "DDDD3", "EEEE3", "FFFF3", "NOVO3"]
    diarios = pd.DataFrame({nome: beta * mercado + gerador.normal(0.0002, 0.006, len(datas))
                            for nome, beta in zip(nomes, (0.4, 0.7, 1.0, 1.2, 1.5, 1.9, 1.0))}, index=datas)
    precos = 10.0 * (1.0 + diarios.assign(BOVA11=mercado)).cumprod()
    antes = precos.index[precos.index < MES]
    precos.loc[precos.index < antes[-400], "NOVO3"] = np.nan
    mensais = pd.DataFrame(gerador.normal(0.01, 0.06, (len(MESES), len(nomes))), index=MESES, columns=nomes)
    mensais.loc[mensais.index < MES - pd.DateOffset(months=40), "NOVO3"] = np.nan
    fundamentos = pd.DataFrame([_linha(nome[:4], conta, valor, "2021-03-31", "2021-05-15")
                                for posicao, nome in enumerate(nomes)
                                for conta, valor in (("net_income_parent", 5.0 + posicao), ("equity", 50.0))])
    return Fontes(precos, pd.Series(0.0003, index=datas), mensais, fundamentos), nomes


def test_the_table_of_a_month_has_the_nine_signals_and_their_mean_normal_score():
    from core.indicadores_no_tempo import CONJUNTO, SINAIS, tabela_no_mes
    from core.views import escore_normal
    fontes, nomes = _fontes()
    tabela = tabela_no_mes(fontes, MES, nomes)
    assert list(tabela.columns) == [*SINAIS, CONJUNTO] and len(SINAIS) == 9
    assert tabela.loc[nomes[:6], list(SINAIS)].notna().all().all()
    esperado = np.mean([escore_normal(tabela[sinal])["AAAA3"] for sinal in SINAIS])
    assert tabela.loc["AAAA3", CONJUNTO] == pytest.approx(esperado)
    # NOVO3 has three of the nine (beta, three-year return, ROE): they stay, the mean needs at least five
    assert tabela.loc["NOVO3", list(SINAIS)].notna().sum() == 3
    assert np.isnan(tabela.loc["NOVO3", CONJUNTO])
    assert tabela.loc["FFFF3", "ROE"] > tabela.loc["AAAA3", "ROE"]


def test_the_residual_is_what_the_components_normal_scores_do_not_explain():
    from core.indicadores_no_tempo import residuo
    from core.views import escore_normal
    gerador = np.random.default_rng(5)
    nomes = [f"N{posicao:03d}" for posicao in range(120)]
    componentes = pd.DataFrame(gerador.normal(size=(120, 3)), index=nomes, columns=["cons", "valor", "lucro"])
    componentes.loc["N007", "valor"] = np.nan                     # a missing component counts as the mean
    sinal = pd.Series(gerador.normal(size=120), index=nomes) + componentes["cons"]
    sinal["N003"] = np.nan                                        # a name without the signal has no residual
    sobra = residuo(sinal, componentes)
    assert "N003" not in sobra.index and "N007" in sobra.index and len(sobra) == 119
    escores = componentes.loc[sobra.index].apply(escore_normal).fillna(0.0)
    assert np.abs(escores.T @ sobra).max() < 1e-9 and abs(sobra.sum()) < 1e-9
    # a signal that is a monotone function of one component carries nothing beyond it
    assert np.abs(residuo(np.exp(componentes["cons"]), componentes[["cons"]])).max() < 1e-9
    assert residuo(sinal.iloc[:4], componentes.iloc[:4]).empty    # too few names to fit three components
