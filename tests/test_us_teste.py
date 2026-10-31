"""The monthly grid of the US test (``core.us_teste``): the index, the engine's adapter, the measures, the comparisons."""

import numpy as np
import pandas as pd
import pytest

from core import us_teste as ut

MESES = pd.date_range("2020-01-01", periods=6, freq="MS")


def test_the_index_starts_at_one_the_month_before_the_first_return():
    retornos = pd.DataFrame({"A": [np.nan, 0.10, 0.10, np.nan, np.nan, np.nan]}, index=MESES)
    indice = ut.indice_total(retornos)
    assert indice.index[0] == pd.Timestamp("2020-01-31")     # month ends: the engine reads the month from the date
    assert np.isnan(indice["A"].iloc[3])                     # after its last price: empty, the engine holds the close
    assert indice["A"].iloc[:3].tolist() == pytest.approx([1.0, 1.10, 1.21])


def test_a_gap_is_not_a_return():
    retornos = pd.DataFrame({"A": [0.10, 0.10, np.nan, 0.10, 0.10, 0.10]}, index=MESES)
    indice = ut.indice_total(retornos)
    assert np.isnan(indice["A"].iloc[2])
    assert indice["A"].iloc[3] == pytest.approx(1.1 ** 3)


def test_a_terminal_loss_falls_in_the_names_last_month():
    retornos = pd.DataFrame({"A": [np.nan, 0.10, 0.10, np.nan, np.nan, np.nan], "B": [0.0] * 6}, index=MESES)
    indice = ut.indice_total(retornos, pd.Series({"A": -0.30}))
    assert indice["A"].iloc[2] == pytest.approx(1.1 * 1.1 * 0.7)
    assert np.isnan(indice["A"].iloc[3])
    assert indice["B"].iloc[-1] == pytest.approx(1.0)


def test_the_adapter_dates_the_training_in_the_decision_month_and_scales_to_a_daily_year():
    vistos = {}

    def estrategia(treino):
        vistos["mes"] = treino.index[-1].to_period("M").to_timestamp()
        vistos["vol_anual"] = treino["A"].std() * np.sqrt(252)
        return pd.Series({"A": 1.0})
    retornos = np.random.default_rng(1).normal(0.0, 0.05, 36)
    treino = pd.DataFrame({"A": retornos}, index=pd.date_range("2017-01-31", periods=36, freq="ME"))
    assert ut.no_grid_mensal(estrategia)(treino).to_dict() == {"A": 1.0}
    assert vistos["mes"] == pd.Timestamp("2020-01-01")       # the last row is December 2019: decided in January
    assert vistos["vol_anual"] == pytest.approx(retornos.std(ddof=1) * np.sqrt(12))


def test_the_config_goes_where_the_engine_reads_it():
    assert (ut.CONFIG.janela_treino, ut.CONFIG.janela_teste, ut.CONFIG.custo_bps) == (36, 3, 10.0)


def curva(*retornos) -> pd.Series:
    datas = pd.date_range("2019-12-31", periods=len(retornos) + 1, freq="ME")
    return pd.Series(np.cumprod([100.0, *[1 + r for r in retornos]]), index=datas)


def test_the_measures_annualize_by_twelve():
    valores = curva(*([0.01] * 24))
    medidas = ut.medidas(valores, pd.Series(0.0, index=valores.index[1:]))
    assert medidas["CAGR"] == pytest.approx(1.01 ** 12 - 1)
    assert medidas["Vol"] == pytest.approx(0.0, abs=1e-12)
    assert medidas["Queda máx."] == 0.0


def test_the_sharpe_is_over_the_risk_free_rate():
    retornos = np.random.default_rng(2).normal(0.01, 0.04, 60)
    valores = curva(*retornos)
    rf = pd.Series(0.002, index=valores.index[1:])
    esperado = (retornos - 0.002).mean() / (retornos - 0.002).std(ddof=1) * np.sqrt(12)
    assert ut.medidas(valores, rf)["Sharpe"] == pytest.approx(esperado)


def test_the_factor_alpha_recovers_a_known_alpha():
    rng = np.random.default_rng(3)
    datas = pd.date_range("2015-01-31", periods=120, freq="ME")
    fatores = pd.DataFrame(rng.normal(0.0, 0.03, (120, 6)), index=datas, columns=list(ut.FATORES))
    excesso = 0.005 + 0.9 * fatores["mkt_rf"] + 0.3 * fatores["smb"] + rng.normal(0.0, 0.001, 120)
    alfa, t, betas = ut.alfa_fatorial(excesso, fatores)
    assert alfa == pytest.approx(0.005 * 12, abs=0.002)
    assert t > 10
    assert betas["mkt_rf"] == pytest.approx(0.9, abs=0.02)


def test_the_gate_needs_every_phase_above_both_limits():
    assert ut.passa_o_portao([0.96, 0.97, 0.95], [0.95, 0.99, 0.95])
    assert not ut.passa_o_portao([0.96, 0.94, 0.95], [0.95, 0.99, 0.95])
    assert not ut.passa_o_portao([0.96, 0.97, 0.95], [0.95, 0.94, 0.95])


def test_a_portfolio_in_reais_follows_the_dollar():
    valores = curva(0.10, 0.0)
    ptax = pd.Series([5.0, 5.0, 5.5], index=valores.index)
    assert ut.em_reais(valores, ptax).tolist() == pytest.approx([100.0, 110.0, 121.0])


def test_the_best_of_the_last_12_months_never_reads_the_month_it_buys():
    meses = pd.date_range("2020-01-01", periods=16, freq="MS")
    retornos = pd.DataFrame({"BOM": [0.05] * 12 + [-0.50] * 4, "MAU": [-0.01] * 12 + [0.20] * 4}, index=meses)
    pesos = ut.melhores_em_12_meses(retornos, meses[12], quantos=1)
    assert pesos.to_dict() == {"BOM": 1.0}


def test_a_name_without_12_full_months_cannot_be_chosen():
    meses = pd.date_range("2020-01-01", periods=13, freq="MS")
    retornos = pd.DataFrame({"NOVO": [np.nan] * 2 + [0.5] * 11, "VELHO": [0.01] * 13}, index=meses)
    assert ut.melhores_em_12_meses(retornos, meses[12], quantos=1).to_dict() == {"VELHO": 1.0}


def test_a_quarterly_rotation_buys_at_each_rebalance_and_holds():
    meses = pd.date_range("2020-01-01", periods=18, freq="MS")
    retornos = pd.DataFrame({"A": [0.02] * 18}, index=meses)
    curva_rot = ut.rotacao_trimestral(retornos, meses[12], quantos=1, custo_bps=0.0)
    assert curva_rot.index[0] == pd.Timestamp("2020-12-31")
    assert curva_rot.iloc[-1] / curva_rot.iloc[0] == pytest.approx(1.02 ** 6)


def test_a_rotation_pays_its_costs_on_what_changes():
    meses = pd.date_range("2020-01-01", periods=16, freq="MS")
    retornos = pd.DataFrame({"A": [0.01] * 16}, index=meses)
    com_custo = ut.rotacao_trimestral(retornos, meses[12], quantos=1, custo_bps=100.0)
    sem_custo = ut.rotacao_trimestral(retornos, meses[12], quantos=1, custo_bps=0.0)
    assert com_custo.iloc[-1] / sem_custo.iloc[-1] == pytest.approx(0.99)    # bought once, never turned over


def test_the_best_over_the_period_is_hindsight_and_equal_weighted():
    meses = pd.date_range("2020-01-01", periods=4, freq="MS")
    retornos = pd.DataFrame({"A": [0.1] * 4, "B": [0.2] * 4, "C": [-0.1] * 4}, index=meses)
    valores = ut.melhores_em_retrospecto(retornos, meses[0], quantos=2)
    assert valores.iloc[-1] / valores.iloc[0] == pytest.approx((1.1 ** 4 + 1.2 ** 4) / 2)


def test_a_reference_starts_on_the_arms_first_date_with_its_own_returns():
    datas = pd.date_range("2020-01-31", periods=3, freq="ME")
    mensais = pd.Series([0.50, 0.10, 0.10], index=pd.date_range("2020-01-01", periods=3, freq="MS"))
    assert ut.referencia_nas_datas(mensais, datas).tolist() == pytest.approx([1.0, 1.1, 1.21])    # January was before it


def test_a_fund_counts_only_after_its_book_qualified_and_became_public():
    meses = pd.date_range("2020-01-01", periods=6, freq="MS")
    retornos = pd.DataFrame({"F": [0.01] * 6, "G": [0.02] * 6}, index=meses)
    fora = ut.so_depois_de_qualificar(retornos, pd.Series({"F": pd.Timestamp("2020-02-01")}), defasagem=2)
    assert fora["F"].notna().tolist() == [False, False, False, False, True, True]    # public from April; returns after it
    assert "G" not in fora.columns


def _pasta_de_resultados(pasta):
    datas = pd.date_range("2020-01-31", periods=3, freq="ME")
    pd.DataFrame({"date": datas, "Composto": [1.0, 1.1, 1.2], "SPY": [1.0, 1.05, 1.1], "B3 Composto": [1.0, 0.9, 1.0],
                  "Trend EUA 2x": [np.nan, 1.0, 1.3]}).to_parquet(pasta / "us-curvas.parquet", index=False)
    pd.DataFrame({"nome": ["Composto", "SPY", "B3 Composto", "Trend EUA 2x"], "CAGR": [0.3, 0.2, 0.1, 0.5], "Vol": [0.2] * 4,
                  "Queda máx.": [-0.2] * 4, "R$ 1 vira": [1.2, 1.1, 1.0, 1.3],
                  "Desde": [datas[0]] * 3 + [datas[1]]}).to_parquet(pasta / "us-resumo.parquet", index=False)
    pd.DataFrame({"nome": ["Composto"], "P[> SPY] mínimo": [0.8], "DSR mínimo": [0.7], "Passou": [False]}).to_parquet(
        pasta / "us-portao.parquet", index=False)
    pd.DataFrame({"carteira": ["Composto"], "mes": [pd.Timestamp("2026-10-01")], "ticker": ["AAPL"], "empresa": ["Apple"],
                  "setor": ["BusEq"], "peso": [1.0]}).to_parquet(pasta / "us-carteiras.parquet", index=False)
    (pasta / "us-cobertura.json").write_text(
        '{"companies": 3000, "dead_from_tiingo": 480, "mortos_candidatos": 1505, "first_month": "2016-11-01", '
        '"last_month": "2026-09-01", "n_tentativas": 53}', encoding="utf-8")


def test_the_loader_reads_the_derived_files_and_says_when_they_are_missing(tmp_path):
    from ui.servicos import carregar_eua
    assert carregar_eua(tmp_path) is None
    _pasta_de_resultados(tmp_path)
    resultado = carregar_eua(tmp_path)
    assert resultado.cobertura["dead_from_tiingo"] == 480
    assert list(resultado.curvas.columns)[:2] == ["Composto", "SPY"]
    assert resultado.curvas.index[0] == pd.Timestamp("2020-01-31")


def _tela_eua(pasta: str):
    from pathlib import Path

    from ui import eua
    from ui.servicos import carregar_eua
    eua.secao(carregar_eua(Path(pasta)))


def test_the_screen_shows_the_table_the_gate_the_picks_and_the_limit_of_the_data(tmp_path):
    from streamlit.testing.v1 import AppTest
    _pasta_de_resultados(tmp_path)
    tela = AppTest.from_function(_tela_eua, args=(str(tmp_path),), default_timeout=60).run()
    assert not tela.exception
    aviso = " ".join(item.value for item in tela.warning)
    assert "480" in aviso and "1.505" in aviso
    assert len(tela.dataframe) == 3
    assert tela.dataframe[0].value["CAGR"].iloc[0] == 0.5        # the best first
    assert "Apple" in tela.dataframe[2].value["empresa"].tolist()


def test_the_screen_says_when_the_test_has_not_run(tmp_path):
    from streamlit.testing.v1 import AppTest
    tela = AppTest.from_function(_tela_eua, args=(str(tmp_path),), default_timeout=60).run()
    assert not tela.exception
    assert tela.info and not tela.dataframe
