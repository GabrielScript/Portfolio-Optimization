"""The five-grade scenario Monte Carlo bench (docs/superpowers/specs/2026-10-28-bancada-cenarios-design.md)."""

import numpy as np
import pandas as pd
import pytest


def test_the_components_are_the_12_month_trend_the_63_session_vol_and_the_6_month_change_of_the_annual_cdi():
    from core.bancada import componentes
    sessoes = pd.bdate_range("2020-01-01", "2021-12-31")
    retornos = np.where(np.arange(len(sessoes)) % 2 == 0, 0.01, -0.005)            # alternating, known std
    bova = pd.Series(np.cumprod(1.0 + retornos), index=sessoes)
    cdi = pd.Series(np.where(sessoes < pd.Timestamp("2021-07-01"), 0.04, 0.05), index=sessoes)   # % per day
    fim = pd.Timestamp("2021-12-31")
    linha = componentes(bova, cdi, pd.DatetimeIndex([fim])).loc[fim]
    ano_antes = bova.loc[:pd.Timestamp("2020-12-31")].iloc[-1]
    assert linha["trend"] == pytest.approx(bova.iloc[-1] / ano_antes - 1.0)
    assert linha["vol"] == pytest.approx(np.log1p(pd.Series(retornos[-63:])).std() * np.sqrt(252))
    assert linha["dcdi"] == pytest.approx(1.0005 ** 252 - 1.0004 ** 252)


def test_the_score_adds_trend_and_subtracts_vol_and_cdi_change_and_the_grades_are_its_quintiles():
    from core.bancada import graus
    meses = pd.date_range("2020-01-31", periods=10, freq="ME")
    passo = np.arange(10.0)
    favoravel = pd.DataFrame({"trend": passo, "vol": -passo, "dcdi": -passo}, index=meses)
    assert graus(favoravel).tolist() == [1, 1, 2, 2, 3, 3, 4, 4, 5, 5]
    juros_subindo = pd.DataFrame({"trend": passo, "vol": 3.0 * passo, "dcdi": passo}, index=meses)   # z(trend) = z(vol)
    assert graus(juros_subindo).tolist() == [5, 5, 4, 4, 3, 3, 2, 2, 1, 1]


def _valores_aleatorios() -> pd.DataFrame:
    gerador = np.random.default_rng(7)
    sessoes = pd.bdate_range("2020-01-01", "2020-05-29")
    retornos = gerador.normal(0.0, 0.03, (len(sessoes), 2))
    return pd.DataFrame(np.cumprod(1.0 + retornos, axis=0), index=sessoes, columns=["A", "B"])


def test_a_path_of_resampled_months_has_the_terminal_return_and_max_drawdown_of_its_stitched_daily_returns():
    from core.bancada import caminhos, resumir_meses
    valores = _valores_aleatorios()
    resumo = resumir_meses(valores)                      # January has no prior value: months Feb..May
    assert list(resumo.meses) == list(pd.date_range("2020-02-01", periods=4, freq="MS"))
    sorteio = np.array([[2, 0, 3, 0], [1, 1, 1, 1]])
    terminal, mdd = caminhos(resumo, sorteio)
    log_diario = np.log(valores).diff()
    for linha, escolhidos in enumerate(sorteio):
        pedacos = [log_diario[log_diario.index.to_period("M").to_timestamp() == resumo.meses[i]] for i in escolhidos]
        cumulado = np.vstack([np.zeros((1, 2)), np.cumsum(pd.concat(pedacos).to_numpy(), axis=0)])
        pior = (np.maximum.accumulate(cumulado, axis=0) - cumulado).max(axis=0)
        assert terminal[linha] == pytest.approx(np.exp(cumulado[-1]) - 1.0)
        assert mdd[linha] == pytest.approx(1.0 - np.exp(-pior))


def test_transitions_count_grade_to_next_grade_and_paths_draw_each_month_from_the_pool_of_its_grade():
    from core.bancada import sortear, transicoes
    graus_inicio = np.array([1, 2, 1, 2, 1, 3])               # month i started in this grade
    contagem, matriz = transicoes(graus_inicio)
    assert contagem[0, 1] == 2 and contagem[1, 0] == 2 and contagem[0, 2] == 1 and contagem.sum() == 5
    assert matriz[1].tolist() == pytest.approx([1.0, 0.0, 0.0, 0.0, 0.0])
    sorteio = sortear(graus_inicio, matriz, 2, 200, 4, np.random.default_rng(0))
    assert sorteio.shape == (200, 4)
    assert set(sorteio[:, 0]) == {1, 3}                       # grade-2 months
    assert set(sorteio[:, 1]) <= {0, 2, 4}                    # from 2 always to 1
    proximos = graus_inicio[sorteio[:, 2]]                   # from 1: to 2 (2 of 3) or 3 (1 of 3)
    assert set(proximos) == {2, 3}


def test_todays_grade_uses_the_windows_z_and_quintile_cut_points():
    from core.bancada import grau_de
    meses = pd.date_range("2020-01-31", periods=10, freq="ME")
    passo = np.arange(10.0)
    janela = pd.DataFrame({"trend": passo, "vol": -passo, "dcdi": -passo}, index=meses)
    hoje = pd.DataFrame({"trend": [9.5, -1.0, 4.6], "vol": [-9.5, 1.0, -4.6], "dcdi": [-9.5, 1.0, -4.6]},
                        index=pd.to_datetime(["2021-01-15", "2021-01-16", "2021-01-17"]))
    assert grau_de(hoje, janela).tolist() == [5, 1, 3]


def test_the_window_graded_by_its_own_cut_points_gives_back_its_quintile_grades():
    from core.bancada import grau_de, graus
    gerador = np.random.default_rng(3)
    janela = pd.DataFrame(gerador.normal(size=(40, 3)), columns=["trend", "vol", "dcdi"],
                          index=pd.date_range("2015-01-31", periods=40, freq="ME"))
    assert grau_de(janela, janela).tolist() == graus(janela).tolist()


def test_the_statistics_are_pairwise_win_shares_terminal_quantiles_and_drawdown_median_and_worst_5pct():
    from core.bancada import estatisticas
    terminal = np.array([[0.10, 0.05], [0.20, 0.30], [0.00, -0.10], [0.40, 0.10]])
    mdd = np.array([[0.1, 0.2], [0.2, 0.4], [0.3, 0.1], [0.4, 0.3]])
    linha = estatisticas(terminal, mdd, ("A", "B"), [("A", "B")])
    assert linha["P[A > B]"] == pytest.approx(0.75)
    assert linha["A ret p50"] == pytest.approx(np.quantile(terminal[:, 0], 0.5))
    assert linha["B ret p5"] == pytest.approx(np.quantile(terminal[:, 1], 0.05))
    assert linha["A MDD p50"] == pytest.approx(0.25)
    assert linha["B MDD worst 5%"] == pytest.approx(np.quantile(mdd[:, 1], 0.95))
