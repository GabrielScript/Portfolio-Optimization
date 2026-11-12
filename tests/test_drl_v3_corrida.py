"""DRL v3: the training universe, the cells of the new arms, the score input and the report's CAPM line."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from drl import executar, leitura, protocolo
from drl.ambiente import EnvConfig
from drl.dados import build_panel
from drl.relatorio import capm
from tests.drl_fixtures import panel_of, universe_of

CORRECTED = EnvConfig(hold=21, cost_bps=10.0, observation="cross", reward="dsr_cdi", invest_all=True, dsr_warmup=252)


def _prices(names: str = "ABCDE") -> pd.DataFrame:
    datas = pd.bdate_range("2010-01-04", "2018-12-28")
    rng = np.random.default_rng(3)
    return pd.DataFrame(100.0 * np.cumprod(1.0 + rng.normal(0.0004, 0.015, size=(len(datas), len(names))), axis=0),
                        index=datas, columns=list(names))


def _panel_with_a_name_that_left_and_one_that_returns():
    """D is listed only up to 2013 (it died inside the training years of 2016); E is listed in 2012-2013 and again
    from June 2016, so it becomes eligible in the middle of the test year."""
    precos = _prices()
    universo = universe_of(precos)
    mes = pd.to_datetime(universo["month"])
    fica = ((universo["security"].isin(list("ABC")))
            | ((universo["security"] == "D") & (mes < "2014-01-01"))
            | ((universo["security"] == "E") & (((mes >= "2012-01-01") & (mes < "2014-01-01")) | (mes >= "2016-06-01"))))
    return build_panel(precos, universo[fica], pd.Series(0.0004, index=precos.index), history=20)


# --- defect 7: survivors only in training -----------------------------------------------------------------------------


def test_the_training_universe_keeps_the_names_that_left_before_the_test() -> None:
    painel = _panel_with_a_name_that_left_and_one_that_returns()
    janela = protocolo.janelas(painel.sessions, 2016, 2016)[0]
    assert executar.window_names(painel, janela) == ["A", "B", "C"]
    assert executar.training_names(painel, janela) == ["A", "B", "C", "D", "E"]


def test_the_test_stretch_trades_only_the_names_known_when_it_starts() -> None:
    painel = _panel_with_a_name_that_left_and_one_that_returns()
    janela = protocolo.janelas(painel.sessions, 2016, 2016)[0]
    trechos = executar.stretches(painel, janela, CORRECTED, episode=12)
    meio_do_ano = int(painel.sessions.get_loc("2016-09-01"))
    assert painel.eligible_names(meio_do_ano) == ["A", "B", "C", "E"]           # E is back on the list by then
    teste = trechos.test
    teste.t = meio_do_ano
    assert teste.tradeable().tolist() == [True, True, True, False, False]       # but the test started without it
    treino = trechos.train()
    treino.t = int(painel.sessions.get_loc("2013-06-03"))
    assert treino.tradeable().tolist() == [True] * 5 and treino.episode == 12   # training saw D and E trade


# --- the cells of the new arms ----------------------------------------------------------------------------------------


def test_a_direct_cell_trains_every_seed_and_trades_the_mean_book() -> None:
    painel = _panel_with_a_name_that_left_and_one_that_returns()
    janela = protocolo.janelas(painel.sessions, 2016, 2016)[0]
    spec = executar.RunSpec(CORRECTED, seeds=(0, 1), direct_steps=20)
    cell = executar.run_cell(painel, janela, "direct/sharpe", spec)
    regra = executar.run_cell(painel, janela, "equal_weight", spec)
    assert cell.sessions == regra.sessions and cell.names == 3 and cell.chosen_seed is None
    assert set(cell.validation_rewards) == {"0", "1"} and len(cell.turnover) == len(regra.turnover)
    assert cell.values[-1] > 0 and cell.values != regra.values
    with pytest.raises(ValueError, match="sharpe"):
        executar.run_cell(painel, janela, "direct/dsr", spec)
    with pytest.raises(ValueError, match="scores"):
        executar.run_cell(painel, janela, "direct_score/sharpe", spec)          # this panel carries no score


def test_a_ppo_cell_runs_on_the_training_universe() -> None:
    painel = _panel_with_a_name_that_left_and_one_that_returns()
    janela = protocolo.janelas(painel.sessions, 2016, 2016)[0]
    spec = executar.RunSpec(CORRECTED, seeds=(0,), timesteps=48, n_envs=1)
    cell = executar.run_cell(painel, janela, "ppo_shared/dsr_cdi", spec)
    assert cell.names == 3 and cell.chosen_seed == 0 and len(cell.values) == len(cell.sessions)


# --- the score input --------------------------------------------------------------------------------------------------


def test_a_decision_reads_the_score_of_the_month_it_earns_in() -> None:
    precos = _prices("AB").loc["2015-01-01":"2015-04-30"]
    notas = pd.DataFrame({"month": pd.to_datetime(["2015-02-01", "2015-02-01", "2015-03-01"]),
                          "security": ["A", "B", "A"], "score": [1.0, 2.0, 3.0]})
    painel = build_panel(precos, universe_of(precos), pd.Series(0.0, index=precos.index), history=1, scores=notas)
    linha = {dia: int(painel.sessions.get_loc(dia)) for dia in ("2015-01-29", "2015-01-30", "2015-02-27", "2015-03-31")}
    assert np.isnan(painel.scores[linha["2015-01-29"]]).all()                   # earns in January: no score
    assert painel.scores[linha["2015-01-30"]].tolist() == [1.0, 2.0]            # the last January close earns in February
    assert painel.scores[linha["2015-02-27"], 0] == 3.0 and np.isnan(painel.scores[linha["2015-02-27"], 1])
    assert np.isnan(painel.scores[linha["2015-03-31"]]).all()
    assert panel_of(precos).scores is None


# --- defect 8: alpha, beta and the t of alpha -------------------------------------------------------------------------


def test_the_capm_line_recovers_a_planted_alpha_and_beta() -> None:
    rng = np.random.default_rng(0)
    sessoes = 2520
    taxa = np.full(sessoes, 0.0004)
    mercado = rng.normal(0.0006, 0.012, sessoes)
    carteira = taxa + 0.0004 + 0.7 * (mercado - taxa) + rng.normal(0.0, 0.004, sessoes)
    linha = capm(np.r_[1.0, np.cumprod(1.0 + carteira)], np.r_[1.0, np.cumprod(1.0 + mercado)], taxa)
    assert linha["beta"] == pytest.approx(0.7, abs=0.02)
    assert linha["alpha"] == pytest.approx(0.0004 * 252, abs=0.03)
    erro_mqo = 0.004 / np.sqrt(sessoes)                             # independent residuals: Newey-West is close to OLS
    assert linha["t_alpha"] == pytest.approx(linha["alpha"] / 252 / erro_mqo, rel=0.15)
    igual = capm(np.r_[1.0, np.cumprod(1.0 + mercado)], np.r_[1.0, np.cumprod(1.0 + mercado)], taxa)
    assert igual["beta"] == pytest.approx(1.0) and igual["alpha"] == pytest.approx(0.0, abs=1e-12)
    with pytest.raises(ValueError):
        capm(np.ones(10), np.ones(9), taxa[:9])


def test_the_table_adds_the_capm_line_against_the_benchmark() -> None:
    painel = panel_of(_prices("ABCD"), cdi=0.0004, history=20)
    janelas = protocolo.janelas(painel.sessions, 2016, 2018)
    spec = executar.RunSpec(EnvConfig(hold=21))
    celulas = {"equal_weight": [executar.run_cell(painel, j, "equal_weight", spec) for j in janelas]}
    curva, _ = leitura.stitch(celulas["equal_weight"])
    tabela = leitura.table(celulas, painel, benchmark=curva * 3.0)
    assert tabela.loc["equal_weight", "beta"] == pytest.approx(1.0)
    assert tabela.loc["equal_weight", "alpha"] == pytest.approx(0.0, abs=1e-9)
    assert "alpha" not in leitura.table(celulas, painel).columns
