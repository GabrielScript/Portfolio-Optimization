"""The DRL comparators, the agents' wiring, the per-cell files and the reading."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from drl import agentes, baselines, executar, leitura, protocolo
from drl.ambiente import EnvConfig, PortfolioEnv, simulate
from drl.relatorio import metrics
from tests.drl_fixtures import panel_of, random_prices


def _env(sessions: int = 400, names: int = 4, hold: int = 21, seed: int = 0) -> PortfolioEnv:
    precos = random_prices(sessions=sessions, names=names, seed=seed)
    return PortfolioEnv(panel_of(precos), np.arange(names), 260, sessions - 1, EnvConfig(hold=hold, cost_bps=10.0))


# --- baselines --------------------------------------------------------------------------------------------------------


def test_equal_weight_pays_for_bringing_the_drifted_book_back() -> None:
    env = simulate(_env(), baselines.equal_weight)
    assert env.turnover[0] == pytest.approx(1.0)                    # from cash
    assert all(0.0 < giro < 0.5 for giro in env.turnover[1:])       # drift is traded, and it is not free
    assert all(pesos[:-1] == pytest.approx(0.25) for pesos in env.decisions)


def test_minimum_variance_matches_the_closed_form_when_no_bound_binds() -> None:
    cov = np.array([[0.04, 0.01], [0.01, 0.09]])
    inversa = np.linalg.solve(cov, np.ones(2))
    assert baselines.min_variance_weights(cov) == pytest.approx(inversa / inversa.sum(), abs=1e-6)


def test_minimum_variance_stays_long_only_when_the_closed_form_would_short() -> None:
    cov = np.array([[0.04, 0.058], [0.058, 0.09]])                  # correlation 0.97: the free solution shorts B
    assert baselines.min_variance_weights(cov) == pytest.approx([1.0, 0.0], abs=1e-6)


def test_maximum_sharpe_matches_the_tangency_portfolio() -> None:
    mu, cov = np.array([0.10, 0.06]), np.array([[0.04, 0.0], [0.0, 0.01]])
    tangente = np.linalg.solve(cov, mu)
    assert baselines.max_sharpe_weights(mu, cov) == pytest.approx(tangente / tangente.sum(), abs=1e-5)


def test_maximum_sharpe_falls_back_to_minimum_variance_without_a_positive_mean() -> None:
    cov = np.array([[0.04, 0.01], [0.01, 0.09]])
    assert baselines.max_sharpe_weights(np.array([-0.1, -0.2]), cov) == pytest.approx(baselines.min_variance_weights(cov))


def test_the_optimized_rules_hold_only_tradeable_names_and_no_cash() -> None:
    precos = random_prices(sessions=400, names=4)
    painel = panel_of(precos, names=["N00", "N01", "N03"])
    env = simulate(PortfolioEnv(painel, np.arange(4), 260, 399, EnvConfig(hold=21)), baselines.min_variance)
    assert all(pesos[2] == 0.0 and pesos[-1] == 0.0 and pesos.sum() == pytest.approx(1.0) for pesos in env.decisions)


def test_a_rule_reads_only_returns_realized_before_its_decision() -> None:
    precos = random_prices(sessions=400, names=4)
    mexido = precos.copy()
    mexido.iloc[301:] *= np.linspace(1.5, 2.0, 99)[:, None]
    a = PortfolioEnv(panel_of(precos), np.arange(4), 300, 399, EnvConfig(hold=21))
    b = PortfolioEnv(panel_of(mexido), np.arange(4), 300, 399, EnvConfig(hold=21))
    assert baselines.max_sharpe(a) == pytest.approx(baselines.max_sharpe(b))


# --- agents -----------------------------------------------------------------------------------------------------------


def test_on_policy_agents_get_a_value_head_and_off_policy_agents_a_q_head() -> None:
    for algoritmo in ("ppo", "a2c"):
        assert set(agentes.policy_kwargs(algoritmo)["net_arch"]) == {"pi", "vf"}
        assert agentes.policy_kwargs(algoritmo)["log_std_init"] == agentes.LOG_STD_INIT
    for algoritmo in agentes.OFF_POLICY:                            # SAC included: it could never run before
        assert set(agentes.policy_kwargs(algoritmo)["net_arch"]) == {"pi", "qf"}
        assert "log_std_init" not in agentes.policy_kwargs(algoritmo)
        assert agentes.algorithm_kwargs(algoritmo, 1000)["buffer_size"] == agentes.REPLAY_BUFFER
    assert agentes.policy_kwargs("ppo_shared", width=6) == {"width": 6, "log_std_init": agentes.LOG_STD_INIT}
    with pytest.raises(ValueError):
        agentes.policy_kwargs("ppo_shared")                         # the shared network must know the row's width
    with pytest.raises(ValueError):
        agentes.policy_kwargs("dqn")


def test_ppo_runs_on_the_papers_table() -> None:
    kwargs = agentes.algorithm_kwargs("ppo", 10_000)
    assert (kwargs["n_steps"], kwargs["batch_size"], kwargs["n_epochs"]) == (756, 1260, 16)
    assert (kwargs["gamma"], kwargs["gae_lambda"], kwargs["clip_range"]) == (0.9, 0.9, 0.25)
    assert kwargs["learning_rate"](1.0) == pytest.approx(3e-4) and kwargs["learning_rate"](0.0) == pytest.approx(1e-5)
    assert agentes.algorithm_kwargs("ppo", 100)["n_steps"] == 100   # a smoke run still fills one rollout


def test_selection_keeps_the_seed_with_the_best_validation_reward() -> None:
    recompensas = {0: 1.0, 1: 3.0, 2: 3.0, 3: -1.0}
    escolha = agentes.select([0, 1, 2, 3], fit=lambda seed: f"model{seed}",
                             validation_reward=lambda modelo: recompensas[int(modelo[-1])])
    assert (escolha.seed, escolha.model) == (1, "model1")           # the lowest seed wins the tie
    assert escolha.validation_rewards == recompensas
    with pytest.raises(ValueError):
        agentes.select([0], fit=lambda seed: seed, validation_reward=lambda modelo: float("nan"))


@pytest.mark.parametrize("algoritmo", ["ppo", "sac"])
def test_an_agent_trains_and_trades_a_test_stretch(algoritmo: str) -> None:
    precos = random_prices(sessions=420, names=3)
    painel = panel_of(precos)

    def ambiente(start: int = 60, stop: int = 300) -> PortfolioEnv:
        return PortfolioEnv(painel, np.arange(3), start, stop, EnvConfig(hold=5, observation="compact"))

    modelo = agentes.train(algoritmo, ambiente, timesteps=120, seed=0, n_envs=1)
    env, recompensa = agentes.evaluate(modelo, ambiente(300, 419))
    assert len(env.values) == 120 and np.isfinite(recompensa) and env.value > 0


# --- cells and the reading --------------------------------------------------------------------------------------------


def _years_panel():
    datas = pd.bdate_range("2010-01-04", "2018-12-28")
    rng = np.random.default_rng(3)
    precos = pd.DataFrame(100.0 * np.cumprod(1.0 + rng.normal(0.0004, 0.015, size=(len(datas), 4)), axis=0),
                          index=datas, columns=list("ABCD"))
    return panel_of(precos, cdi=0.0004, history=20)


def test_a_rule_cell_covers_its_test_year_and_adjoins_the_next() -> None:
    painel = _years_panel()
    janelas = protocolo.janelas(painel.sessions, 2016, 2018)
    spec = executar.RunSpec(EnvConfig(hold=21))
    celulas = [executar.run_cell(painel, janela, "equal_weight", spec) for janela in janelas]
    assert [cell.year for cell in celulas] == [2016, 2017, 2018]
    assert celulas[0].sessions[-1] == celulas[1].sessions[0]        # the year-end return is earned, once
    assert celulas[0].values[0] == 1.0 and celulas[0].chosen_seed is None
    curva, giro = leitura.stitch(celulas)
    assert curva.index.is_monotonic_increasing and curva.index.is_unique
    assert curva.iloc[-1] == pytest.approx(np.prod([cell.values[-1] for cell in celulas]))
    assert len(giro) == sum(len(cell.turnover) for cell in celulas)


def test_stitching_refuses_a_missing_year() -> None:
    painel = _years_panel()
    janelas = protocolo.janelas(painel.sessions, 2016, 2018)
    spec = executar.RunSpec(EnvConfig(hold=21))
    celulas = [executar.run_cell(painel, janela, "equal_weight", spec) for janela in (janelas[0], janelas[2])]
    with pytest.raises(ValueError, match="missing"):
        leitura.stitch(celulas)


def test_a_cell_survives_the_round_trip_to_its_file(tmp_path) -> None:
    painel = _years_panel()
    janela = protocolo.janelas(painel.sessions, 2016, 2016)[0]
    spec = executar.RunSpec(EnvConfig(hold=21))
    cell = executar.run_cell(painel, janela, "min_variance", spec)
    caminho = executar.cell_path("r", "ppo/dsr", 2016, tmp_path)
    assert caminho.name == "ppo-dsr-2016.json"
    executar.write_cell(cell, spec, caminho)
    assert executar.read_cell(caminho) == cell and not list(tmp_path.rglob("*.tmp"))


def test_the_table_measures_every_strategy_on_the_shared_years_over_the_cdi() -> None:
    painel = _years_panel()
    janelas = protocolo.janelas(painel.sessions, 2016, 2018)
    spec = executar.RunSpec(EnvConfig(hold=21))
    celulas = {"equal_weight": [executar.run_cell(painel, j, "equal_weight", spec) for j in janelas],
               "min_variance": [executar.run_cell(painel, j, "min_variance", spec) for j in janelas[:2]]}
    curva, _ = leitura.stitch(celulas["equal_weight"][:2])
    tabela = leitura.table(celulas, painel, composite=curva * 7.0)   # a reference with the same returns
    assert set(tabela["years"]) == {2}
    referencia = tabela.loc["Composite (house engine, reference)"]
    assert referencia["sharpe"] == pytest.approx(tabela.loc["equal_weight", "sharpe"])
    assert referencia["turnover"] == 0.0


# --- metrics ----------------------------------------------------------------------------------------------------------


def test_the_sharpe_is_over_the_risk_free_rate_of_each_session() -> None:
    rng = np.random.default_rng(0)
    retornos = rng.normal(0.001, 0.01, 500)
    valores = np.r_[1.0, np.cumprod(1.0 + retornos)]
    sem, com = metrics(valores), metrics(valores, np.full(500, 0.0005))
    assert sem["sharpe"] == pytest.approx(retornos.mean() / retornos.std(ddof=1) * np.sqrt(252))
    assert com["sharpe"] == pytest.approx((retornos.mean() - 0.0005) / retornos.std(ddof=1) * np.sqrt(252))
    assert metrics(np.cumprod(np.full(300, 1.0005)), np.full(299, 0.0005))["sharpe"] == 0.0   # the CDI itself


def test_drawdown_cagr_and_turnover_are_in_house_units() -> None:
    valores = np.r_[np.linspace(1.0, 2.0, 127), np.linspace(2.0, 1.5, 126)[1:], 1.5]
    linha = metrics(valores, turnover=[1.0, 0.5, 0.5])
    assert linha["mdd"] == pytest.approx(-0.25)
    assert linha["cagr"] == pytest.approx(1.5 ** (252 / 252) - 1.0)
    assert linha["turnover"] == pytest.approx(2.0)                  # summed over one year
    with pytest.raises(ValueError):
        metrics([1.0, 1.1])
    with pytest.raises(ValueError):
        metrics(valores, risk_free=[0.0])
