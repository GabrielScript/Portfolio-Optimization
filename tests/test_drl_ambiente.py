from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from drl import ambiente


def _ambiente(sessoes: int = 60, *, cresce: float = 0.01, recompensa: str = "valor", custo_bps: float = 0.0,
              modo: str = "matriz") -> ambiente.AmbienteAcoes:
    datas = pd.bdate_range("2020-01-02", periods=sessoes)
    a = 10.0 * (1.0 + cresce) ** np.arange(sessoes)
    b = 20.0 * np.ones(sessoes)
    fechamentos = pd.DataFrame({"A": a, "B": b}, index=datas)
    elegiveis = pd.DataFrame(True, index=datas, columns=["A", "B"])
    return ambiente.AmbienteAcoes(fechamentos, elegiveis, 1, sessoes, recompensa=recompensa, custo_bps=custo_bps,
                                  modo=modo)


def test_all_cash_earns_nothing_and_pays_no_cost():
    env = _ambiente()
    obs, _ = env.reset()
    acao = np.array([-1e3, -1e3, 0.0])                 # softmax sends everything to the cash leg
    _, recompensa, _, _, info = env.step(acao)
    assert info["valor"] == pytest.approx(1_000_000.0, rel=1e-6)
    assert recompensa == pytest.approx(0.0, abs=1e-6)


def test_the_portfolio_earns_the_weighted_return_of_the_next_session():
    env = _ambiente(cresce=0.01)
    env.reset()
    _, recompensa, _, _, info = env.step(np.array([1e3, -1e3, -1e3]))   # all in A, which grows 1% a session
    assert recompensa == pytest.approx(0.01, rel=1e-6)
    assert info["valor"] == pytest.approx(1_000_000.0 * 1.01, rel=1e-9)


def test_a_cost_is_charged_on_the_turnover_only():
    env = _ambiente(custo_bps=10.0)
    env.reset()
    _, r1, _, _, info1 = env.step(np.array([1e3, -1e3, -1e3]))          # cash -> A: trade of 1.0, cost 0.10%
    assert info1["giro"] == pytest.approx(1.0, rel=1e-6)
    assert info1["custo"] == pytest.approx(0.001 * 1.0, rel=1e-6)
    _, r2, _, _, info2 = env.step(np.array([1e3, -1e3, -1e3]))          # A -> A: no trade on the held leg
    assert info2["giro"] < 1e-6
    assert r2 == pytest.approx(0.01, rel=1e-4)


def test_an_ineligible_name_never_receives_weight():
    datas = pd.bdate_range("2020-01-02", periods=30)
    fechamentos = pd.DataFrame({"A": np.linspace(10, 20, 30), "B": np.linspace(20, 10, 30)}, index=datas)
    elegiveis = pd.DataFrame(True, index=datas, columns=["A", "B"])
    elegiveis["B"] = False                                                   # B is never tradeable here
    env = ambiente.AmbienteAcoes(fechamentos, elegiveis, 1, 30, recompensa="valor")
    env.reset()
    env.step(np.array([0.0, 1e3, -1e3]))                                     # the action begs for B
    assert env.historico_pesos[-1][1] == 0.0
    assert env.historico_pesos[-1][0] > 0.999


def test_the_dsr_matches_the_hand_computed_recursion():
    d, a1, b1 = ambiente.dsr(0.01, 0.0, 0.0)                                 # no variance yet: reward is 0, state moves
    assert d == 0.0
    assert a1 == pytest.approx(0.01 * ambiente.ETA)
    assert b1 == pytest.approx(0.0001 * ambiente.ETA)
    d2, _, _ = ambiente.dsr(0.02, a1, b1)
    variancia = b1 - a1 * a1
    esperado = (b1 * (0.02 - a1) - 0.5 * a1 * (0.0004 - b1)) / variancia ** 1.5
    assert d2 == pytest.approx(esperado)


def test_a_name_that_stops_trading_does_not_leak_a_nan():
    datas = pd.bdate_range("2020-01-02", periods=40)
    a = np.full(40, 10.0)
    a[20:] = np.nan
    fechamentos = pd.DataFrame({"A": a, "B": np.full(40, 20.0)}, index=datas)
    elegiveis = pd.DataFrame(True, index=datas, columns=["A", "B"])
    elegiveis.loc[datas[20]:, "A"] = False
    env = ambiente.AmbienteAcoes(fechamentos, elegiveis, 1, 40, recompensa="valor")
    env.reset()
    for _ in range(38):
        _, recompensa, terminou, _, info = env.step(np.array([1e3, -1e3, -1e3]))
        assert np.isfinite(recompensa)
        if terminou:
            break
    assert np.isfinite(info["valor"])


def test_the_observation_never_looks_at_a_later_session():
    env = _ambiente()
    obs1, _ = env.reset()
    outro = _ambiente()
    outro.fechamentos.iloc[2:, 0] *= 3.0            # change every session after the first decision
    outro.transacoes = outro.fechamentos.to_numpy(dtype=float)
    outro.ret = outro.transacoes[1:] / outro.transacoes[:-1] - 1.0
    obs2, _ = outro.reset()
    np.testing.assert_allclose(obs1, obs2)


def test_an_unknown_reward_or_mode_is_rejected():
    with pytest.raises(ValueError):
        _ambiente(recompensa="marcowitz")
    with pytest.raises(ValueError):
        _ambiente(modo="tensor")


def test_the_last_session_of_the_tape_cannot_be_stepped():
    env = _ambiente(sessoes=20)                         # fim lands on the last row; the episode must stop before it
    env.reset()
    for _ in range(50):
        _, _, terminou, _, _ = env.step(np.array([1.0, 1.0, 1.0]))
        if terminou:
            break
    assert terminou
    assert env.t == env.fim


def test_a_monthly_rebalancing_rule_holds_the_same_weights_between_decisions():
    env = _ambiente(sessoes=60)
    env.rebalancear_a_cada = 5
    env.reset()
    env.step(np.array([5.0, -5.0, 0.0]))            # decision: heavy on A
    w1 = env.historico_pesos[-1].copy()
    for _ in range(4):                              # the next four sessions must not trade
        env.step(np.array([-5.0, 5.0, 0.0]))        # the action begs to switch: it must be ignored
        assert env.giro[-1] < 1e-6
    env.step(np.array([-5.0, 5.0, 0.0]))            # the fifth session is a decision again
    assert env.giro[-1] > 0.01


def test_the_observation_never_contains_the_return_the_next_action_earns():
    """A look-ahead test in both observation modes: the last session's price is the future at the decision point, so
    changing it must not change the observation the agent acts on."""
    datas = pd.bdate_range("2020-01-02", periods=60)
    base = pd.DataFrame({"A": 10.0 * np.ones(60), "B": 20.0 * np.ones(60)}, index=datas)
    futuro = base.copy()
    futuro.iloc[-1] = [99.0, 5.0]                      # only the return into the last session changes
    elegiveis = pd.DataFrame(True, index=datas, columns=["A", "B"])
    for modo in ("matriz", "compacto"):
        a = ambiente.AmbienteAcoes(base, elegiveis, 58, 60, modo=modo)
        b = ambiente.AmbienteAcoes(futuro, elegiveis, 58, 60, modo=modo)
        assert np.allclose(a._observacao(), b._observacao()), f"look-ahead in modo={modo}"
