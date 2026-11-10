"""Tests of scripts.qnn_v4_avaliar: decision dates, rank IC, arm naming and the pre-registered gates."""

from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.avaliar_universo_dinamico import CFG_BT
from scripts.qnn_v4_avaliar import (MICHAUD, SUFIXOS, T_SIGNAL, base_of, decision_dates, rank_ic, screened_arms,
                                    signal_gate)


def _prices(sessions: int) -> pd.DataFrame:
    datas = pd.bdate_range("2005-01-03", periods=sessions)
    return pd.DataFrame({"AAAA3": np.linspace(10, 20, sessions), "BOVA11": 1.0}, index=datas)


def test_decision_dates_are_the_last_training_session_of_each_rebalance_and_the_one_after() -> None:
    precos = _prices(CFG_BT.janela_treino + 2 * CFG_BT.janela_teste + 5)
    datas = decision_dates(precos, fases=(0, 21))
    retornos = precos.index[1:]                                  # the engine's return sessions
    for fase in (0, 21):
        sessoes = retornos[fase:]
        for posicao in range(CFG_BT.janela_treino, len(sessoes), CFG_BT.janela_teste):
            assert sessoes[posicao - 1] in datas                 # the training window's last row
            assert sessoes[posicao] in datas                     # the session a one-day freeze would read
    assert datas == sorted(set(datas))


def test_rank_ic_is_one_for_a_perfect_ranking_and_skips_unrealized_dates() -> None:
    nomes = [f"N{i:02d}" for i in range(12)]
    datas = pd.bdate_range("2020-01-01", periods=3)
    p = pd.DataFrame(np.tile(np.linspace(0.4, 0.6, 12), (3, 1)), index=datas, columns=nomes)
    futuro = pd.DataFrame(np.tile(np.linspace(-0.1, 0.1, 12), (3, 1)), index=datas, columns=nomes)
    futuro.iloc[1] = -futuro.iloc[1]
    futuro.iloc[2] = np.nan                                      # not realized yet
    ic = rank_ic(p, futuro)
    assert list(ic.index) == list(datas[:2])
    assert np.isclose(ic.iloc[0], 1.0) and np.isclose(ic.iloc[1], -1.0)


def test_base_of_reads_the_method_behind_a_screened_arm() -> None:
    assert base_of("HRP" + SUFIXOS["qnn"]) == "HRP"
    assert base_of("Kelly (λ = 2)" + SUFIXOS["shuffled"]) == "Kelly (λ = 2)"
    assert base_of("HRP") is None
    assert base_of("Tilt QNN v4 (Composite 15, Meio-Kelly)") is None


class _Half:
    def predict(self, date: pd.Timestamp) -> pd.Series:
        return pd.Series(dtype=float)


def test_slow_methods_get_only_the_qnn_screen() -> None:
    def base(treino: pd.DataFrame) -> pd.Series:
        return pd.Series(dtype=float)

    fontes = {chave: _Half() for chave in SUFIXOS}
    bracos = screened_arms({"HRP": base, MICHAUD: base}, fontes)
    assert set(bracos) == {"HRP", *(f"HRP{s}" for s in SUFIXOS.values()), MICHAUD, MICHAUD + SUFIXOS["qnn"]}


def _table(cagr_qnn: float, t: float, cagr_logistic: float = 0.10, cagr_shuffled: float = 0.10) -> pd.DataFrame:
    linhas = {"X": (0.12, 0.5, np.nan), "X + QNN": (cagr_qnn, 0.6, t), "X + logistic": (cagr_logistic, 0.5, 1.0),
              "X + shuffled": (cagr_shuffled, 0.5, 0.0), "X + anti": (0.08, 0.4, -1.0)}
    return pd.DataFrame(linhas, index=["cagr", "sharpe", "t_base"]).T


def test_signal_gate_needs_every_phase_above_the_bonferroni_t() -> None:
    passa = {fase: _table(0.16, T_SIGNAL + 0.1) for fase in (0, 21, 42)}
    assert signal_gate(passa, "X") == (True, True)
    uma_fraca = {**passa, 21: _table(0.16, T_SIGNAL - 0.1)}
    assert signal_gate(uma_fraca, "X") == (False, False)


def test_signal_gate_fails_when_a_placebo_does_as_well() -> None:
    tabelas = {fase: _table(0.16, 3.5, cagr_shuffled=0.17) for fase in (0, 21, 42)}
    assert signal_gate(tabelas, "X") == (False, False)


def test_quantum_gate_fails_when_the_logistic_screen_does_better() -> None:
    tabelas = {fase: _table(0.16, 3.5, cagr_logistic=0.17) for fase in (0, 21, 42)}
    assert signal_gate(tabelas, "X") == (True, False)
