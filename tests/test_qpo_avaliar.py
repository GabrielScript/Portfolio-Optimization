"""Gates and QAOA result loading of the QPO evaluation (scripts/qpo_avaliar.py)."""

import json

import pandas as pd
import pytest

from scripts.qpo_avaliar import load_qaoa_picks, solver_gate, strategy_gate


def test_strategy_gate_needs_both_references_at_095_in_every_phase():
    assert strategy_gate([0.97, 0.96, 0.95], [0.99, 0.99, 0.98]) is True
    assert strategy_gate([0.97, 0.94, 0.99], [0.99, 0.99, 0.99]) is False
    assert strategy_gate([0.99, 0.99, 0.99], [0.99, 0.949, 0.99]) is False


def test_solver_gate_needs_80_percent_matches_and_mean_ratio_098():
    assert solver_gate([True] * 8 + [False] * 2, [1.0] * 8 + [0.95, 0.95]) is True
    assert solver_gate([True] * 7 + [False] * 3, [1.0] * 10) is False
    assert solver_gate([True] * 10, [0.97] * 10) is False


def _resultado(pasta, fase, data, p, sha, tickers):
    destino = pasta / f"fase{fase}"
    destino.mkdir(parents=True, exist_ok=True)
    (destino / f"{data}_p{p}.json").write_text(json.dumps({"sha256": sha, "tickers": tickers, "match": True,
                                                           "ratio": 1.0}), encoding="utf-8")


def test_load_qaoa_picks_checks_each_hash_against_sha256sums(tmp_path):
    (tmp_path / "qubos").mkdir()
    (tmp_path / "qubos" / "SHA256SUMS").write_text("aa  fase0/2010-02-09.npz\nbb  fase0/2010-05-12.npz\n",
                                                  encoding="utf-8")
    _resultado(tmp_path / "res", 0, "2010-02-09", 2, "aa", ["PETR4", "VALE3"])
    _resultado(tmp_path / "res", 0, "2010-05-12", 2, "bb", ["ITUB4", "VALE3"])
    escolhas = load_qaoa_picks(tmp_path / "res", tmp_path / "qubos", fase=0, p=2)
    assert escolhas == {pd.Timestamp("2010-02-09"): ("PETR4", "VALE3"), pd.Timestamp("2010-05-12"): ("ITUB4", "VALE3")}
    _resultado(tmp_path / "res", 0, "2010-05-12", 2, "zz", ["ITUB4", "VALE3"])
    with pytest.raises(ValueError, match="2010-05-12"):
        load_qaoa_picks(tmp_path / "res", tmp_path / "qubos", fase=0, p=2)


def test_choose_p_takes_the_best_mean_ratio_and_the_smaller_p_on_a_tie():
    from scripts.qpo_avaliar import choose_p
    assert choose_p({1: [0.97, 0.99], 2: [1.0, 0.99], 3: [0.99, 0.99]}) == 2
    assert choose_p({1: [1.0], 2: [1.0], 3: [0.99]}) == 1
